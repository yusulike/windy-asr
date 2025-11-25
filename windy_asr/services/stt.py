"""
Real-time Speech-to-Text Client Module
Handles audio processing and transcription using RealtimeSTT
"""

import logging
import threading
import time
import numpy as np
from scipy import signal

logger = logging.getLogger('windy_asr.stt')

REALTIMESTT_LOG_LEVEL = logging.INFO

# Check RealtimeSTT availability
try:
    from RealtimeSTT import AudioToTextRecorder
    REALTIMESTT_AVAILABLE = True
except ImportError:
    REALTIMESTT_AVAILABLE = False
    logger.warning("RealtimeSTT library not available. RealtimeSTT transcription will be disabled.")
    logger.info("Install with: pip install RealtimeSTT")


class RealtimeSTTClient:
    """RealtimeSTT 클라이언트 - 로컬 STT 엔진"""
    
    def __init__(self, model='tiny', language='ja', device='cuda', 
                 enable_realtime_transcription=False, post_speech_silence_duration=0.1,
                 silero_sensitivity=0.3, webrtc_sensitivity=3,
                 beam_size=5, batch_size=16, min_length_of_recording=0.5,
                 use_extended_logging=False,
                 silero_use_onnx=False,
                 silero_deactivity_detection=False
                 ):
        self.model = model
        self.language = language
        self.device = device
        self.enable_realtime_transcription = enable_realtime_transcription
        self.post_speech_silence_duration = post_speech_silence_duration
        self.silero_sensitivity = silero_sensitivity
        self.webrtc_sensitivity = webrtc_sensitivity
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.min_length_of_recording = min_length_of_recording
        self.use_extended_logging = use_extended_logging
        self.silero_use_onnx = silero_use_onnx
        self.silero_deactivity_detection = silero_deactivity_detection
        self.recorder = None
        self.callback = None
        self.realtime_callback = None
        self.connected = False
        self.stop_event = threading.Event()
        self.audio_buffer = np.array([], dtype=np.int16)
    
    def set_callback(self, callback):
        """결과를 받을 콜백 함수 설정"""
        self.callback = callback
    
    def set_realtime_callback(self, callback):
        """실시간 트랜스크립션 콜백 설정"""
        self.realtime_callback = callback
        
    def connect(self):
        """RealtimeSTT 초기화"""
        if not REALTIMESTT_AVAILABLE:
            logger.error("RealtimeSTT library not installed")
            return False
            
        try:
            # RealtimeSTT 초기화 파라미터
            recorder_params = {
                'use_microphone': False,
                'model': self.model,
                'language': self.language,
                'device': self.device,
                'spinner': False,
                'post_speech_silence_duration': self.post_speech_silence_duration,
                'min_length_of_recording': self.min_length_of_recording,
                'silero_sensitivity': self.silero_sensitivity,
                'webrtc_sensitivity': self.webrtc_sensitivity,
                'batch_size': self.batch_size,
                'beam_size': self.beam_size,
                'handle_buffer_overflow': True,
                'no_log_file': True,
                'level': REALTIMESTT_LOG_LEVEL,
                'enable_realtime_transcription': self.enable_realtime_transcription,
                'use_main_model_for_realtime': False,
                'realtime_model_type': 'tiny',
                'realtime_processing_pause': 0.2,
                'beam_size_realtime': 3,
                'on_realtime_transcription_update': self._on_realtime_update,
                'on_vad_stop': self.on_vad_stop,
                'on_vad_start': self.on_vad_start,
                'on_vad_detect_start': self.on_vad_detect_start,
                'on_vad_detect_stop': self.on_vad_detect_stop,
                'use_extended_logging': self.use_extended_logging,
                'silero_use_onnx': self.silero_use_onnx,
                'silero_deactivity_detection': self.silero_deactivity_detection,
            }
            
            # AudioToTextRecorder 초기화
            self.recorder = AudioToTextRecorder(**recorder_params)
            self.connected = True

            logger.info("RealtimeSTT initialized (model=%s, language=%s, device=%s)", self.model, self.language or 'auto', self.device)
            logger.debug(
                "Realtime transcription=%s, VAD sensitivities: Silero=%s WebRTC=%s",
                self.enable_realtime_transcription,
                self.silero_sensitivity,
                self.webrtc_sensitivity,
            )

            # 모든 파라미터 출력
            for key, value in recorder_params.items():
                logger.debug("Recorder param %s=%s", key, value)
                
            
            # 트랜스크립션 스레드 시작
            threading.Thread(target=self._transcription_thread, daemon=True).start()
            
            return True
            
        except Exception as e:
            logger.exception("Failed to initialize RealtimeSTT")
            self.connected = False
            return False
    
    def disconnect(self):
        """연결 종료"""
        self.stop_event.set()
        self.connected = False
        if self.recorder:
            try:
                self.recorder.shutdown()
            except:
                pass
        logger.info("RealtimeSTT disconnected")
    
    def send_audio(self, audio_data):
        """오디오 데이터를 RealtimeSTT에 feed"""
        if not self.connected or self.stop_event.is_set():
            return
            
        try:
            # 1. 스테레오 -> 모노 변환
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # 2. 리샘플링 (44.1kHz -> 16kHz)
            if 44100 != 16000:
                num_samples = int(len(audio_data) * 16000 / 44100)
                audio_data = signal.resample(audio_data, num_samples)
            
            # 3. float32 -> int16 변환
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # 4. RealtimeSTT에 feed (16-bit mono PCM)
            pcm_bytes = audio_int16.tobytes()
            self.recorder.feed_audio(pcm_bytes)
            
        except Exception as e:
            logger.error("Audio processing error: %s", e)
    
    def _on_realtime_update(self, text):
        """실시간 트랜스크립션 업데이트 콜백"""
        # RealtimeSTT에서 진행중인 텍스트를 실시간으로 받음
        if text and self.realtime_callback:
            # is_final=False로 진행중 상태임을 표시
            self.realtime_callback(text, is_final=False)

    def on_vad_start(self):
        """VAD 시작 콜백"""
        logger.info("RealtimeSTT VAD: 목소리 발생 시작 (on_vad_start)")
    
    def on_vad_stop(self):
        """VAD 종료 콜백"""
        logger.info("RealtimeSTT VAD: 목소리 발생 종료 (on_vad_stop)")
    
    def on_vad_detect_start(self):
        """VAD 감지 시작 콜백"""
        logger.info("RealtimeSTT VAD: 발생 감지 대기 시작 (on_vad_detect_start)")
    
    def on_vad_detect_stop(self):
        """VAD 감지 종료 콜백"""
        logger.info("RealtimeSTT VAD: 발생 감지 대기 종료 (on_vad_detect_stop)")
    
    def _transcription_thread(self):
        """트랜스크립션 결과 수신 스레드"""
        try:
            while not self.stop_event.is_set():
                try:
                    # text() 메서드는 blocking call
                    # 새로운 트랜스크립션 결과를 가져옴
                    logger.debug("RealtimeSTT waiting for transcription result")
                    text = self.recorder.text()
                    logger.debug("RealtimeSTT transcription received: %s", text)
                    
                    if text and self.callback:
                        # 최종 결과 콜백
                        logger.debug("RealtimeSTT final transcription: %s", text)
                        self.callback(text, is_final=True)
                    
                    time.sleep(0.1)  # CPU 사용량 조절
                    
                except Exception as e:
                    if not self.stop_event.is_set():
                        logger.error("Transcription error: %s", e)
                    break
                    
        except Exception as e:
            logger.error("Transcription thread error: %s", e)
        finally:
            logger.info("RealtimeSTT transcription thread terminated")

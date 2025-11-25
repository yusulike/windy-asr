"""
Windy ASR Client
실시간 음성 인식 클라이언트 (RealtimeSTT + Ollama)

Usage:
    python simple-sensevoice-client.py
"""

import logging
import os
import signal
import tkinter as tk
import customtkinter as ctk
import soundcard as sc
import soundfile as sf
import numpy as np
import io
import json
import threading
import queue
import time
from websocket import create_connection, ABNF
from datetime import datetime
from collections import deque
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageTk

# Basic logging configuration (aligns with external component format)
LOG_LEVEL = os.environ.get('WINDY_ASR_LOG_LEVEL', 'INFO').upper()
LOG_PREFIX = os.environ.get('WINDY_ASR_LOG_PREFIX', 'WindyASR')
LOG_FORMAT = os.environ.get(
    'WINDY_ASR_LOG_FORMAT',
    f"{LOG_PREFIX}: %(name)s - %(levelname)s - %(message)s"
)
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT
)
logging.getLogger("RealtimeSTT.safepipe").disabled = True

REALTIMESTT_LOG_LEVEL = logging.INFO


def _align_external_logger(logger_name):
    """Ensure external library loggers bubble up to our root formatter."""
    ext_logger = logging.getLogger(logger_name)
    # Remove handlers those libraries may have attached to avoid duplicate output
    for handler in list(ext_logger.handlers):
        ext_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    ext_logger.handlers.clear()
    ext_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    ext_logger.propagate = True


logger = logging.getLogger('windy_asr')

# _align_external_logger('realtimestt')

# ollama import
try:
    from ollama import chat, ChatResponse
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama library not available. Translation will be disabled.")
    logger.info("Install with: pip install ollama")

# RealtimeSTT import
try:
    from RealtimeSTT import AudioToTextRecorder
    REALTIMESTT_AVAILABLE = True
except ImportError:
    REALTIMESTT_AVAILABLE = False
    logger.warning("RealtimeSTT library not available. RealtimeSTT transcription will be disabled.")
    logger.info("Install with: pip install RealtimeSTT")

DEFAULT_SETTINGS = {
    'sample_rate': 44100,  # 녹음 샘플레이트
    'chunk_duration': 0.1,  # 녹음 청크 길이 (초)
    # RealtimeSTT 설정
    'realtimestt_model': 'large-v3',  # tiny, base, small, medium, large
    'realtimestt_device': 'cuda',  # 'cuda' 또는 'cpu'
    'realtimestt_post_speech_silence_duration': 0.1,  # 말 끝난 후 침묵 시간 (초)
    'realtimestt_silero_sensitivity': 0.8,  # Silero VAD 민감도 (0.0-1.0)
    'realtimestt_webrtc_sensitivity': 3,  # WebRTC VAD 민감도 (0-3, 3이 가장 aggressive)
    'realtimestt_beam_size': 5,  # 디코딩 beam size
    'realtimestt_batch_size': 16,  # 배치 처리 크기
    'realtimestt_min_length_of_recording': 0.1,  # 최소 녹음 시간 (초)
    # 번역 설정 (Ollama 전용)
    'translation_enabled': True,
    'ollama_model': 'yanolja-rosetta',  # Ollama 모델명
    # 언어 설정 (기본값: ja)
    'language': 'ja',
    # 'silero_use_onnx': True,
    # 'silero_deactivity_detection': True,
    'use_extended_logging': False,
}

CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(CONFIG_DIR, 'settings.json')


def load_settings():
    """settings.json에서 설정을 로드하거나 기본값을 생성"""
    global SETTINGS
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as fp:
                loaded = json.load(fp)
                SETTINGS = {**DEFAULT_SETTINGS, **loaded}
                logger.info("Settings loaded from %s", CONFIG_FILE)
                return
        except Exception as exc:
            logger.warning("Failed to load settings: %s", exc)

    SETTINGS = DEFAULT_SETTINGS.copy()
    save_settings()
    logger.info("Default settings saved to %s", CONFIG_FILE)


def save_settings():
    """현재 SETTINGS를 settings.json에 저장"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as fp:
            json.dump(SETTINGS, fp, indent=2, ensure_ascii=False)
            logger.debug("Settings persisted to %s", CONFIG_FILE)
    except Exception as exc:
        logger.error("Failed to save settings: %s", exc)


SETTINGS = {}
load_settings()


def get_available_ollama_models():
    """Retrieve installed Ollama models (best-effort)."""
    if not OLLAMA_AVAILABLE:
        logger.debug("Ollama library unavailable; skipping model discovery")
        return []

    try:
        from ollama import list as ollama_list

        models_response = ollama_list()
        # 모델 이름 끝에 :latest는 제거
        model_names = [m.get('model', '').replace(':latest', '') for m in models_response.get('models', []) if m.get('model')]
        return sorted(set(model_names))
    except Exception as exc:
        logger.warning("Failed to list Ollama models: %s", exc)
        return []


class TranslationService:
    """번역 서비스 (Ollama)"""
    
    def __init__(self, backend='ollama', ollama_model='yanolja-rosetta'):
        self.backend = backend
        self.enabled = False
        self.model_name = ollama_model
        
        if backend == 'ollama':
            self._init_ollama(ollama_model)
    
    def _init_ollama(self, model_name):
        """Ollama 초기화"""
        if not OLLAMA_AVAILABLE:
            logger.error("ollama library not installed")
            return
            
        try:
            model_names = get_available_ollama_models()

            if any(model_name in name for name in model_names):
                self.enabled = True
                logger.info("Ollama translation service initialized with model: %s", model_name)
            else:
                logger.warning("Model '%s' not found in Ollama. Available models: %s", model_name, model_names)
                logger.info("Run: ollama pull %s", model_name)
        except Exception as e:
            logger.error("Failed to initialize Ollama: %s", e)
            logger.info("Make sure Ollama is running (ollama serve)")
    
    def translate(self, text):
        """텍스트를 번역"""
        if not self.enabled or not text:
            return text
        
        try:
            return self._translate_with_ollama(text)
        except Exception as e:
            logger.error("Translation error: %s", e)
            return text
    
    def _translate_with_ollama(self, text):
        """Ollama를 이용한 번역"""
        response: ChatResponse = chat(
            model=self.model_name,
            messages=[
                {
                    'role': 'user',
                    'content': text,
                },
            ],
            options={
                'temperature': 0.1,
            }
        )
        translated = response.message.content.strip()
        logger.debug("Ollama translated: %s -> %s", text, translated)
        return translated
    
    def is_available(self):
        """번역 서비스 사용 가능 여부"""
        return self.enabled

import numpy as np
from scipy import signal
import queue


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
            # print(f"⏳ RealtimeSTT realtime: {text}")

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


class AudioLevelDisplay:
    """오디오 레벨(좌/우 채널) 시각화 위젯"""

    def __init__(self, parent, width=400, height=60):
        self.width = width
        self.height = height

        self.frame = ctk.CTkFrame(parent, width=width, height=height, fg_color="#1a1a1a")

        ctk.CTkLabel(self.frame, text="L:", text_color="#888888").pack(side="left", padx=5, pady=5)
        self.progress_l = ctk.CTkProgressBar(self.frame, width=150, progress_color="#00ff00")
        self.progress_l.pack(side="left", padx=3, pady=5)
        self.label_l = ctk.CTkLabel(self.frame, text="0%", text_color="#888888", width=40)
        self.label_l.pack(side="left", padx=3, pady=5)

        ctk.CTkLabel(self.frame, text="R:", text_color="#888888").pack(side="left", padx=5, pady=5)
        self.progress_r = ctk.CTkProgressBar(self.frame, width=150, progress_color="#00ff00")
        self.progress_r.pack(side="left", padx=3, pady=5)
        self.label_r = ctk.CTkLabel(self.frame, text="0%", text_color="#888888", width=40)
        self.label_r.pack(side="left", padx=3, pady=5)

    def get_widget(self):
        return self.frame

    def update_level(self, audio_data):
        try:
            if audio_data.ndim > 1:
                level_l = np.sqrt(np.mean(audio_data[:, 0] ** 2))
                if audio_data.shape[1] > 1:
                    level_r = np.sqrt(np.mean(audio_data[:, 1] ** 2))
                else:
                    level_r = level_l
            else:
                level_l = level_r = np.sqrt(np.mean(audio_data ** 2))

            level_l = np.log10(max(level_l, 0.001)) + 2
            level_r = np.log10(max(level_r, 0.001)) + 2

            lvl_l_display = max(0, min(level_l / 4, 1.0))
            lvl_r_display = max(0, min(level_r / 4, 1.0))

            self.progress_l.set(lvl_l_display)
            self.progress_r.set(lvl_r_display)
            self.label_l.configure(text=f"{int(lvl_l_display * 100)}%")
            self.label_r.configure(text=f"{int(lvl_r_display * 100)}%")
        except Exception as exc:
            logger.debug("Audio level update skipped: %s", exc)

    def reset(self):
        self.progress_l.set(0)
        self.progress_r.set(0)
        self.label_l.configure(text="0%")
        self.label_r.configure(text="0%")


class SubtitleOverlay(tk.Toplevel):
    """자막 오버레이 창 - Canvas 기반 그림자 효과 + 스크롤 애니메이션"""
    
    def __init__(self, master=None, compact=False):
        super().__init__(master)
        
        # Compact 모드 플래그
        self.compact_mode = compact
        
        # 창 설정
        self.title("Subtitle Overlay")
        self.attributes('-topmost', True)
        self.attributes('-alpha', 0.5)
        self.overrideredirect(True)

        # 윈도우 배경을 특정 색으로 통일한 뒤 해당 색상을 투명 처리하여 둥근 모서리가 제대로 보이도록 함
        self.transparent_color = "#010101"
        self.configure(bg=self.transparent_color)
        try:
            self.wm_attributes('-transparentcolor', self.transparent_color)
        except tk.TclError:
            logger.warning("Transparent color attribute not supported on this platform")
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        # Compact 모드에서는 더 작은 크기
        if self.compact_mode:
            self.window_width = 1600
            self.window_height = 120  # 1줄만 표시
        else:
            self.window_width = 1900
            self.window_height = 300  # 높이 증가 (3줄 자막에 맞춤)
        x = (screen_width - self.window_width) // 2
        y = screen_height - self.window_height - 100
        
        self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        # 메인 Canvas
        self.canvas = tk.Canvas(
            self,
            width=self.window_width,
            height=self.window_height,
            bg=self.transparent_color,
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True)
        
        # 둥근 배경 그리기
        self.bg_radius = 60
        self.bg_rect = self._create_rounded_rectangle(
            0, 0, self.window_width, self.window_height,
            radius=self.bg_radius,
            fill='black'
        )
        
        # 드래그 및 메뉴 바인딩
        self.canvas.bind('<Button-1>', self.start_move)
        self.canvas.bind('<B1-Motion>', self.on_move)
        self.canvas.bind('<Button-3>', self.show_menu)
        
        # 자막 데이터
        if self.compact_mode:
            self.subtitles = deque(maxlen=1)  # Compact 모드: 최근 1개만
        else:
            self.subtitles = deque(maxlen=2)  # 완성된 자막 2개
        
        self.current_text = ""  # 진행중 텍스트
        self.animation_counter = 0  # 진행 중 인디케이터 애니메이션용 카운터
        
        # Canvas 아이템 ID들
        self.canvas_items = []
        
        # 애니메이션 관련
        self.animation_running = False
        self.animation_y_offset = 0
        self.target_y_offset = 0
        
        # 폰트 캐싱
        self.fonts = {}
        self._init_fonts()
        
        # 우클릭 메뉴
        self.menu = tk.Menu(self, tearoff=0)
        self.menu.add_command(label="투명도 높이기", command=lambda: self.change_alpha(0.1))
        self.menu.add_command(label="투명도 낮추기", command=lambda: self.change_alpha(-0.1))
        self.menu.add_separator()
        mode_label = "📋 일반 모드로" if self.compact_mode else "📋 컴팩트 모드로"
        self.menu.add_command(label=mode_label, command=self.toggle_compact_mode)
        self.menu.add_separator()
        self.menu.add_command(label="자막 지우기", command=self.clear_subtitles)
        self.menu.add_separator()
        self.menu.add_command(label="닫기", command=self.withdraw)
        
        # 드래그용 변수
        self._drag_start_x = 0
        self._drag_start_y = 0
        
        # 초기 렌더링
        self.render_subtitles()
    
    def _init_fonts(self):
        """폰트 초기화 (PIL)"""
        try:
            # 로컬 NanumSquareB.ttf 사용 (현재 디렉토리 기준)
            self.fonts['large'] = ImageFont.truetype("NanumSquareB.ttf", 45)
            self.fonts['medium'] = ImageFont.truetype("NanumSquareB.ttf", 40)
            self.fonts['small'] = ImageFont.truetype("malgunbd.ttf", 32)

        except:
            try:
                # 절대 경로로 시도 (파일이 다른 위치에 있을 경우)
                self.fonts['large'] = ImageFont.truetype("e:/myWork/WhisperLive/streaming-sensevoice/NanumSquareB.ttf", 42)
                self.fonts['medium'] = ImageFont.truetype("e:/myWork/WhisperLive/streaming-sensevoice/NanumSquareB.ttf", 38 )
            except:
                try:
                    # 맑은 고딕 볼드 폴백
                    self.fonts['large'] = ImageFont.truetype("malgunbd.ttf", 38)
                    self.fonts['medium'] = ImageFont.truetype("malgunbd.ttf", 34)
                except:
                    try:
                        # 일반 맑은 고딕
                        self.fonts['large'] = ImageFont.truetype("malgun.ttf", 42)
                        self.fonts['medium'] = ImageFont.truetype("malgun.ttf", 38)
                    except:
                        try:
                            # Arial Bold
                            self.fonts['large'] = ImageFont.truetype("arialbd.ttf", 38)
                            self.fonts['medium'] = ImageFont.truetype("arialbd.ttf", 34)
                        except:
                            # 기본 폰트
                            self.fonts['large'] = ImageFont.load_default()
                            self.fonts['medium'] = ImageFont.load_default()
    
    def _create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        """둥근 사각형 그리기"""
        points = [
            x1 + radius, y1,
            x1 + radius, y1,
            x2 - radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1 + radius,
            x1, y1
        ]
        return self.canvas.create_polygon(points, smooth=True, **kwargs)
    
    def _create_text_image(self, text, font, text_color, shadow_color, shadow_offset=3, shadow_blur=2):
        """그림자 효과가 있는 텍스트 이미지 생성 (PIL)"""
        if not text:
            return None
        
        # 텍스트 크기 측정
        dummy_img = Image.new('RGBA', (1, 1), (0, 0, 0, 0))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        # 여러 줄로 나누기 (wraplength 고려)
        max_width = self.window_width - 100
        lines = self._wrap_text(text, font, max_width)
        
        # 각 줄의 bbox 계산
        line_heights = []
        line_widths = []
        for line in lines:
            bbox = dummy_draw.textbbox((0, 0), line, font=font)
            line_widths.append(bbox[2] - bbox[0])
            line_heights.append(bbox[3] - bbox[1])
        
        total_width = max(line_widths) if line_widths else 0
        total_height = sum(line_heights) + (len(lines) - 1) * 5  # 줄 간격 5px
        
        # 이미지 생성 (여유 공간 포함)
        img_width = total_width + shadow_offset * 2 + 40
        img_height = total_height + shadow_offset * 2 + 40
        
        # 투명 배경
        img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        
        # 그림자 레이어
        shadow_img = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        
        # 그림자 텍스트 그리기
        y_offset = 20 + shadow_offset
        for line in lines:
            shadow_draw.text(
                (20 + shadow_offset, y_offset),
                line,
                font=font,
                fill=shadow_color
            )
            y_offset += line_heights[lines.index(line)] + 5
        
        # 그림자 블러
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
        
        # 그림자 합성
        img = Image.alpha_composite(img, shadow_img)
        
        # 메인 텍스트 그리기
        main_draw = ImageDraw.Draw(img)
        y_offset = 20
        for line in lines:
            main_draw.text(
                (20, y_offset),
                line,
                font=font,
                fill=text_color
            )
            y_offset += line_heights[lines.index(line)] + 5
        
        return img, img_width, img_height
    
    def _wrap_text(self, text, font, max_width):
        """텍스트를 최대 너비에 맞게 줄바꿈"""
        dummy_img = Image.new('RGBA', (1, 1))
        dummy_draw = ImageDraw.Draw(dummy_img)
        
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + (" " if current_line else "") + word
            bbox = dummy_draw.textbbox((0, 0), test_line, font=font)
            width = bbox[2] - bbox[0]
            
            if width <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines if lines else [text]
    
    def render_subtitles(self):
        """자막 렌더링 (모든 Canvas 아이템 다시 그리기)"""
        # 기존 자막 아이템 제거 (배경 제외)
        for item in self.canvas_items:
            self.canvas.delete(item)
        self.canvas_items.clear()
        
        subtitles_list = list(self.subtitles)
        
        if self.compact_mode:
            # Compact 모드: 최근 1개만 표시 (가운데)
            if subtitles_list:
                translated, original = subtitles_list[-1]  # 가장 최근 항목
                display_text = translated if translated else original
                
                result = self._create_text_image(
                    display_text,
                    self.fonts['large'],
                    "white",
                    "#000000",
                    shadow_offset=0,
                    shadow_blur=0
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # 중앙에 배치
                    item_id = self.canvas.create_image(
                        30, 10,
                        image=photo,
                        anchor='nw'
                    )
                    
                    self.canvas.itemconfig(item_id, tags="subtitle_latest")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    self.canvas_items.append(item_id)
            
            # Compact 모드에서 진행중 텍스트가 있으면 인디케이터 표시
            if self.current_text:
                # 점 애니메이션 인디케이터 (진행 중임을 표시)
                dots_count = (self.animation_counter % 3) + 1
                indicator_text = "●" * dots_count + "○" * (3 - dots_count)
                
                result = self._create_text_image(
                    indicator_text,
                    self.fonts['small'],
                    "#666666",
                    "#333333",
                    shadow_offset=0,
                    shadow_blur=0
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # 우측 하단에 배치
                    item_id = self.canvas.create_image(
                        self.window_width - img_w - 20, self.window_height - img_h - 5,
                        image=photo,
                        anchor='nw'
                    )
                    
                    self.canvas.itemconfig(item_id, tags="indicator")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    self.canvas_items.append(item_id)
        else:
            # 일반 모드: 원래대로 동작
            # Y 위치 계산
            y_positions = [
                10 + self.animation_y_offset,   # 가장 오래된 자막 (위)
                100 + self.animation_y_offset,  # 최근 자막 (중간)
                200  # 진행중 자막 (하단, 고정)
            ]
            
            # 완성된 자막 렌더링 (오래된 것부터)
            for i, (translated, original) in enumerate(subtitles_list):
                display_text = translated if translated else original
                
                # 색상 및 폰트 설정
                if i == 0:  # 가장 오래된 자막
                    text_color = "#999999"
                    shadow_color = "#333333"
                    font = self.fonts['medium']
                else:  # 최근 자막
                    text_color = "white"
                    shadow_color = "#000000"
                    font = self.fonts['large']
                
                # 텍스트 이미지 생성
                result = self._create_text_image(
                    display_text,
                    font,
                    text_color,
                    shadow_color,
                    shadow_offset=0,
                    shadow_blur=0
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    # Canvas에 이미지 추가
                    item_id = self.canvas.create_image(
                        30, y_positions[i],
                        image=photo,
                        anchor='nw'
                    )
                    
                    # 참조 유지 (가비지 컬렉션 방지)
                    self.canvas.itemconfig(item_id, tags=f"subtitle_{i}")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    
                    self.canvas_items.append(item_id)
            
            # 진행중 자막 렌더링 (주황색, 항상 하단)
            if self.current_text:
                result = self._create_text_image(
                    f"> {self.current_text}",
                    self.fonts['small'],
                    "orange",
                    "#663300",
                    shadow_offset=3,
                    shadow_blur=2
                )
                
                if result:
                    pil_img, img_w, img_h = result
                    photo = ImageTk.PhotoImage(pil_img)
                    
                    item_id = self.canvas.create_image(
                        30, y_positions[2],
                        image=photo,
                        anchor='nw'
                    )
                    
                    self.canvas.itemconfig(item_id, tags="subtitle_current")
                    self.canvas.photo_refs = getattr(self.canvas, 'photo_refs', [])
                    self.canvas.photo_refs.append(photo)
                    
                    self.canvas_items.append(item_id)
    
    def toggle_compact_mode(self):
        """Compact 모드 토글"""
        self.compact_mode = not self.compact_mode
        
        # 창 크기 조정
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        if self.compact_mode:
            self.window_width = 1600
            self.window_height = 100
            self.subtitles = deque(self.subtitles, maxlen=1)  # 최근 1개만 유지
            logger.info("Switched to Compact Mode (latest entry only)")
        else:
            self.window_width = 1900
            self.window_height = 300
            self.subtitles = deque(self.subtitles, maxlen=2)  # 2개 유지
            logger.info("Switched to Normal Mode (latest two entries)")
        
        x = (screen_width - self.window_width) // 2
        y = screen_height - self.window_height - 100
        
        self.geometry(f"{self.window_width}x{self.window_height}+{x}+{y}")
        
        # Canvas 크기 업데이트
        self.canvas.configure(width=self.window_width, height=self.window_height)
        self.canvas.delete(self.bg_rect)
        self.bg_rect = self._create_rounded_rectangle(
            0, 0, self.window_width, self.window_height,
            radius=self.bg_radius,
            fill='black'
        )
        
        # 메뉴 텍스트 업데이트
        mode_label = "📋 일반 모드로" if self.compact_mode else "📋 컴팩트 모드로"
        self.menu.entryconfig(3, label=mode_label)  # 메뉴 항목 인덱스
        
        self.render_subtitles()
    
    def animate_scroll_up(self, duration=300):
        """위로 스크롤 애니메이션 (부드러운 이동)"""
        if self.animation_running:
            return
        
        self.animation_running = True
        self.target_y_offset = -90
        start_offset = self.animation_y_offset
        start_time = time.time()
        
        def animate_step():
            if not self.animation_running:
                return
            
            elapsed = time.time() - start_time
            progress = min(elapsed / (duration / 1000.0), 1.0)
            
            # Ease-out 효과
            progress = 1 - (1 - progress) ** 3
            
            self.animation_y_offset = start_offset + (self.target_y_offset - start_offset) * progress
            self.render_subtitles()
            
            if progress < 1.0:
                self.after(16, animate_step)  # ~60 FPS
            else:
                self.animation_running = False
                self.animation_y_offset = 0  # 리셋
                self.render_subtitles()
        
        animate_step()
    
    def add_subtitle(self, translated_text, original_text):
        """자막 추가 (스크롤 애니메이션 포함)"""
        if not translated_text or not translated_text.strip():
            return
        
        # 중복 체크
        if self.subtitles and self.subtitles[-1][0] == translated_text:
            return
        
        # 스크롤 애니메이션 시작
        if len(self.subtitles) == 2:  # 이미 2개가 있으면 스크롤
            self.animate_scroll_up()
            # 애니메이션 후 추가
            self.after(350, lambda: self._add_subtitle_internal(translated_text, original_text))
        else:
            self._add_subtitle_internal(translated_text, original_text)
    
    def _add_subtitle_internal(self, translated_text, original_text):
        """내부: 자막 추가 및 렌더링"""
        self.subtitles.append((translated_text, original_text))
        self.current_text = ""  # 진행중 텍스트 클리어
        self.render_subtitles()
    
    def update_current(self, text):
        """진행중 자막 업데이트"""
        self.current_text = text if text else ""
        # 컴팩트 모드에서 인디케이터 애니메이션을 위해 카운터 증가
        if self.compact_mode and self.current_text:
            self.animation_counter += 1
        elif not self.current_text:
            self.animation_counter = 0
        self.render_subtitles()
    
    def clear_subtitles(self):
        """자막 지우기"""
        self.subtitles.clear()
        self.current_text = ""
        self.animation_counter = 0  # 인디케이터 리셋
        self.canvas.photo_refs = []
        self.render_subtitles()
    
    def start_move(self, event):
        """드래그 시작"""
        self._drag_start_x = event.x
        self._drag_start_y = event.y
    
    def on_move(self, event):
        """드래그 중"""
        x = self.winfo_x() + event.x - self._drag_start_x
        y = self.winfo_y() + event.y - self._drag_start_y
        self.geometry(f"+{x}+{y}")
    
    def show_menu(self, event):
        """우클릭 메뉴"""
        try:
            self.menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.menu.grab_release()
    
    def change_alpha(self, delta):
        """투명도 변경"""
        current = self.attributes('-alpha')
        new_alpha = max(0.1, min(1.0, current + delta))
        self.attributes('-alpha', new_alpha)
    
    def show_overlay(self):
        """오버레이 표시"""
        self.deiconify()
        self.lift()
    
    def hide_overlay(self):
        """오버레이 숨기기"""
        self.withdraw()


class WindyASRApp(ctk.CTk):
    """간단한 Windy ASR GUI 앱"""
    
    def __init__(self):
        super().__init__()
        
        self.title("RealtimeSTT Client (with Ollama Translation)")
        self.geometry("900x750")
        
        self.client = None
        self.translator = None
        self.recording = False
        self.recording_thread = None
        self.mic = None
        self.audio_buffer = np.array([], dtype=np.float32)
        
        # 오버레이 창
        self.overlay = None
        self.level_display = None
        
        # 번역 서비스 초기화 (기본값: Ollama)
        self.translator = None
        self.init_translator()
    
        self.init_gui()
    
    def init_translator(self):
        """번역 서비스 초기화 (Ollama)"""
        try:
            if OLLAMA_AVAILABLE:
                self.translator = TranslationService(
                    backend='ollama',
                    ollama_model=SETTINGS.get('ollama_model', 'yanolja-rosetta')
                )
        except Exception as e:
            logger.exception("Failed to initialize translator")
        self.update_translation_status_label()
        
    def init_gui(self):
        """GUI 초기화"""
        # 설정 프레임
        settings_frame = ctk.CTkFrame(self)
        settings_frame.pack(padx=10, pady=10, fill="x")
        
        # RealtimeSTT 설정 프레임
        realtimestt_frame = ctk.CTkFrame(settings_frame)
        realtimestt_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        
        ctk.CTkLabel(realtimestt_frame, text="모델:").pack(side="left", padx=3)
        self.realtimestt_model_combo = ctk.CTkComboBox(
            realtimestt_frame,
            values=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo", "RoachLin/kotoba-whisper-v2.2-faster", "distil-whisper/distil-large-v3.5-ct2"],
            state="readonly",
            width=200,
            command=self.on_realtimestt_model_change
        )
        self.realtimestt_model_combo.set(SETTINGS.get('realtimestt_model', 'tiny'))
        self.realtimestt_model_combo.pack(side="left", padx=3)
        
        ctk.CTkLabel(realtimestt_frame, text="디바이스:").pack(side="left", padx=3)
        self.realtimestt_device_combo = ctk.CTkComboBox(
            realtimestt_frame,
            values=["cuda", "cpu"],
            state="readonly",
            width=100,
            command=self.on_realtimestt_device_change
        )
        self.realtimestt_device_combo.set(SETTINGS.get('realtimestt_device', 'cuda'))
        self.realtimestt_device_combo.pack(side="left", padx=3)
        
        # 입력 소스 선택
        ctk.CTkLabel(settings_frame, text="입력 소스:").grid(row=1, column=0, padx=5, pady=5)
        
        mics = sc.all_microphones(include_loopback=True)
        mic_names = [mic.name for mic in mics]
        
        self.mic_combo = ctk.CTkComboBox(settings_frame, values=mic_names, width=400)
        if mic_names:
            # 기본 값을 스피커로
            speaker_index = next((i for i, mic in enumerate(mics) if "스피커" in mic.name.lower()), 0)
            self.mic_combo.set(mic_names[speaker_index])
        self.mic_combo.grid(row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        
        # 언어 설정 추가
        ctk.CTkLabel(settings_frame, text="입력 언어:").grid(row=2, column=0, padx=5, pady=5)
        
        languages = ["auto", "zh", "en", "ja", "ko", "yue"]
        language_labels = {
            "auto": "자동 감지",
            "zh": "중국어 (简体中文)",
            "en": "영어 (English)",
            "ja": "일본어 (日本語)",
            "ko": "한국어 (한국어)",
            "yue": "광둥어 (粵語)"
        }
        
        language_display = [f"{lang} - {language_labels[lang]}" for lang in languages]
        
        self.language_combo = ctk.CTkComboBox(
            settings_frame, 
            values=language_display, 
            width=300,
            command=self.on_language_change
        )
        # 기본값 ja로 설정
        default_lang = SETTINGS.get('language', 'ja')
        default_idx = languages.index(default_lang)
        self.language_combo.set(language_display[default_idx])
        self.language_combo.grid(row=2, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        
        # 번역 설정
        translation_frame = ctk.CTkFrame(settings_frame)
        translation_frame.grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="ew")
        
        # 번역 활성화/비활성화 (SETTINGS에서 초기값 읽음)
        self.translation_var = tk.BooleanVar(value=SETTINGS.get('translation_enabled', False))
        self.translation_check = ctk.CTkCheckBox(
            translation_frame,
            text="🌐 한국어 번역",
            variable=self.translation_var,
            command=self.on_translation_toggle
        )
        self.translation_check.pack(side="left", padx=5)
        
        # Ollama 모델 선택 UI
        self.fetch_ollama_models()
        ctk.CTkLabel(translation_frame, text="모델:").pack(side="left", padx=5)
        self.ollama_model_combo = ctk.CTkComboBox(
            translation_frame,
            values=self.ollama_models,
            width=220,
            command=self.on_ollama_model_change
        )
        default_model = SETTINGS.get('ollama_model', self.ollama_models[0] if self.ollama_models else '')
        if default_model:
            self.ollama_model_combo.set(default_model)
        self.ollama_model_combo.pack(side="left", padx=5)

        ctk.CTkButton(
            translation_frame,
            text="🔄 새로고침",
            width=110,
            command=self.refresh_ollama_models
        ).pack(side="left", padx=5)

        # 상태 표시 레이블
        self.translation_status_label = ctk.CTkLabel(
            translation_frame,
            text="",
            text_color="gray"
        )
        self.translation_status_label.pack(side="left", padx=10)
        self.update_translation_status_label()
        
        # 제어 프레임
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(padx=10, pady=10, fill="x")
        
        self.start_btn = ctk.CTkButton(
            control_frame,
            text="🎙️ 시작",
            command=self.start_recording,
            fg_color="green",
            hover_color="darkgreen",
            height=40,
            font=("Arial", 16, "bold")
        )
        self.start_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        self.stop_btn = ctk.CTkButton(
            control_frame,
            text="⏹️ 정지",
            command=self.stop_recording,
            fg_color="red",
            hover_color="darkred",
            height=40,
            font=("Arial", 16, "bold"),
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        # 오버레이 토글 버튼
        self.overlay_btn = ctk.CTkButton(
            control_frame,
            text="📺 오버레이",
            command=self.toggle_overlay,
            fg_color="purple",
            hover_color="darkviolet",
            height=40,
            font=("Arial", 16, "bold")
        )
        self.overlay_btn.pack(side="left", padx=5, expand=True, fill="x")
        
        # 상태 표시
        self.status_label = ctk.CTkLabel(
            control_frame,
            text="● 준비",
            font=("Arial", 14),
            text_color="gray"
        )
        self.status_label.pack(side="left", padx=10)

        level_frame = ctk.CTkFrame(self)
        level_frame.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(level_frame, text="📊 오디오 레벨", font=("Arial", 12, "bold")).pack(anchor="w", padx=5, pady=5)
        self.level_display = AudioLevelDisplay(level_frame, width=500, height=60)
        self.level_display.get_widget().pack(fill="x", padx=5, pady=5)
        
        # 결과 표시 프레임
        result_frame = ctk.CTkFrame(self)
        result_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        ctk.CTkLabel(result_frame, text="인식 결과:", font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=5)
        
        # 진행중 텍스트
        self.current_label = ctk.CTkLabel(
            result_frame,
            text="",
            font=("Malgun Gothic", 12),
            text_color="orange",
            anchor="w",
            justify="left"
        )
        self.current_label.pack(fill="x", padx=5, pady=5)
        
        # 완료된 텍스트
        self.result_text = ctk.CTkTextbox(result_frame, font=("Malgun Gothic", 12))
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # 클리어 버튼
        ctk.CTkButton(
            result_frame,
            text="🗑️ 결과 지우기",
            command=self.clear_results
        ).pack(padx=5, pady=5)
    
    def on_translation_toggle(self):
        """번역 토글 콜백"""
        enabled = self.translation_var.get()
        SETTINGS['translation_enabled'] = enabled
        save_settings()
        logger.info("Translation %s", 'enabled' if enabled else 'disabled')
        logger.debug("translation_enabled setting=%s", SETTINGS['translation_enabled'])

    def fetch_ollama_models(self):
        """Fetch and cache available Ollama models."""
        models = get_available_ollama_models()
        if not models:
            fallback = SETTINGS.get('ollama_model', DEFAULT_SETTINGS.get('ollama_model'))
            if fallback:
                models = [fallback]
        self.ollama_models = models
        return models

    def refresh_ollama_models(self):
        """Reload model list and update UI."""
        models = self.fetch_ollama_models()
        if hasattr(self, 'ollama_model_combo') and self.ollama_model_combo:
            if models:
                self.ollama_model_combo.configure(values=models)
                current = self.ollama_model_combo.get()
                if current not in models:
                    self.ollama_model_combo.set(SETTINGS.get('ollama_model', models[0]))
            else:
                self.ollama_model_combo.configure(values=[''])
        logger.info("Ollama models refreshed: %s", models)
        self.init_translator()

    def on_ollama_model_change(self, selection):
        """Handle Ollama model change from combo box."""
        if not selection:
            return
        if selection == SETTINGS.get('ollama_model'):
            return
        SETTINGS['ollama_model'] = selection
        save_settings()
        logger.info("Ollama model changed to %s", selection)
        self.init_translator()
        self.update_translation_status_label()

    def update_translation_status_label(self):
        """Update translation availability indicator."""
        if not hasattr(self, 'translation_status_label'):
            return

        if self.translator and self.translator.is_available():
            self.translation_status_label.configure(text="✅ Ready (Ollama)", text_color="green")
            self.translation_check.configure(state="normal")
        else:
            self.translation_status_label.configure(text="❌ Not Available", text_color="red")
            self.translation_check.configure(state="disabled")
    
    def on_realtimestt_device_change(self, selection):
        """RealtimeSTT 디바이스 변경"""
        SETTINGS['realtimestt_device'] = selection
        save_settings()
        logger.info("RealtimeSTT device changed to %s", selection)
    
    def on_realtimestt_model_change(self, selection):
        """RealtimeSTT 모델 변경"""
        SETTINGS['realtimestt_model'] = selection
        save_settings()
        logger.info("RealtimeSTT model changed to %s", selection)
    
    def on_language_change(self, selection):
        """언어 변경 시 설정 저장"""
        language = selection.split(" - ")[0]
        SETTINGS['language'] = language
        save_settings()
        logger.info("Input language changed to %s", language)

    def start_recording(self):
        """녹음 시작"""
        try:
            # 언어 추출 (콤보박스에서 언어 코드만 추출)
            language_display = self.language_combo.get()
            language = language_display.split(" - ")[0]
            
            logger.info("Selected language: %s", language)
            
            # Whisper 모델이고 언어가 'auto'면 빈 공백으로 전달
            if language == 'auto':
                language = ''
                logger.info("Auto-detect mode enabled (empty language string for Whisper)")
            
            self.client = RealtimeSTTClient(
                model=SETTINGS.get('realtimestt_model', 'tiny'),
                language=language,
                device=SETTINGS.get('realtimestt_device', 'cuda'),
                enable_realtime_transcription=True,  # 실시간 업데이트 활성화
                post_speech_silence_duration=SETTINGS.get('realtimestt_post_speech_silence_duration', 0.1),
                silero_sensitivity=SETTINGS.get('realtimestt_silero_sensitivity', 0.3),
                webrtc_sensitivity=SETTINGS.get('realtimestt_webrtc_sensitivity', 3),
                beam_size=SETTINGS.get('realtimestt_beam_size', 5),
                batch_size=SETTINGS.get('realtimestt_batch_size', 16),
                min_length_of_recording=SETTINGS.get('realtimestt_min_length_of_recording', 0.1),
                use_extended_logging=SETTINGS.get('use_extended_logging', False),
                silero_use_onnx=SETTINGS.get('silero_use_onnx', False),
                silero_deactivity_detection=SETTINGS.get('silero_deactivity_detection', False)
            )
            # 실시간 콜백 설정 (진행중인 텍스트 표시)
            self.client.set_realtime_callback(self.on_result)
            self.client.set_callback(self.on_result)
            
            if not self.client.connect():
                self.update_status("연결 실패", "red")
                return
            
            # 마이크 선택
            mic_name = self.mic_combo.get()
            mics = sc.all_microphones(include_loopback=True)
            self.mic = next((m for m in mics if m.name == mic_name), None)
            
            if not self.mic:
                self.update_status("마이크 없음", "red")
                return
            
            # 녹음 시작
            self.recording = True
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()
            
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.update_status("녹음 중", "green")
            
        except Exception as e:
            logger.exception("Start error")
            self.update_status("시작 실패", "red")
    
    def stop_recording(self):
        """녹음 정지"""
        self.recording = False
        
        if self.client:
            self.client.disconnect()
            self.client = None
        
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.update_status("정지", "gray")
        if self.level_display:
            self.level_display.reset()
    
    def _recording_loop(self):
        """녹음 루프"""
        try:
            with self.mic.recorder(
                samplerate=SETTINGS['sample_rate'],
                channels=2
            ) as recorder:
                while self.recording:
                    # 0.1초씩 녹음
                    data = recorder.record(numframes=int(SETTINGS['sample_rate'] * SETTINGS['chunk_duration']))
                    
                    if self.level_display:
                        self.after(0, lambda chunk=data.copy(): self.level_display.update_level(chunk))

                    if self.client and self.client.connected:
                        self.client.send_audio(data)
                        
        except Exception as e:
            logger.exception("Recording error")
            self.recording = False
        finally:
            if self.level_display:
                self.after(0, self.level_display.reset)
    
    def on_result(self, text, is_final):
        """인식 결과 콜백"""
        if is_final:
            # 완료된 결과만 번역
            if self.translation_var.get() and self.translator and self.translator.is_available():
                # 비동기로 번역 (UI 블로킹 방지)
                threading.Thread(
                    target=self._translate_and_display,
                    args=(text,),
                    daemon=True
                ).start()
            else:
                # 번역 없이 바로 표시
                self._display_final_result(text, text)
        else:
            # 진행중 결과는 번역 없이 원문만 표시
            self._display_current_result(text)
    
    def _translate_and_display(self, original_text):
        """번역 후 표시 (별도 스레드)"""
        try:
            # LM Studio에 원문만 전달 (프롬프트 없음)
            translated = self.translator.translate(original_text)
            self._display_final_result(translated, original_text)
        except Exception as e:
            logger.exception("Translation thread error")
            self._display_final_result(original_text, original_text)
    
    def _display_current_result(self, text):
        """진행중 결과 표시 (메인 스레드)"""
        def update_ui():
            self.current_label.configure(text=f"🎤 {text}")
            
            # 오버레이 업데이트 (진행중은 원문만)
            if self.overlay:
                self.overlay.update_current(text)
            
            # print(f"⏳ Progress: {text}")
        
        self.after(0, update_ui)
    
    def _display_final_result(self, translated_text, original_text):
        """완료된 결과 표시 (메인 스레드)"""
        def update_ui():
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # 결과 텍스트박스에 표시: "한국어 (원문)"
            if original_text != translated_text:
                display_line = f"[{timestamp}] {translated_text} ({original_text})\n"
            else:
                display_line = f"[{timestamp}] {translated_text}\n"
            
            self.result_text.insert("end", display_line)
            self.result_text.see("end")
            self.current_label.configure(text="")
            
            # 오버레이에 추가: "한국어 (원문)"
            if self.overlay:
                self.overlay.add_subtitle(translated_text, original_text)
            
            logger.debug("Final displayed: %s (%s)", translated_text, original_text)
        
        self.after(0, update_ui)
    
    def update_status(self, text, color):
        """상태 업데이트"""
        self.status_label.configure(text=f"● {text}", text_color=color)
    
    def clear_results(self):
        """결과 지우기"""
        self.result_text.delete("1.0", "end")
        self.current_label.configure(text="")
        if self.level_display:
            self.level_display.reset()
        if self.overlay:
            self.overlay.clear_subtitles()
    
    def toggle_overlay(self):
        """오버레이 토글"""
        if self.overlay is None:
            # 오버레이 생성 (compact=False로 일반 모드 시작)
            self.overlay = SubtitleOverlay(self, compact=False)
            self.overlay.show_overlay()
            self.overlay_btn.configure(text="📺 오버레이 ON", fg_color="green")
        else:
            # 오버레이 표시/숨김 토글
            if self.overlay.winfo_viewable():
                self.overlay.hide_overlay()
                self.overlay_btn.configure(text="📺 오버레이 OFF", fg_color="purple")
            else:
                self.overlay.show_overlay()
                self.overlay_btn.configure(text="📺 오버레이 ON", fg_color="green")
    
    def on_closing(self):
        """종료 처리"""
        self.stop_recording()
        if self.overlay:
            self.overlay.destroy()
        self.destroy()


def main():
    """메인 함수"""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    app = WindyASRApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()

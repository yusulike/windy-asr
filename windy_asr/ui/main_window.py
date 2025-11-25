"""
Main Application Window Module
Contains the primary GUI for Windy ASR
"""

import logging
import threading
import tkinter as tk
from datetime import datetime
import numpy as np
import customtkinter as ctk
import soundcard as sc

from windy_asr.config import SETTINGS, DEFAULT_SETTINGS, save_settings, get_available_ollama_models
from windy_asr.services.translation import TranslationService, OLLAMA_AVAILABLE
from windy_asr.services.stt import RealtimeSTTClient
from windy_asr.ui.widgets import AudioLevelDisplay
from windy_asr.ui.overlay import SubtitleOverlay

logger = logging.getLogger('windy_asr.ui.main_window')


class WindyASRApp(ctk.CTk):
    """간단한 Windy ASR GUI 앱"""
    
    def __init__(self):
        super().__init__()
        
        self.title("Windy ASR")
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

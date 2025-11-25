"""
Windy ASR Client
실시간 음성 인식 클라이언트 (RealtimeSTT + Ollama)

Usage:
    python app.py
"""

import logging
import customtkinter as ctk
from windy_asr import WindyASRApp
from windy_asr.config import LOG_LEVEL

# Configure basic logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(name)s - %(levelname)s - %(message)s"
)

# Disable noisy loggers
logging.getLogger("RealtimeSTT.safepipe").disabled = True


def main():
    """메인 함수"""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    
    app = WindyASRApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()

"""
Windy ASR - Real-time Speech-to-Text Client
"""

from windy_asr.config import SETTINGS, load_settings, save_settings, get_available_ollama_models
from windy_asr.services.translation import TranslationService
from windy_asr.services.stt import RealtimeSTTClient
from windy_asr.ui.widgets import AudioLevelDisplay
from windy_asr.ui.overlay import SubtitleOverlay
from windy_asr.ui.main_window import WindyASRApp

__all__ = [
    'SETTINGS',
    'load_settings',
    'save_settings',
    'get_available_ollama_models',
    'TranslationService',
    'RealtimeSTTClient',
    'AudioLevelDisplay',
    'SubtitleOverlay',
    'WindyASRApp',
]

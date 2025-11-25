"""
Configuration and Settings Management for Windy ASR
"""

import logging
import os
import json

# Basic logging configuration (aligns with external component format)
LOG_LEVEL = os.environ.get('WINDY_ASR_LOG_LEVEL', 'INFO').upper()
LOG_PREFIX = os.environ.get('WINDY_ASR_LOG_PREFIX', 'WindyASR')
LOG_FORMAT = os.environ.get(
    'WINDY_ASR_LOG_FORMAT',
    f"{LOG_PREFIX}: %(name)s - %(levelname)s - %(message)s"
)

# Initialize logger
logger = logging.getLogger('windy_asr')


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


# Default settings configuration
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
    'use_extended_logging': False,
}

CONFIG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def get_available_ollama_models():
    """Retrieve installed Ollama models (best-effort)."""
    try:
        from ollama import list as ollama_list

        models_response = ollama_list()
        # 모델 이름 끝에 :latest는 제거
        model_names = [m.get('model', '').replace(':latest', '') for m in models_response.get('models', []) if m.get('model')]
        return sorted(set(model_names))
    except ImportError:
        logger.debug("Ollama library unavailable; skipping model discovery")
        return []
    except Exception as exc:
        logger.warning("Failed to list Ollama models: %s", exc)
        return []


# Initialize settings on module import
SETTINGS = {}
load_settings()

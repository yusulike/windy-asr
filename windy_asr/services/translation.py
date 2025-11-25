"""
Translation Service Module
Handles translation using Ollama backend
"""

import logging
from windy_asr.config import get_available_ollama_models

logger = logging.getLogger('windy_asr.translation')

# Check Ollama availability
try:
    from ollama import chat, ChatResponse
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama library not available. Translation will be disabled.")
    logger.info("Install with: pip install ollama")


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

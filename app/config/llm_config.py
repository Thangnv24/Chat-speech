import os
from typing import Optional, Dict, Any
from enum import Enum
from app.utils.logger import setup_logging
from app.core.llm_client import get_llm_client_for_config

logger = setup_logging("llm_config")

class LLMProvider(Enum):
    GEMINI = "gemini"
    QWEN = "qwen"
    OLLAMA = "ollama"

class LLMConfig:
    def __init__(self):
        self.provider = self._detect_preferred_provider()
        self.config = self._load_config()
    
    def _detect_preferred_provider(self) -> LLMProvider:
        if os.getenv("GEMINI_API_KEY"):
            return LLMProvider.GEMINI
        elif os.getenv("QWEN_API_KEY"):
            return LLMProvider.QWEN
        else:
            return LLMProvider.OLLAMA
    
    def _load_config(self) -> Dict[str, Any]:
        if self.provider == LLMProvider.GEMINI:
            return self._get_gemini_config()
        elif self.provider == LLMProvider.QWEN:
            return self._get_qwen_config()
        else:
            return self._get_ollama_config()  
    
    def _get_gemini_config(self) -> Dict[str, Any]:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found, falling back to Ollama")
            return self._get_ollama_config()
        
        return {
            "provider": "gemini",
            "api_key": api_key,
            "model_name": "gemini-2.5-flash",  
            "base_url": "https://generativelanguage.googleapis.com/v1beta/models/",
            "max_tokens": 2048,
            "temperature": 0.3,
            "rate_limit": {
                "requests_per_minute": 60, 
                "tokens_per_minute": 32000
            }
        }
    
    def _get_qwen_config(self) -> Dict[str, Any]:
        api_key = os.getenv("QWEN_API_KEY")
        if not api_key:
            logger.warning("QWEN_API_KEY not found, falling back to Ollama")
            return self._get_ollama_config()
        
        return {
            "provider": "qwen",
            "api_key": api_key,
            "model_name": "qwen-plus",  
            "base_url": "https://dashscope.aliyuncs.com/api/v1/",
            "max_tokens": 1500,
            "temperature": 0.3,
            "rate_limit": {
                "requests_per_minute": 1000,
                "tokens_per_minute": 100000
            }
        }
    
    def _get_ollama_config(self) -> Dict[str, Any]:
        return {
            "provider": "ollama",
            "base_url": "http://localhost:11434",
            "model_name": "llama2", 
            "max_tokens": 2000,
            "temperature": 0.3,
            "rate_limit": {
                "requests_per_minute": float('inf'),  
                "tokens_per_minute": float('inf')
            }
        }
    
    def get_llm_client(self):
        return get_llm_client_for_config(self.config)

llm_config = LLMConfig()
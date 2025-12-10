import time
from typing import Dict, Any, Optional
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import Generation, LLMResult
from langchain_community.llms import Ollama
from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from app.utils.logger import setup_logging

logger = setup_logging("llm_client")

class RateLimitedLLM:
    """Rate limiting wrapper for LLM calls"""
    
    def __init__(self, llm: LLM, config: Dict[str, Any]):
        self.llm = llm
        self.config = config
        self.last_call_time = 0
        self.requests_per_minute = config.get("rate_limit", {}).get("requests_per_minute", 60)
        self.min_interval = 60.0 / self.requests_per_minute
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def __call__(self, *args, **kwargs):
        self._rate_limit()
        return self.llm(*args, **kwargs)

def get_llm_client_for_config(config: Dict[str, Any]) -> Optional[LLM]:
    provider = config["provider"]
    
    try:
        if provider == "gemini":
            return _get_gemini_client(config)
        elif provider == "qwen":
            return _get_qwen_client(config)
        elif provider == "ollama":
            return _get_ollama_client(config)
        else:
            logger.error(f"Unsupported provider: {provider}")
            return None
            
    except Exception as e:
        logger.error(f"Failed to create LLM client for {provider}: {str(e)}")
        return None

def _get_gemini_client(config: Dict[str, Any]) -> LLM:
    try:      
        llm = ChatGoogleGenerativeAI(
            model=config["model_name"],
            google_api_key=config["api_key"],
            temperature=config["temperature"],
            max_output_tokens=config["max_tokens"]
        )
        
        logger.info("Gemini LLM client created successfully")
        return RateLimitedLLM(llm, config)
        
    except ImportError:
        logger.error("google-generativeai package not installed")
        raise

def _get_qwen_client(config: Dict[str, Any]) -> LLM:
    try:
        
        llm = OpenAI(
            model_name=config["model_name"],
            openai_api_key=config["api_key"],
            openai_api_base=config["base_url"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        logger.info("Qwen LLM client created successfully")
        return RateLimitedLLM(llm, config)
        
    except Exception as e:
        logger.error(f"Failed to create Qwen client: {str(e)}")
        raise

def _get_ollama_client(config: Dict[str, Any]) -> LLM:
    try:
        llm = Ollama(
            model=config["model_name"],
            base_url=config["base_url"],
            temperature=config["temperature"],
            num_predict=config["max_tokens"]
        )
        
        logger.info("Ollama LLM client created successfully")
        return llm  
        
    except Exception as e:
        logger.error(f"Failed to create Ollama client: {str(e)}")
        raise
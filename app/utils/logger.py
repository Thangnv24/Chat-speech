import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json

class jsonFormat(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

    if hasattr(record, 'extra_data'):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry)

class ProjectLogger:
    def __init__(self, name=None, log_level="INFO", log_dir="logs"):
        self.name = name or __name__
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with multiple handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        json_formatter = JSONFormatter()
    
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        
        # File Handler - All logs
        file_handler = logging.FileHandler(self.log_dir / 'app.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Error Handler
        error_handler = logging.FileHandler(self.log_dir / 'error.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        
        # JSON Handler (for structured logging)
        json_handler = logging.FileHandler(self.log_dir / 'structured.log')
        json_handler.setLevel(logging.INFO)
        json_handler.setFormatter(json_formatter)
        
        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)
        logger.addHandler(json_handler)
        
        return logger
    
    # Proxy methods 
    def debug(self, message, extra_data=None):
        self._log_with_extra(logging.DEBUG, message, extra_data)
    
    def info(self, message, extra_data=None):
        self._log_with_extra(logging.INFO, message, extra_data)
    
    def warning(self, message, extra_data=None):
        self._log_with_extra(logging.WARNING, message, extra_data)
    
    def error(self, message, extra_data=None):
        self._log_with_extra(logging.ERROR, message, extra_data)
    
    def critical(self, message, extra_data=None):
        self._log_with_extra(logging.CRITICAL, message, extra_data)
    
    def _log_with_extra(self, level, message, extra_data=None):
        if extra_data:
            self.logger.log(level, message, extra={'extra_data': extra_data})
        else:
            self.logger.log(level, message)

# global logger
def setup_logging(name=None, log_level="INFO", log_dir="logs"):
    return ProjectLogger(name, log_level, log_dir)

# defaut
logger = setup_logging("llm_rag_project")

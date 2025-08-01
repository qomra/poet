# config/config_manager.py

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data source configuration"""
    local_knowledge_path: str
    rhyme_dict_path: str
    enable_search: bool = False
    search_api_key: Optional[str] = None

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3

@dataclass
class GenerationConfig:
    """Generation parameters configuration"""
    max_iterations: int = 5
    creativity_weight: float = 0.7
    constraint_strictness: float = 0.9
    temperature: float = 0.7
    max_tokens: int = 2000

@dataclass
class EvaluationConfig:
    """Evaluation thresholds configuration"""
    min_prosody_score: float = 0.85
    min_semantic_coherence: float = 0.75
    max_refinement_cycles: int = 3

@dataclass
class PerformanceConfig:
    """Performance settings configuration"""
    corpus_cache_size: int = 1000
    search_result_limit: int = 50
    parallel_requests: bool = False

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None

class ConfigManager:
    """
    Manages configuration loading and access for the Poet Library.
    
    Handles loading from YAML files, environment variable overrides,
    and provides typed access to configuration sections.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self.logger = logging.getLogger(__name__)
        
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent
        return current_dir / "default_config.yaml"
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            
            # Apply environment variable overrides
            self._apply_env_overrides()
            
            self.logger.info(f"Loaded configuration from {self.config_path}")
            
        except FileNotFoundError:
            self.logger.warning(f"Configuration file not found: {self.config_path}")
            self._config = {}
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing configuration file: {e}")
            self._config = {}
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Data paths
        if os.getenv("POET_LOCAL_KNOWLEDGE_PATH"):
            self._config.setdefault("data", {})["local_knowledge_path"] = os.getenv("POET_LOCAL_KNOWLEDGE_PATH")
        
        if os.getenv("POET_RHYME_DICT_PATH"):
            self._config.setdefault("data", {})["rhyme_dict_path"] = os.getenv("POET_RHYME_DICT_PATH")
        
        # LLM API keys
        if os.getenv("OPENAI_API_KEY"):
            self._config.setdefault("llm", {}).setdefault("openai", {})["api_key"] = os.getenv("OPENAI_API_KEY")
        
        if os.getenv("ANTHROPIC_API_KEY"):
            self._config.setdefault("llm", {}).setdefault("anthropic", {})["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        
        if os.getenv("GEMINI_API_KEY"):
            self._config.setdefault("llm", {}).setdefault("gemini", {})["api_key"] = os.getenv("GEMINI_API_KEY")
        
        # Search API
        if os.getenv("SERPAPI_KEY"):
            self._config.setdefault("data", {})["search_api_key"] = os.getenv("SERPAPI_KEY")
            self._config.setdefault("data", {})["enable_search"] = True
    
    def get_data_config(self) -> DataConfig:
        """Get data source configuration"""
        data_config = self._config.get("data", {})
        
        return DataConfig(
            local_knowledge_path=data_config.get("local_knowledge_path", "kb/"),
            rhyme_dict_path=data_config.get("rhyme_dict_path", "kb/fahras.json"),
            enable_search=data_config.get("enable_search", False),
            search_api_key=data_config.get("search_api_key")
        )
    
    def get_llm_config(self, provider: str) -> LLMConfig:
        """Get LLM provider configuration"""
        llm_configs = self._config.get("llm", {})
        provider_config = llm_configs.get(provider, {})
        
        return LLMConfig(
            api_key=provider_config.get("api_key"),
            base_url=provider_config.get("base_url"),
            timeout=provider_config.get("timeout", 30),
            max_retries=provider_config.get("max_retries", 3)
        )
    
    def get_generation_config(self) -> GenerationConfig:
        """Get generation parameters configuration"""
        gen_config = self._config.get("generation", {})
        
        return GenerationConfig(
            max_iterations=gen_config.get("max_iterations", 5),
            creativity_weight=gen_config.get("creativity_weight", 0.7),
            constraint_strictness=gen_config.get("constraint_strictness", 0.9),
            temperature=gen_config.get("temperature", 0.7),
            max_tokens=gen_config.get("max_tokens", 2000)
        )
    
    def get_evaluation_config(self) -> EvaluationConfig:
        """Get evaluation thresholds configuration"""
        eval_config = self._config.get("evaluation", {})
        
        return EvaluationConfig(
            min_prosody_score=eval_config.get("min_prosody_score", 0.85),
            min_semantic_coherence=eval_config.get("min_semantic_coherence", 0.75),
            max_refinement_cycles=eval_config.get("max_refinement_cycles", 3)
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance settings configuration"""
        perf_config = self._config.get("performance", {})
        
        return PerformanceConfig(
            corpus_cache_size=perf_config.get("corpus_cache_size", 1000),
            search_result_limit=perf_config.get("search_result_limit", 50),
            parallel_requests=perf_config.get("parallel_requests", False)
        )
    
    def get_logging_config(self) -> LoggingConfig:
        """Get logging configuration"""
        log_config = self._config.get("logging", {})
        
        return LoggingConfig(
            level=log_config.get("level", "INFO"),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file=log_config.get("file")
        )
    
    def get_primary_model(self) -> str:
        """Get primary model name"""
        models_config = self._config.get("models", {})
        return models_config.get("primary", "gpt-4o")
    
    def get_fallback_model(self) -> str:
        """Get fallback model name"""
        models_config = self._config.get("models", {})
        return models_config.get("fallback", "gpt-3.5-turbo")
    
    def get_local_model(self) -> Optional[str]:
        """Get local model name"""
        models_config = self._config.get("models", {})
        return models_config.get("local")
    
    def get_local_knowledge_path(self) -> str:
        """Get local knowledge path (convenience method)"""
        return self.get_data_config().local_knowledge_path
    
    def get_rhyme_dict_path(self) -> str:
        """Get rhyme dictionary path (convenience method)"""
        return self.get_data_config().rhyme_dict_path
    
    def is_search_enabled(self) -> bool:
        """Check if external search is enabled (convenience method)"""
        return self.get_data_config().enable_search
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return self._config.copy()
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
    
    def validate_config(self) -> bool:
        """Validate configuration completeness"""
        data_config = self.get_data_config()
        
        # Check if local knowledge path exists
        local_knowledge_path = Path(data_config.local_knowledge_path)
        if not local_knowledge_path.exists():
            self.logger.warning(f"Local knowledge path does not exist: {local_knowledge_path}")
            return False
        
        # Check if at least one LLM is configured
        has_llm = False
        for provider in ["openai", "anthropic", "gemini"]:
            llm_config = self.get_llm_config(provider)
            if llm_config.api_key:
                has_llm = True
                break
        
        if not has_llm:
            self.logger.warning("No LLM provider configured with API key")
            return False
        
        return True

# Singleton instance for global access
_config_manager: Optional[ConfigManager] = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None or config_path is not None:
        _config_manager = ConfigManager(config_path)
    
    return _config_manager

def get_local_knowledge_path() -> str:
    """Get local knowledge path from global config"""
    return get_config_manager().get_local_knowledge_path()

def get_rhyme_dict_path() -> str:
    """Get rhyme dictionary path from global config"""
    return get_config_manager().get_rhyme_dict_path()

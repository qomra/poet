# poet/llm/llm_factory.py

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .base_llm import BaseLLM, LLMConfig, LLMError
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter
from .groq_adapter import GroqAdapter

def get_real_llm_from_env() -> Optional[BaseLLM]:
    """
    Get real LLM instance from environment variable and configuration.
    
    Returns:
        BaseLLM instance if configured, None otherwise
        
    Environment Variables:
        TEST_REAL_LLMS: Must be set to enable real LLM loading
        REAL_LLM_PROVIDER: Specify which provider to use (default: openai)
    """
    if not os.getenv("TEST_REAL_LLMS"):
        return None
    
    provider = os.getenv("REAL_LLM_PROVIDER", "anthropic").lower()
    
    # Load configuration
    config_data = _load_llm_config()
    if not config_data or provider not in config_data:
        print(f"DEBUG: No config data or provider not found, returning None")
        return None
    
    provider_config = config_data[provider]
    
    # Skip if API key is placeholder or get from environment
    api_key = provider_config.get("api_key", "")
    
    # Override with environment variable if available
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "groq" and os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "sk-" or api_key == "your-openai-api-key-here":
        return None
    
    # Create LLM instance based on provider
    if provider == "openai":
        return _create_openai_llm(provider_config, api_key)
    elif provider == "anthropic":
        return _create_anthropic_llm(provider_config, api_key)
    elif provider == "groq":
        return _create_groq_llm(provider_config, api_key)
    elif provider == "gemini":
        raise NotImplementedError("Gemini adapter not implemented yet")
    else:
        raise LLMError(f"Unknown LLM provider: {provider}")

def _load_llm_config() -> Optional[Dict[str, Any]]:
    """Load LLM configuration from fixtures."""
    config_files = ["llms.json", "llms_example.json"]
    
    # Look in tests/fixtures directory
    base_path = Path(__file__).parent.parent.parent / "tests" / "fixtures"
    
    for config_file in config_files:
        config_path = base_path / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    
    return None

def _create_openai_llm(config_data: Dict[str, Any], api_key: str) -> OpenAIAdapter:
    """Create OpenAI LLM instance from configuration."""
    config = LLMConfig(
        model_name=config_data["model"],
        api_key=api_key,
        base_url=config_data.get("api_base"),
        timeout=30
    )
    
    return OpenAIAdapter(config)

def _create_anthropic_llm(config_data: Dict[str, Any], api_key: str) -> AnthropicAdapter:
    """Create Anthropic LLM instance from configuration."""
    config = LLMConfig(
        model_name=config_data["model"],
        api_key=api_key,
        base_url=config_data.get("api_base"),
        timeout=30
    )
    
    return AnthropicAdapter(config)

def _create_groq_llm(config_data: Dict[str, Any], api_key: str) -> GroqAdapter:
    """Create Groq LLM instance from configuration."""
    config = LLMConfig(
        model_name=config_data["model"],
        api_key=api_key,
        base_url=config_data.get("api_base"),
        timeout=30
    )
    
    return GroqAdapter(config) 
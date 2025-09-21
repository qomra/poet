# poet/llm/vllm_adapter.py

import json
import time
from typing import Optional, Dict, Any
import logging

from .base_llm import BaseLLM, LLMConfig, LLMResponse, LLMError, LLMConnectionError, LLMTimeoutError, LLMRateLimitError, LLMInvalidRequestError

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

class VLLMAdapter(BaseLLM):
    """vLLM LLM adapter using OpenAI client to connect to local vLLM server."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not OPENAI_AVAILABLE:
            raise LLMError("OpenAI package not installed. Install with: pip install openai")
        
        # For vLLM, we don't require an API key (can be empty string)
        # but we do require a base_url pointing to the vLLM server
        if not config.base_url:
            raise LLMError("vLLM base_url is required (e.g., http://localhost:8000)")
        
        # Initialize OpenAI client pointing to vLLM server
        self.client = openai.OpenAI(
            api_key=config.api_key or "",  # vLLM doesn't require API key
            base_url=config.base_url,
            timeout=config.timeout
        )
        
        self.logger.info(f"Initialized vLLM adapter with model: {config.model_name}")
        self.logger.info(f"vLLM server URL: {config.base_url}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using vLLM server."""
        response = self.generate_with_metadata(prompt, **kwargs)
        return response.content
    
    def generate_with_metadata(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with full metadata using vLLM server."""
        try:
            # Merge parameters
            params = self._merge_params(**kwargs)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Make API call
            self.logger.debug(f"Making vLLM API call with model: {params['model']}")
            start_time = time.time()
            
            # Prepare API parameters, filtering out None values
            api_params = {
                "model": params["model"],
                "messages": messages,
            }
            
            # Add optional parameters only if they're explicitly provided in kwargs
            if "temperature" in kwargs and kwargs["temperature"] is not None:
                api_params["temperature"] = kwargs["temperature"]
            if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
                api_params["max_tokens"] = kwargs["max_tokens"]
            if "top_p" in kwargs and kwargs["top_p"] is not None:
                api_params["top_p"] = kwargs["top_p"]
            if "frequency_penalty" in kwargs and kwargs["frequency_penalty"] is not None:
                api_params["frequency_penalty"] = kwargs["frequency_penalty"]
            if "presence_penalty" in kwargs and kwargs["presence_penalty"] is not None:
                api_params["presence_penalty"] = kwargs["presence_penalty"]
            
            # Add other parameters that aren't None
            for k, v in params.items():
                if k not in ["model", "temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"] and v is not None:
                    api_params[k] = v
            
            response = self.client.chat.completions.create(**api_params)
            
            end_time = time.time()
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            
            # Build usage info
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            # Build metadata
            metadata = {
                "response_time": end_time - start_time,
                "model": response.model,
                "created": response.created,
                "id": response.id,
                "system_fingerprint": getattr(response, 'system_fingerprint', None),
                "vllm_server": self.config.base_url
            }
            
            self.logger.debug(f"vLLM API call completed in {metadata['response_time']:.2f}s")
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
                metadata=metadata
            )
            
        except openai.AuthenticationError as e:
            self.logger.error(f"vLLM authentication error: {e}")
            raise LLMConnectionError(f"Authentication failed: {e}")
        
        except openai.RateLimitError as e:
            self.logger.error(f"vLLM rate limit error: {e}")
            raise LLMRateLimitError(f"Rate limit exceeded: {e}")
        
        except openai.APITimeoutError as e:
            self.logger.error(f"vLLM timeout error: {e}")
            raise LLMTimeoutError(f"Request timed out: {e}")
        
        except openai.BadRequestError as e:
            self.logger.error(f"vLLM bad request error: {e}")
            raise LLMInvalidRequestError(f"Invalid request: {e}")
        
        except openai.APIConnectionError as e:
            self.logger.error(f"vLLM connection error: {e}")
            raise LLMConnectionError(f"Connection failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected vLLM error: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def is_available(self) -> bool:
        """Check if vLLM service is available."""
        if not OPENAI_AVAILABLE:
            return False
        
        try:
            # Make a minimal API call to check availability
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1,
                temperature=0
            )
            return True
        except Exception as e:
            self.logger.warning(f"vLLM availability check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = super().get_model_info()
        info.update({
            "provider": "vllm",
            "supports_streaming": True,
            "supports_functions": False,  # vLLM doesn't support function calling
            "context_length": self._get_context_length(self.config.model_name),
            "server_url": self.config.base_url
        })
        return info
    
    def _get_context_length(self, model_name: str) -> int:
        """Get context length for vLLM models."""
        # Common context lengths for models typically used with vLLM
        context_lengths = {
            "meta-llama/Meta-Llama-3-8B": 8192,
            "meta-llama/Meta-Llama-3-70B": 8192,
            "meta-llama/Llama-2-7b-hf": 4096,
            "meta-llama/Llama-2-13b-hf": 4096,
            "meta-llama/Llama-2-70b-hf": 4096,
            "microsoft/DialoGPT-medium": 1024,
            "microsoft/DialoGPT-large": 1024,
            "microsoft/DialoGPT-small": 1024,
            "EleutherAI/gpt-neo-2.7B": 2048,
            "EleutherAI/gpt-neo-1.3B": 2048,
            "EleutherAI/gpt-j-6B": 2048,
        }
        
        # Check for exact match first
        if model_name in context_lengths:
            return context_lengths[model_name]
        
        # Check for partial matches
        for model_prefix, length in context_lengths.items():
            if model_name.startswith(model_prefix):
                return length
        
        # Default fallback for vLLM models
        return 4096

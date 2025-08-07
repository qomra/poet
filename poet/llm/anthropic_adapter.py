# poet/llm/anthropic_adapter.py

import json
import time
from typing import Optional, Dict, Any
import logging

from .base_llm import BaseLLM, LLMConfig, LLMResponse, LLMError, LLMConnectionError, LLMTimeoutError, LLMRateLimitError, LLMInvalidRequestError

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

class AnthropicAdapter(BaseLLM):
    """Anthropic LLM adapter using the official Anthropic Python client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not ANTHROPIC_AVAILABLE:
            raise LLMError("Anthropic package not installed. Install with: pip install anthropic")
        
        if not config.api_key:
            raise LLMError("Anthropic API key is required")
        
        # Initialize Anthropic client
        # Note: Anthropic doesn't use base_url parameter, it's handled internally
        self.client = anthropic.Anthropic(
            api_key=config.api_key,
            timeout=config.timeout
        )
        
        self.logger.info(f"Initialized Anthropic adapter with model: {config.model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API."""
        response = self.generate_with_metadata(prompt, **kwargs)
        return response.content
    
    def generate_with_metadata(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with full metadata using Anthropic API."""
        try:
            # Merge parameters (this now filters out unsupported parameters)
            params = self._merge_params(**kwargs)
            
            # Make API call
            self.logger.debug(f"Making Anthropic API call with model: {params['model']}")
            start_time = time.time()
            
            # Prepare API parameters
            api_params = {
                "model": params["model"],
                "max_tokens": params.get("max_tokens", 1000),  # Default max_tokens for Anthropic
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add all other parameters from merged params (they're already filtered)
            for k, v in params.items():
                if k not in ["model", "max_tokens", "messages"] and v is not None:
                    api_params[k] = v
            
            response = self.client.messages.create(**api_params)
            
            end_time = time.time()
            
            # Extract response data
            content = response.content[0].text if response.content else ""
            
            # Build usage info
            usage = None
            if response.usage:
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                }
            
            # Build metadata
            metadata = {
                "response_time": end_time - start_time,
                "model": response.model,
                "id": response.id,
                "type": response.type,
                "role": response.role,
                "stop_reason": response.stop_reason,
                "stop_sequence": response.stop_sequence
            }
            
            self.logger.debug(f"Anthropic API call completed in {metadata['response_time']:.2f}s")
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason,
                metadata=metadata
            )
            
        except anthropic.AuthenticationError as e:
            self.logger.error(f"Anthropic authentication error: {e}")
            raise LLMConnectionError(f"Authentication failed: {e}")
        
        except anthropic.RateLimitError as e:
            self.logger.error(f"Anthropic rate limit error: {e}")
            raise LLMRateLimitError(f"Rate limit exceeded: {e}")
        
        except anthropic.APITimeoutError as e:
            self.logger.error(f"Anthropic timeout error: {e}")
            raise LLMTimeoutError(f"Request timed out: {e}")
        
        except anthropic.BadRequestError as e:
            self.logger.error(f"Anthropic bad request error: {e}")
            raise LLMInvalidRequestError(f"Invalid request: {e}")
        
        except anthropic.APIConnectionError as e:
            self.logger.error(f"Anthropic connection error: {e}")
            raise LLMConnectionError(f"Connection failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected Anthropic error: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def is_available(self) -> bool:
        """Check if Anthropic service is available."""
        if not ANTHROPIC_AVAILABLE:
            return False
        
        try:
            # Make a minimal API call to check availability
            response = self.client.messages.create(
                model=self.config.model_name,
                max_tokens=1,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            self.logger.warning(f"Anthropic availability check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = super().get_model_info()
        info.update({
            "provider": "anthropic",
            "supports_streaming": True,
            "supports_functions": True,
            "context_length": self._get_context_length(self.config.model_name)
        })
        return info
    
    def _get_context_length(self, model_name: str) -> int:
        """Get context length for Anthropic models."""
        context_lengths = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
            "claude-3-5-sonnet-20241022": 200000,
            "claude-3-5-haiku-20241022": 200000,
            "claude-2.1": 200000,
            "claude-2.0": 100000,
            "claude-instant-1.2": 100000,
        }
        
        # Check for exact match first
        if model_name in context_lengths:
            return context_lengths[model_name]
        
        # Check for partial matches
        for model_prefix, length in context_lengths.items():
            if model_name.startswith(model_prefix):
                return length
        
        # Default fallback
        return 100000
    
    def _merge_params(self, **kwargs) -> Dict[str, Any]:
        """
        Merge configuration with runtime parameters, filtering out unsupported parameters.
        
        Args:
            **kwargs: Runtime parameters
            
        Returns:
            Merged parameters dictionary with only Anthropic-supported parameters
        """
        # Start with basic parameters that Anthropic supports
        params = {
            'model': self.config.model_name,
        }
        
        # Add supported parameters only if they're not None
        if self.config.temperature is not None:
            params['temperature'] = self.config.temperature
        if self.config.max_tokens is not None:
            params['max_tokens'] = self.config.max_tokens
        if self.config.top_p is not None:
            params['top_p'] = self.config.top_p
        
        # Add extra params from config (filter out unsupported ones)
        for k, v in self.config.extra_params.items():
            if k not in ['frequency_penalty', 'presence_penalty'] and v is not None:
                params[k] = v
        
        # Override with runtime kwargs (filter out unsupported ones)
        for k, v in kwargs.items():
            if k not in ['frequency_penalty', 'presence_penalty'] and v is not None:
                params[k] = v
        
        return params

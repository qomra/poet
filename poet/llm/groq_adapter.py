# poet/llm/groq_adapter.py

import json
import time
from typing import Optional, Dict, Any
import logging

from .base_llm import BaseLLM, LLMConfig, LLMResponse, LLMError, LLMConnectionError, LLMTimeoutError, LLMRateLimitError, LLMInvalidRequestError

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    groq = None

class GroqAdapter(BaseLLM):
    """Groq LLM adapter using the official Groq Python client."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        
        if not GROQ_AVAILABLE:
            raise LLMError("Groq package not installed. Install with: pip install groq")
        
        if not config.api_key:
            raise LLMError("Groq API key is required")
        
        # Initialize Groq client
        self.client = groq.Groq(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout
        )
        
        self.logger.info(f"Initialized Groq adapter with model: {config.model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Groq API."""
        response = self.generate_with_metadata(prompt, **kwargs)
        return response.content
    
    def generate_with_metadata(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response with full metadata using Groq API."""
        try:
            # Merge parameters
            params = self._merge_params(**kwargs)
            
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # # Log the input prompt for debugging
            # self.logger.info(f"=== GROQ INPUT PROMPT ===")
            # self.logger.info(f"Model: {params['model']}")
            # self.logger.info(f"Temperature: {kwargs.get('temperature', 'default')}")
            # self.logger.info(f"Max Tokens: {kwargs.get('max_tokens', 'default')}")
            # self.logger.info(f"Prompt Length: {len(prompt)} characters")
            # self.logger.info(f"Prompt Preview: {prompt[:200]}...")
            # self.logger.info(f"Full Prompt:")
            # self.logger.info(prompt)
            # self.logger.info(f"=== END INPUT PROMPT ===")
            
            # Make API call
            self.logger.debug(f"Making Groq API call with model: {params['model']}")
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
            
            # Log the output response for debugging
            # self.logger.info(f"=== GROQ OUTPUT RESPONSE ===")
            # self.logger.info(f"Response Length: {len(content)} characters")
            # self.logger.info(f"Response Preview: {content[:200]}...")
            # self.logger.info(f"Full Response:")
            # self.logger.info(content)
            # self.logger.info(f"=== END OUTPUT RESPONSE ===")
            
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
                "id": response.id,
                "created": getattr(response, 'created', None),
                "system_fingerprint": getattr(response, 'system_fingerprint', None)
            }
            
            self.logger.debug(f"Groq API call completed in {metadata['response_time']:.2f}s")
            
            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason,
                metadata=metadata
            )
            
        except groq.AuthenticationError as e:
            self.logger.error(f"Groq authentication error: {e}")
            raise LLMConnectionError(f"Authentication failed: {e}")
        
        except groq.RateLimitError as e:
            self.logger.error(f"Groq rate limit error: {e}")
            raise LLMRateLimitError(f"Rate limit exceeded: {e}")
        
        except groq.APITimeoutError as e:
            self.logger.error(f"Groq timeout error: {e}")
            raise LLMTimeoutError(f"Request timed out: {e}")
        
        except groq.BadRequestError as e:
            self.logger.error(f"Groq bad request error: {e}")
            raise LLMInvalidRequestError(f"Invalid request: {e}")
        
        except groq.APIConnectionError as e:
            self.logger.error(f"Groq connection error: {e}")
            raise LLMConnectionError(f"Connection failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected Groq error: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def is_available(self) -> bool:
        """Check if Groq service is available."""
        if not GROQ_AVAILABLE:
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
            self.logger.warning(f"Groq availability check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = super().get_model_info()
        info.update({
            "provider": "groq",
            "supports_streaming": True,
            "supports_functions": False,  # Groq doesn't support function calling yet
            "context_length": self._get_context_length(self.config.model_name)
        })
        return info
    
    def _get_context_length(self, model_name: str) -> int:
        """Get context length for Groq models."""
        context_lengths = {
            "llama3-8b-8192": 8192,
            "llama3-70b-8192": 8192,
            "mixtral-8x7b-32768": 32768,
            "gemma2-9b-it": 8192,
            "llama3.1-8b-instant": 8192,
            "llama3.1-70b-versatile": 8192,
            "llama3.1-405b-reasoning": 8192,
            "llama3.1-8b": 8192,
            "llama3.1-70b": 8192,
            "llama3.1-405b": 8192,
        }
        
        # Check for exact match first
        if model_name in context_lengths:
            return context_lengths[model_name]
        
        # Check for partial matches
        for model_prefix, length in context_lengths.items():
            if model_name.startswith(model_prefix):
                return length
        
        # Default fallback for Groq models
        return 8192

# poet/llm/base_llm.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import logging

@dataclass
class LLMConfig:
    """Configuration for LLM providers"""
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 320
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}

@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class LLMError(Exception):
    """Base exception for LLM-related errors"""
    pass

class LLMConnectionError(LLMError):
    """Raised when connection to LLM provider fails"""
    pass

class LLMTimeoutError(LLMError):
    """Raised when LLM request times out"""
    pass

class LLMRateLimitError(LLMError):
    """Raised when rate limit is exceeded"""
    pass

class LLMInvalidRequestError(LLMError):
    """Raised when request is invalid"""
    pass

class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    Defines the interface that all LLM providers must implement.
    Provides common functionality and error handling.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._validate_config()
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text response from prompt.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters to override config
            
        Returns:
            Generated text response
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    def generate_with_metadata(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate text response with metadata.
        
        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters to override config
            
        Returns:
            LLMResponse with content and metadata
            
        Raises:
            LLMError: If generation fails
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available.
        
        Returns:
            True if provider is available, False otherwise
        """
        pass
    
    def _validate_config(self):
        """Validate the configuration"""
        if not self.config.model_name:
            raise ValueError("model_name is required")
        
        if not 0 <= self.config.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        
        if not 0 <= self.config.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        
        if self.config.max_tokens is not None and self.config.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
    
    def _merge_params(self, **kwargs) -> Dict[str, Any]:
        """
        Merge configuration with runtime parameters.
        
        Args:
            **kwargs: Runtime parameters
            
        Returns:
            Merged parameters dictionary
        """
        params = {
            'model': self.config.model_name,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'frequency_penalty': self.config.frequency_penalty,
            'presence_penalty': self.config.presence_penalty,
        }
        
        if self.config.max_tokens:
            params['max_tokens'] = self.config.max_tokens
        
        # Add extra params from config
        params.update(self.config.extra_params)
        
        # Override with runtime kwargs
        params.update(kwargs)
        
        return params
    
    def _handle_error(self, error: Exception, operation: str = "generation") -> None:
        """
        Handle and re-raise errors with appropriate types.
        
        Args:
            error: Original exception
            operation: Operation that failed
            
        Raises:
            Appropriate LLMError subclass
        """
        error_msg = f"LLM {operation} failed: {str(error)}"
        self.logger.error(error_msg)
        
        # Convert common errors to specific types
        if "timeout" in str(error).lower():
            raise LLMTimeoutError(error_msg) from error
        elif "rate limit" in str(error).lower() or "quota" in str(error).lower():
            raise LLMRateLimitError(error_msg) from error
        elif "connection" in str(error).lower() or "network" in str(error).lower():
            raise LLMConnectionError(error_msg) from error
        elif "invalid" in str(error).lower() or "bad request" in str(error).lower():
            raise LLMInvalidRequestError(error_msg) from error
        else:
            raise LLMError(error_msg) from error
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'provider': self.__class__.__name__,
            'model': self.config.model_name,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'available': self.is_available()
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.__class__.__name__}(model={self.config.model_name})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return (f"{self.__class__.__name__}(model={self.config.model_name}, "
                f"temperature={self.config.temperature}, "
                f"max_tokens={self.config.max_tokens})")


class MockLLM(BaseLLM):
    """
    Mock LLM implementation for testing.
    
    Returns predefined responses or echoes the prompt.
    """
    
    def __init__(self, config: LLMConfig, responses: Optional[List[str]] = None):
        super().__init__(config)
        self.responses = responses or []
        self.call_count = 0
        self.last_prompt = None
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        self.last_prompt = prompt
        self.call_count += 1
        
        # Debug: Print first 200 characters of prompt to see what we're getting
        print(f"MockLLM call {self.call_count}: {prompt[:200]}...")
        
        if self.responses:
            # Cycle through predefined responses
            response = self.responses[(self.call_count - 1) % len(self.responses)]
        else:
            # Return appropriate mock responses based on prompt content
            if "constraint" in prompt.lower() or "parse" in prompt.lower():
                # Mock constraint parsing response
                response = '''{
  "meter": "بحر الكامل",
  "qafiya": "ق",
  "line_count": 2,
  "theme": "غزل",
  "tone": "حزينة",
  "language": "فصحى",
  "style": "كلاسيكي",
  "imagery": ["الدموع", "الفراق", "القلب النابض"],
  "keywords": ["غزل", "فراق", "حزن", "دموع"],
  "register": "فصحى",
  "era": "كلاسيكي",
  "poet_style": "عاطفي",
  "sections": ["مقدمة", "موضوع", "خاتمة"],
  "ambiguities": [],
  "suggestions": ["استخدام صور قوية", "تركيز على العاطفة"],
  "reasoning": "الطلب يتعلق بموضوع الغزل والفراق مع التركيز على العاطفة"
}'''
            elif "qafiya" in prompt.lower() or "qafiya_completion" in prompt.lower() or "qafiya_selection" in prompt.lower():
                # Mock qafiya selection/completion response
                response = '''{
  "qafiya_letter": "ق",
  "qafiya_type": "مقيدة",
  "qafiya_harakah": "مفتوح"
}'''
            elif "selection" in prompt.lower():
                # Mock selection response
                response = '''{
  "selected_candidate": 0,
  "reasoning": "This candidate shows the best quality and meets all criteria",
  "criterion_scores": {
    "overall_quality": 8,
    "meter_accuracy": 9,
    "qafiya_accuracy": 8
  }
}'''
            elif "refinement" in prompt.lower() or "prosody_refinement" in prompt.lower():
                # Mock prosody refinement response - generate improved poem
                response = '''يَا مُهَلِّمٌ لِلمُشتاقينَ إِذا لَقِيتَ
حَيَّيتَ مِنهُم بِالوَفاءِ لِلمُصافِي قِفاً
وَأَيُّ مُصافٍ مِن مَراشِفِه يَعصى
غَيرَكَ وَما أَحسَنَ المُصافِي الوِقافِ
وَلَكنَّ قَلبِي في حُبِّكَ مُشتَعِلٌ
يُذَكِّرُني بِالوَصلِ كُلَّ خِلافِ'''
            elif "qafiya_refinement" in prompt.lower():
                # Mock qafiya refinement response - generate improved poem
                response = '''يَا مُهَلِّمٌ لِلمُشتاقينَ إِذا لَقِيتَ
حَيَّيتَ مِنهُم بِالوَفاءِ لِلمُصافِي قِفاً
وَأَيُّ مُصافٍ مِن مَراشِفِه يَعصى
غَيرَكَ وَما أَحسَنَ المُصافِي الوِقافِ
وَلَكنَّ قَلبِي في حُبِّكَ مُشتَعِلٌ
يُذَكِّرُني بِالوَصلِ كُلَّ خِلافِ'''
            elif "evaluation" in prompt.lower():
                # Mock evaluation response - indicate poem needs refinement
                response = '''{
  "overall_score": 0.65,
  "prosody_validation": {
    "overall_valid": false,
    "bait_results": [
      {"is_valid": false, "details": "Verse too short, needs more content"},
      {"is_valid": false, "details": "Incomplete bait structure"}
    ]
  },
  "qafiya_validation": {
    "overall_valid": false,
    "bait_results": [
      {"is_valid": false, "details": "Qafiya not properly developed"},
      {"is_valid": false, "details": "Rhyme scheme incomplete"}
    ]
  }
}'''
            else:
                # Generic mock response
                response = f"Mock response for: {prompt[:50]}..."
        
        return response
    
    def generate_with_metadata(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response with metadata"""
        content = self.generate(prompt, **kwargs)
        
        return LLMResponse(
            content=content,
            model=self.config.model_name,
            usage={'prompt_tokens': len(prompt.split()), 'completion_tokens': len(content.split())},
            finish_reason='stop',
            metadata={'mock': True, 'call_count': self.call_count}
        )
    
    def is_available(self) -> bool:
        """Mock is always available"""
        return True
    
    def reset(self):
        """Reset mock state"""
        self.call_count = 0
        self.last_prompt = None

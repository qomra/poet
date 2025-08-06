# poet/generation/poem_generator.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
import json
from poet.models.constraints import UserConstraints
from poet.models.poem import LLMPoem
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class BasePoemGenerator(ABC):
    """
    Abstract base class for poem generators.
    
    Defines the interface that all poem generators must implement.
    Provides common functionality for generating Arabic poetry based on constraints.
    """
    
    def __init__(self, llm_provider: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm_provider
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_poem(self, constraints: UserConstraints) -> LLMPoem:
        """
        Generate a poem based on the given constraints.
        
        Args:
            constraints: UserConstraints object specifying poem requirements
            
        Returns:
            LLMPoem object containing the generated poem
            
        Raises:
            GenerationError: If poem generation fails
        """
        pass
    
    @abstractmethod
    def can_handle_constraints(self, constraints: UserConstraints) -> bool:
        """
        Check if this generator can handle the given constraints.
        
        Args:
            constraints: UserConstraints to check
            
        Returns:
            True if this generator can handle the constraints, False otherwise
        """
        pass


class SimplePoemGenerator(BasePoemGenerator):
    """
    Simple poem generator that focuses on prosody and qafiya compliance.
    
    Generates Arabic poetry attempting to follow specified meter and rhyme patterns.
    This is a basic implementation for testing prosody validation.
    """
    
    def __init__(self, llm_provider: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        super().__init__(llm_provider, prompt_manager)
    
    def generate_poem(self, constraints: UserConstraints) -> LLMPoem:
        """
        Generate a simple poem based on the given constraints.
        
        Args:
            constraints: UserConstraints object specifying poem requirements
            
        Returns:
            LLMPoem object containing the generated poem
            
        Raises:
            GenerationError: If poem generation fails
        """
        try:
            # Calculate verse count (each bait = 2 verses)
            line_count = constraints.line_count or 4
            verse_count = line_count * 2
            
            # Format the generation prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'simple_poem_generation',
                meter=constraints.meter or "غير محدد",
                qafiya=constraints.qafiya or "غير محدد",
                line_count=line_count,
                verse_count=verse_count
            )
            
            # Generate poem using LLM
            response = self.llm.generate(formatted_prompt)
            
            # Parse the response to extract verses
            verses = self._parse_llm_response(response)
            
            # Create LLMPoem object
            poem = LLMPoem(
                verses=verses,
                llm_provider=self.llm.__class__.__name__,
                model_name=getattr(self.llm.config, 'model_name', 'unknown'),
                constraints=constraints.to_dict()
            )
            
            return poem
            
        except Exception as e:
            self.logger.error(f"Failed to generate poem: {e}")
            raise GenerationError(f"Poem generation failed: {e}")
    
    def can_handle_constraints(self, constraints: UserConstraints) -> bool:
        """
        Check if this generator can handle the given constraints.
        
        SimplePoemGenerator can handle basic constraints but may not
        produce sophisticated imagery or complex thematic development.
        """
        # Can handle basic constraints
        return True
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract verses from JSON structure.
        
        Args:
            response: Raw LLM response containing JSON with verses array
            
        Returns:
            List of verses (strings)
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Extract verses array
            if 'verses' not in data:
                raise ValueError("No 'verses' key found in JSON response")
            
            verses = data['verses']
            
            # Validate verses
            if not isinstance(verses, list):
                raise ValueError("'verses' must be a list")
            
            if not verses:
                raise ValueError("Verses list is empty")
            
            # Ensure all verses are strings
            verses = [str(verse).strip() for verse in verses if verse and str(verse).strip()]
            
            if not verses:
                raise ValueError("No valid verses found after processing")
            
            return verses
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise GenerationError(f"Invalid JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise GenerationError(f"Response parsing failed: {e}")


class GenerationError(Exception):
    """Raised when poem generation fails"""
    pass

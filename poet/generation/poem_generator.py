# poet/generation/poem_generator.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import logging
import json
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.llm.base_llm import BaseLLM
from poet.prompts import get_global_prompt_manager
from poet.core.node import Node
from datetime import datetime


class BasePoemGenerator(ABC):
    """
    Abstract base class for poem generators.
    
    Defines the interface that all poem generators must implement.
    Provides common functionality for generating Arabic poetry based on constraints.
    """
    
    def __init__(self, llm_provider: BaseLLM, **kwargs):
        self.llm = llm_provider
        # Use global prompt manager instead of creating new instance
        self.prompt_manager = get_global_prompt_manager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_poem(self, constraints: Constraints) -> LLMPoem:
        """
        Generate a poem based on the given constraints.
        
        Args:
            constraints: Constraints object specifying poem requirements
            
        Returns:
            LLMPoem object containing the generated poem
            
        Raises:
            GenerationError: If poem generation fails
        """
        pass
    
    @abstractmethod
    def can_handle_constraints(self, constraints: Constraints) -> bool:
        """
        Check if this generator can handle the given constraints.
        
        Args:
            constraints: Constraints to check
            
        Returns:
            True if this generator can handle the constraints, False otherwise
        """
        pass


class SimplePoemGenerator(BasePoemGenerator, Node):
    """
    Simple poem generator that focuses on prosody and qafiya compliance.
    
    Generates Arabic poetry attempting to follow specified meter and rhyme patterns.
    This is a basic implementation for testing prosody validation.
    """
    
    def __init__(self, llm: BaseLLM, **kwargs):
        BasePoemGenerator.__init__(self, llm)
        Node.__init__(self, **kwargs)
        # Override with global prompt manager
        self.prompt_manager = get_global_prompt_manager()
    
    def generate_poem(self, constraints: Constraints) -> LLMPoem:
        """Generate a poem based on the given constraints."""
        try:
            # Format the prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'simple_poem_generation',
                meter=constraints.meter or "غير محدد",
                meeter_tafeelat=constraints.meeter_tafeelat or "غير محدد",
                qafiya=constraints.qafiya or "غير محدد",
                qafiya_type=constraints.qafiya_type or "غير محدد",
                qafiya_type_description_and_examples=constraints.qafiya_type_description_and_examples or "غير محدد",
                qafiya_harakah=constraints.qafiya_harakah or "",
                theme=constraints.theme or "غير محدد",
                tone=constraints.tone or "غير محدد",
                line_count=constraints.line_count or 4,
                verse_count=(constraints.line_count or 4) * 2,  # Each bait = 2 verses
                imagery=constraints.imagery or [],
                keywords=constraints.keywords or [],
                sections=constraints.sections or [],
                register=constraints.register or "فصيح",
                era=constraints.era or "كلاسيكي",
                poet_style=constraints.poet_style or "غير محدد"
            )
            
            # Try multiple times to generate a valid poem
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Generate poem using LLM
                    response = self.llm.generate(formatted_prompt)
                    
                    # Parse the response to extract verses
                    verses = self._parse_llm_response(response)
                    
                    # If we get here, parsing succeeded
                    break
                    
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                        continue
                    else:
                        # All retries failed, raise the last error
                        raise last_error
            
            # Create and return the poem
            poem = LLMPoem(
                verses=verses,
                llm_provider=self.llm.__class__.__name__,
                model_name=getattr(self.llm, 'model_name', 'unknown'),
                constraints=constraints,
                generation_timestamp=datetime.now()
            )
            
            return poem
            
        except Exception as e:
            raise GenerationError(f"Poem generation failed: {e}")
    
    def can_handle_constraints(self, constraints: Constraints) -> bool:
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
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the poem generation node.
        
        Args:
            input_data: Input data containing constraints
            context: Pipeline context with LLM and prompt_manager
            
        Returns:
            Output data with generated poem
        """
        # Set up context
        self.llm = context.get('llm')
        # Use global prompt manager instead of context or new instance
        self.prompt_manager = get_global_prompt_manager()
        
        if not self.llm:
            raise ValueError("LLM not provided in context")
        
        # Extract required data
        constraints = input_data.get('constraints')
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Generate poem
        poem = self.generate_poem(constraints)
        
        return {
            'poem': poem,
            'poem_generated': True
        }
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'poem_generated']


class GenerationError(Exception):
    """Raised when poem generation fails"""
    pass

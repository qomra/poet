# poet/refinement/tashkeel_refiner.py

import logging
from typing import List, Optional
from poet.models.poem import LLMPoem
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class TashkeelRefiner:
    """
    Applies Arabic diacritics (tashkeel) to poem verses using LLM.
    
    This refiner enhances the poem by adding proper diacritics
    which are essential for prosody and qafiya validation.
    """
    
    def __init__(self, llm_provider: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm_provider
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(__name__)
    
    def apply_tashkeel(self, poem: LLMPoem) -> LLMPoem:
        """
        Apply diacritics to all verses in the poem.
        
        Args:
            poem: LLMPoem object to refine
            
        Returns:
            LLMPoem with diacritized verses
        """
        try:
            if not poem.verses:
                self.logger.warning("No verses to apply tashkeel to")
                return poem
            
            # Format the tashkeel prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'tashkeel',
                text='\n'.join(poem.verses)
            )
            
            # Get LLM response
            response = self.llm.generate(formatted_prompt)
            
            # Parse the structured response
            diacritized_verses = self._parse_llm_response(response)
            
            # Clean diacritics by removing haraka-shaddah or shaddah-haraka sequences
            cleaned_verses = [self._clean_diacritics(verse) for verse in diacritized_verses]
            
            # Create new poem with diacritized verses
            refined_poem = LLMPoem(
                verses=cleaned_verses,
                llm_provider=poem.llm_provider,
                model_name=poem.model_name,
                constraints=poem.constraints,
                generation_timestamp=poem.generation_timestamp,
                quality=poem.quality
            )
            
            return refined_poem
            
        except Exception as e:
            self.logger.error(f"Failed to apply tashkeel: {e}")
            # Return original poem if tashkeel fails
            return poem
    
    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse the structured JSON response from the LLM.
        
        Args:
            response: Raw LLM response containing JSON
            
        Returns:
            List of diacritized verses
            
        Raises:
            ValueError: If JSON parsing fails
        """
        import json
        
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate required structure
            if 'diacritized_verses' not in data:
                raise ValueError("Missing 'diacritized_verses' field")
            
            if not isinstance(data['diacritized_verses'], list):
                raise ValueError("'diacritized_verses' must be a list")
            
            return data['diacritized_verses']
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Invalid JSON response: {e}")
        except ValueError as e:
            self.logger.error(f"Invalid response format: {e}")
            raise ValueError(f"Invalid response format: {e}")
    
    def _clean_diacritics(self, text: str) -> str:
        """
        Clean diacritics by removing haraka-shaddah or shaddah-haraka sequences.
        Keep only the shaddah character.
        
        Args:
            text: Arabic text with diacritics
            
        Returns:
            Cleaned text with proper shaddah handling
        """
        # Define Arabic diacritics
        # kasra, fatha, damma, sukun, tanween fatha, tanween damma, tanween kasra
        harakat = ["\u0650", "\u064E", "\u064F", "\u0652", "\u064B", "\u064C", "\u064D"]
        shadda = "\u0651"
        
        # Remove haraka-shaddah sequences (haraka followed by shaddah)
        for haraka in harakat:
            text = text.replace(haraka + shadda, shadda)
        
        # Remove shaddah-haraka sequences (shaddah followed by haraka)
        for haraka in harakat:
            text = text.replace(shadda + haraka, shadda)
        
        return text 
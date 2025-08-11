# poet/refinement/tashkeel_refiner.py

import logging
from poet.refinement.base import BaseRefiner
from typing import List, Optional, Dict, Any
from poet.models.poem import LLMPoem
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager
from poet.models.quality import QualityAssessment
from poet.models.constraints import Constraints


class TashkeelRefiner(BaseRefiner):
    """
    Applies Arabic diacritics (tashkeel) to poem verses using LLM.
    
    This refiner enhances the poem by adding proper diacritics
    which are essential for prosody and qafiya validation.
    """
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger.setLevel(logging.INFO)
    
    @property
    def name(self) -> str:
        return "tashkeel_refiner"
    
    @name.setter
    def name(self, value: str):
        """Set the refiner name (ignored, always returns custom name)"""
        pass
    
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        """Check if tashkeel needs fixing"""
        if not evaluation.tashkeel_validation:
            return False
        return not evaluation.tashkeel_validation.overall_valid
            
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """Apply diacritics to all verses in the poem"""
        return await self._apply_tashkeel(poem)
    
    async def _apply_tashkeel(self, poem: LLMPoem) -> LLMPoem:
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
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tashkeel refiner node.
        
        Args:
            input_data: Input data containing poem and constraints
            context: Pipeline context
            
        Returns:
            Output data with refined poem
        """
        # Set up context
        self.llm = context.get('llm')
        self.prompt_manager = context.get('prompt_manager') or PromptManager()
        
        if not self.llm:
            raise ValueError("LLM not provided in context")
        
        # Extract required data
        poem = input_data.get('poem')
        constraints = input_data.get('constraints')
        evaluation = input_data.get('evaluation')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Check if refinement is needed
        if evaluation and not self.should_refine(evaluation):
            self.logger.info(f"{self.name}: No tashkeel refinement needed")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
        
        # Apply refinement
        try:
            refined_poem = self._apply_sync_refinement(poem, constraints, evaluation)
            
            self.logger.info(f"{self.name}: Tashkeel refinement applied successfully")
            return {
                'poem': refined_poem,
                'refined': True,
                'refinement_iterations': 1
            }
        except Exception as e:
            self.logger.error(f"{self.name}: Tashkeel refinement failed: {e}")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
    
    def _apply_sync_refinement(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """
        Apply tashkeel refinement synchronously.
        """
        if not evaluation.tashkeel_validation:
            return poem
        
        # Get broken verses from evaluation
        broken_verses = self._identify_broken_verses(evaluation.tashkeel_validation)
        
        if not broken_verses:
            return poem
        
        self.logger.info(f"Fixing tashkeel for {len(broken_verses)} broken verses")
        
        # Fix each broken verse
        fixed_verses = poem.verses.copy()
        for verse_index, error_details in broken_verses:
            fixed_verse = self._fix_single_verse_sync(
                poem.verses[verse_index], 
                constraints, 
                error_details
            )
            if fixed_verse:
                fixed_verses[verse_index] = fixed_verse
            else:
                self.logger.warning(f"Verse {verse_index} is not yet tashkeel fixed. Using original verse.")
                
        # Create new poem
        return LLMPoem(
            verses=fixed_verses,
            llm_provider=poem.llm_provider,
            model_name=poem.model_name,
            constraints=poem.constraints,
            generation_timestamp=poem.generation_timestamp
        )
    
    def _identify_broken_verses(self, tashkeel_validation) -> List[tuple]:
        """Identify verses with tashkeel violations"""
        broken_verses = []
        
        if not hasattr(tashkeel_validation, 'verse_results') or tashkeel_validation.verse_results is None:
            return broken_verses
        
        for i, verse_result in enumerate(tashkeel_validation.verse_results):
            if not verse_result.is_valid:
                broken_verses.append((i, verse_result.error_details))
        
        return broken_verses
    
    def _fix_single_verse_sync(self, verse: str, constraints: Constraints, error_details: str) -> Optional[str]:
        """Fix a single verse's tashkeel synchronously"""
        # Format prompt for fixing verse
        formatted_prompt = self.prompt_manager.format_prompt(
            'tashkeel',  # Use the evaluation prompt since there's no refinement prompt
            meter=constraints.meter or "غير محدد",
            qafiya=constraints.qafiya or "غير محدد",
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            existing_verse=verse,
            context=f"إصلاح التشكيل. المشكلة: {error_details}"
        )
        
        # Generate fixed verse
        response = self.llm.generate(formatted_prompt)
        fixed_verses = self._parse_verses_from_response(response)
        
        return fixed_verses[0] if fixed_verses else None
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints', 'evaluation']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations']
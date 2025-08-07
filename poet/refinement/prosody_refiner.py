# poet/refinement/prosody_refiner.py

import logging
from typing import List, Optional
from poet.refinement.base_refiner import BaseRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class ProsodyRefiner(BaseRefiner):
    """Fixes meter violations in verses"""
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        return "prosody_refiner"
    
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        """Check if prosody needs fixing"""
        if not evaluation.prosody_validation:
            return False
        return not evaluation.prosody_validation.overall_valid
    
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """Fix meter violations in verses"""
        try:
            if not evaluation.prosody_validation:
                return poem
            
            # Get broken verses from evaluation
            broken_verses = self._identify_broken_verses(evaluation.prosody_validation)
            
            if not broken_verses:
                return poem
            
            self.logger.info(f"Fixing prosody for {len(broken_verses)} broken verses")
            
            # Fix each broken verse
            fixed_verses = poem.verses.copy()
            for verse_index, error_details in broken_verses:
                fixed_verse = await self._fix_single_verse(
                    poem.verses[verse_index], 
                    constraints, 
                    error_details
                )
                fixed_verses[verse_index] = fixed_verse
            
            # Create new poem
            return LLMPoem(
                verses=fixed_verses,
                llm_provider=poem.llm_provider,
                model_name=poem.model_name,
                constraints=poem.constraints,
                generation_timestamp=poem.generation_timestamp
            )
                
        except Exception as e:
            self.logger.error(f"Failed to refine prosody: {e}")
            return poem  # Return original poem if refinement fails
    
    def _identify_broken_verses(self, prosody_validation) -> List[tuple]:
        """Identify verses with meter violations"""
        broken_verses = []
        
        if not hasattr(prosody_validation, 'bait_results') or prosody_validation.bait_results is None:
            return broken_verses
        
        for i, bait_result in enumerate(prosody_validation.bait_results):
            if not bait_result.is_valid:
                # Calculate verse indices for this bait
                bait_index = i
                first_verse_index = bait_index * 2
                
                # Since the new model doesn't have first_verse_valid/second_verse_valid,
                # we'll assume the first verse in the bait is broken if the bait is invalid
                broken_verses.append((first_verse_index, bait_result.error_details))
        
        return broken_verses
    
    async def _fix_single_verse(self, verse: str, constraints: Constraints, error_details: str) -> str:
        """Fix a single verse's meter"""
        # Format prompt for fixing verse
        formatted_prompt = self.prompt_manager.format_prompt(
            'prosody_refinement',
            meter=constraints.meter or "غير محدد",
            qafiya=constraints.qafiya or "غير محدد",
            qafiya_pattern=constraints.qafiya_pattern or "",
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            existing_verses=verse,
            context=f"إصلاح الوزن العروضي للبيت. المشكلة: {error_details}"
        )
        
        # Generate fixed verse
        response = self.llm.generate(formatted_prompt)
        fixed_verses = self._parse_verses_from_response(response)
        
        # Return the first fixed verse, or original if parsing failed
        return fixed_verses[0] if fixed_verses else verse
    
    def _parse_verses_from_response(self, response: str) -> List[str]:
        """Parse verses from LLM response"""
        try:
            # Extract JSON from response
            import json
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # Fallback: split by newlines
                return [line.strip() for line in response.split('\n') if line.strip()]
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            if 'verses' in data:
                return data['verses']
            else:
                # Fallback: split by newlines
                return [line.strip() for line in response.split('\n') if line.strip()]
                
        except Exception as e:
            self.logger.error(f"Failed to parse verses from response: {e}")
            # Fallback: split by newlines
            return [line.strip() for line in response.split('\n') if line.strip()] 
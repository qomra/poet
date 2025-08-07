# poet/refinement/qafiya_refiner.py

import logging
from typing import List, Optional
from poet.refinement.base_refiner import BaseRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class QafiyaRefiner(BaseRefiner):
    """Fixes rhyme scheme violations"""
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        return "qafiya_refiner"
    
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        """Check if qafiya needs fixing"""
        if not evaluation.qafiya_validation:
            return False
        return not evaluation.qafiya_validation.overall_valid
    
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """Fix rhyme scheme violations"""
        try:
            if not evaluation.qafiya_validation:
                return poem
            
            # Get verses with wrong qafiya from evaluation
            wrong_qafiya_verses = self._identify_wrong_qafiya_verses(evaluation.qafiya_validation)
            
            if not wrong_qafiya_verses:
                return poem
            
            self.logger.info(f"Fixing qafiya for {len(wrong_qafiya_verses)} verses")
            
            # Fix each verse with wrong qafiya
            fixed_verses = poem.verses.copy()
            for verse_index, expected_qafiya in wrong_qafiya_verses:
                fixed_verse = await self._fix_single_verse_qafiya(
                    poem.verses[verse_index], 
                    constraints, 
                    expected_qafiya
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
            self.logger.error(f"Failed to refine qafiya: {e}")
            return poem  # Return original poem if refinement fails
    
    def _identify_wrong_qafiya_verses(self, qafiya_validation) -> List[tuple]:
        """Identify verses with wrong qafiya"""
        wrong_qafiya_verses = []
        
        if not hasattr(qafiya_validation, 'bait_results') or qafiya_validation.bait_results is None:
            return wrong_qafiya_verses
        
        for i, bait_result in enumerate(qafiya_validation.bait_results):
            if not bait_result.is_valid:
                # Calculate verse indices for this bait
                bait_index = i
                first_verse_index = bait_index * 2
                
                # Since the new model doesn't have first_verse_valid/second_verse_valid,
                # we'll assume the first verse in the bait is wrong if the bait is invalid
                expected_qafiya = getattr(bait_result, 'expected_qafiya', '')
                wrong_qafiya_verses.append((first_verse_index, expected_qafiya))
        
        return wrong_qafiya_verses
    
    async def _fix_single_verse_qafiya(self, verse: str, constraints: Constraints, expected_qafiya: str) -> str:
        """Fix a single verse's qafiya"""
        # Format prompt for fixing verse qafiya
        formatted_prompt = self.prompt_manager.format_prompt(
            'qafiya_refinement',
            meter=constraints.meter or "غير محدد",
            qafiya=constraints.qafiya or "غير محدد",
            qafiya_pattern=constraints.qafiya_pattern or "",
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            existing_verses=verse,
            context=f"إصلاح القافية للبيت. القافية المطلوبة: {expected_qafiya}"
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
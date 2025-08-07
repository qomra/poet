# poet/refinement/prosody_refiner.py

import logging
from typing import List, Optional
from poet.refinement.base import BaseRefiner
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
        self.logger.setLevel(logging.INFO)
    
    @property
    def name(self) -> str:
        return "prosody_refiner"
    
    def should_refine(self,evaluation: QualityAssessment) -> bool:
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
            broken_bait = self._identify_broken_bait(evaluation.prosody_validation)
            
            if not broken_bait:
                return poem
            
            self.logger.info(f"Fixing prosody for {len(broken_bait)} broken verses")
            
            # Fix each broken verse
            fixed_verses = poem.verses.copy()
            for bait_index, error_details in broken_bait:
                bait = "#".join(poem.verses[bait_index*2:bait_index*2+2])
                fixed_bait = await self._fix_single_verse(
                    bait, 
                    constraints, 
                    error_details
                )
                if len(fixed_bait) == 2:
                    fixed_verses[bait_index*2] = fixed_bait[0]
                    fixed_verses[bait_index*2+1] = fixed_bait[1]
                else:
                    self.logger.warning(f"Bait {bait_index} is not yet prosody fixed. Using original verses.")
                    
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
    
    def _identify_broken_bait(self, prosody_validation) -> List[tuple]:
        """Identify verses with meter violations"""
        broken_bait = []
        
        if not hasattr(prosody_validation, 'bait_results') or prosody_validation.bait_results is None:
            return broken_bait
        
        for i, bait_result in enumerate(prosody_validation.bait_results):
            if not bait_result.is_valid:
                broken_bait.append((i, bait_result.error_details))
        
        return broken_bait
    
    async def _fix_single_verse(self, verse: str, constraints: Constraints, error_details: str) -> str:
        """Fix a single verse's meter"""
        # Format prompt for fixing verse
        formatted_prompt = self.prompt_manager.format_prompt(
            'prosody_refinement',
            meter=constraints.meter or "غير محدد",
            meeter_tafeelat=constraints.meeter_tafeelat or "غير محدد",
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
        
        return fixed_verses
    
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
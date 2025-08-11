# poet/refinement/line_count_refiner.py

import logging
from typing import List, Optional
from poet.refinement.base import BaseRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class LineCountRefiner(BaseRefiner):
    """Fixes poems with incorrect number of lines"""
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @property
    def name(self) -> str:
        return "line_count_refiner"
    
    def should_refine(self,evaluation: QualityAssessment) -> bool:
        """Check if line count needs fixing"""
        if not evaluation.line_count_validation:
            return False
        return not evaluation.line_count_validation.is_valid
    
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """Fix line count by adding or removing verses"""
        try:
            target_count = constraints.line_count or 4
            current_count = len(poem.verses)
            
            self.logger.info(f"Fixing line count: current={current_count}, target={target_count}")
            
            if current_count == target_count:
                return poem
            
            if current_count < target_count:
                return await self._add_verses(poem, constraints, target_count - current_count)
            else:
                return await self._remove_verses(poem, constraints, current_count - target_count)
                
        except Exception as e:
            self.logger.error(f"Failed to refine line count: {e}")
            return poem  # Return original poem if refinement fails

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
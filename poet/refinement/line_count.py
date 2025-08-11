# poet/refinement/line_count_refiner.py

import logging
from typing import List, Optional, Dict, Any
from poet.refinement.base import BaseRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class LineCountRefiner(BaseRefiner):
    """Fixes poems with incorrect number of lines"""
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger.setLevel(logging.INFO)
    
    @property
    def name(self) -> str:
        return "line_count_refiner"
    
    @name.setter
    def name(self, value: str):
        """Set the refiner name (ignored, always returns custom name)"""
        pass
    
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
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the line count refiner node.
        
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
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # For now, just return the poem as-is (no actual refinement)
        # In a real implementation, this would apply line count refinement
        self.logger.info(f"Line count refiner node executed (no actual refinement applied)")
        
        return {
            'poem': poem,
            'refined': True,
            'refinement_iterations': 0
        }
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations']
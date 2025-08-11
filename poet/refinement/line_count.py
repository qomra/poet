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
        evaluation = input_data.get('evaluation')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Check if refinement is needed
        if evaluation and not self.should_refine(evaluation):
            self.logger.info(f"{self.name}: No line count refinement needed")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
        
        # Apply refinement
        try:
            refined_poem = self._apply_sync_refinement(poem, constraints, evaluation)
            
            self.logger.info(f"{self.name}: Line count refinement applied successfully")
            return {
                'poem': refined_poem,
                'refined': True,
                'refinement_iterations': 1
            }
        except Exception as e:
            self.logger.error(f"{self.name}: Line count refinement failed: {e}")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
    
    def _apply_sync_refinement(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """
        Apply line count refinement synchronously.
        """
        if not evaluation.line_count_validation:
            return poem
        
        # Check if line count is correct
        if poem.evaluate_line_count():
            return poem
        
        self.logger.info("Fixing line count - ensuring even number of verses")
        
        # Get target line count from constraints or use default
        target_lines = getattr(constraints, 'line_count', 4)
        if target_lines % 2 != 0:
            target_lines = target_lines + 1  # Ensure even number
        
        current_lines = len(poem.verses)
        
        if current_lines < target_lines:
            # Add more verses
            additional_verses = self._generate_additional_verses(poem, constraints, target_lines - current_lines)
            fixed_verses = poem.verses + additional_verses
        elif current_lines > target_lines:
            # Remove extra verses (keep even number)
            target_lines = max(2, target_lines - (target_lines % 2))  # Ensure even
            fixed_verses = poem.verses[:target_lines]
        else:
            # Already correct
            return poem
        
        # Create new poem
        return LLMPoem(
            verses=fixed_verses,
            llm_provider=poem.llm_provider,
            model_name=poem.model_name,
            constraints=poem.constraints,
            generation_timestamp=poem.generation_timestamp
        )
    
    def _generate_additional_verses(self, poem: LLMPoem, constraints: Constraints, count: int) -> List[str]:
        """Generate additional verses to reach target line count"""
        if count <= 0:
            return []
        
        # Format prompt for generating additional verses
        formatted_prompt = self.prompt_manager.format_prompt(
            'line_count_refinement',
            meter=constraints.meter or "غير محدد",
            qafiya=constraints.qafiya or "غير محدد",
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            existing_verses="\n".join(poem.verses),
            additional_verses_needed=count,
            context="إضافة أبيات جديدة للحفاظ على الوزن والقافية"
        )
        
        # Generate additional verses
        response = self.llm.generate(formatted_prompt)
        additional_verses = self._parse_verses_from_response(response)
        
        # Ensure we get the right number of verses
        if len(additional_verses) >= count:
            return additional_verses[:count]
        else:
            # If not enough verses generated, pad with simple verses
            while len(additional_verses) < count:
                additional_verses.append("بيت إضافي للحفاظ على الوزن")
            return additional_verses
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints', 'evaluation']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations']
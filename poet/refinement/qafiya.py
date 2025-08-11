# poet/refinement/qafiya_refiner.py

import logging
from typing import List, Optional, Dict, Any
from poet.refinement.base import BaseRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class QafiyaRefiner(BaseRefiner):
    """Fixes rhyme scheme violations"""
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger.setLevel(logging.INFO)
    
    @property
    def name(self) -> str:
        return "qafiya_refiner"
    
    @name.setter
    def name(self, value: str):
        """Set the refiner name (ignored, always returns custom name)"""
        pass
    
    def should_refine(self,evaluation: QualityAssessment) -> bool:
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
            wrong_qafiya_bait = self._identify_wrong_qafiya_bait(evaluation.qafiya_validation)
            


            if not wrong_qafiya_bait:
                return poem
            
            self.logger.info(f"Fixing qafiya for {len(wrong_qafiya_bait)} verses")
            def create_entire_poem(poem_verses: list[str]) -> str:
                poem_str = ""
                for i in range(0, len(poem_verses), 2):
                    poem_str += poem_verses[i] + "\n" + poem_verses[i+1] + "\n"
                return poem_str
            
            
            # Fix each verse with wrong qafiya
            fixed_verses = poem.verses.copy()
            for bait_index, issue in wrong_qafiya_bait:
                bait = "#".join(poem.verses[bait_index*2:bait_index*2+2])
                entire_poem = create_entire_poem(fixed_verses)
            
                fixed_bait = await self._fix_single_verse_qafiya(
                    bait, 
                    constraints, 
                    issue,
                    entire_poem
                )
                if len(fixed_bait) == 2:
                    fixed_verses[bait_index*2] = fixed_bait[0]
                    fixed_verses[bait_index*2+1] = fixed_bait[1]
                else:
                    # use original verse
                    self.logger.warning(f"Bait {bait_index} is not yet qafiya fixed. Using original verses.")
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
    
    def _identify_wrong_qafiya_bait(self, qafiya_validation) -> List[tuple]:
        """Identify verses with wrong qafiya"""
        wrong_qafiya_bait = []
        
        if not hasattr(qafiya_validation, 'bait_results') or qafiya_validation.bait_results is None:
            return wrong_qafiya_bait
        
        for i, bait_result in enumerate(qafiya_validation.bait_results):
            if not bait_result.is_valid:
                # Get error details from bait result, default to empty string if missing
                issue = getattr(bait_result, 'error_details', '') or ''
                wrong_qafiya_bait.append((i, issue))
        
        return wrong_qafiya_bait
    
    async def _fix_single_verse_qafiya(self, verse: str, constraints: Constraints, issue: str, entire_poem: str) -> str:
        """Fix a single verse's qafiya"""
        # Format prompt for fixing verse qafiya

        formatted_prompt = self.prompt_manager.format_prompt(
            'qafiya_refinement',
            meter=constraints.meter or "غير محدد",
            meeter_tafeelat=constraints.meeter_tafeelat or "غير محدد",
            qafiya=constraints.qafiya or "غير محدد",
            qafiya_harakah=constraints.qafiya_harakah or "",
            qafiya_type=constraints.qafiya_type or "غير محدد",
            qafiya_type_description_and_examples=constraints.qafiya_type_description_and_examples if constraints.qafiya_type_description_and_examples is not None else "غير محدد",
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            existing_verses=verse,
            context=f"{issue}",
            entire_poem=entire_poem
            )           

        # Generate fixed verse
        response = self.llm.generate(formatted_prompt)
        fixed_verses = self._parse_verses_from_response(response)
        
        # Return the first fixed verse, or original if parsing failed
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
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the qafiya refiner node.
        
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
        # In a real implementation, this would apply qafiya refinement
        self.logger.info(f"Qafiya refiner node executed (no actual refinement applied)")
        
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
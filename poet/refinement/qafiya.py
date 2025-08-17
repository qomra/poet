# poet/refinement/qafiya.py

import logging
from typing import Optional, Dict, Any, List
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.prompts import get_global_prompt_manager
from poet.llm.base_llm import BaseLLM
from poet.core.node import Node
from poet.models.quality import QualityAssessment
import json


def create_entire_poem(poem_verses: list[str]) -> str:
    poem_str = ""
    for i in range(0, len(poem_verses), 2):
        poem_str += poem_verses[i] + "\n" + poem_verses[i+1] + "\n"
    return poem_str

class QafiyaRefiner(Node):
    """Fixes rhyme scheme violations"""
    
    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_manager = get_global_prompt_manager()
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
            
            self.logger.info(f"Fixing qafiya for {len(wrong_qafiya_bait)/2} baits")
            
            # Fix each verse with wrong qafiya
            fixed_verses = poem.verses.copy()
            for bait_index, issue in wrong_qafiya_bait:
                bait = "#".join(poem.verses[bait_index*2:bait_index*2+2])
                entire_poem = create_entire_poem(fixed_verses)
            
                fixed_bait = await self._fix_single_bait_qafiya(
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
    
    async def _fix_single_bait_qafiya(self, verse: str, constraints: Constraints, issue: str, entire_poem: str) -> str:
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
            # Debug: Log the raw response
            self.logger.debug(f"Raw LLM response: {response[:500]}...")
            
            # Extract JSON from response using robust parsing
            import json
            import re
            
            # First try to find JSON code blocks (most reliable)
            json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
            json_blocks = re.findall(json_block_pattern, response, re.DOTALL)
            
            if json_blocks:
                # Use the first JSON block found
                json_str = json_blocks[0]
                self.logger.debug(f"Found JSON in code block: {json_str[:200]}...")
            else:
                # Fallback: find first { and last } if no code block
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0 or json_end <= json_start:
                    self.logger.warning("No JSON found in response, falling back to line splitting")
                    # No valid JSON found, fallback to line splitting
                    return [line.strip() for line in response.split('\n') if line.strip()]
                
                json_str = response[json_start:json_end]
                self.logger.debug(f"Extracted JSON from response: {json_str[:200]}...")
            
            # Parse the JSON
            data = json.loads(json_str)
            
            if 'verses' in data:
                self.logger.debug(f"Successfully parsed {len(data['verses'])} verses from JSON")
                return data['verses']
            else:
                self.logger.warning("JSON parsed but no 'verses' field found")
                # Fallback: split by newlines
                return [line.strip() for line in response.split('\n') if line.strip()]
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Attempted to parse: {json_str[:200] if 'json_str' in locals() else 'N/A'}")
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
        self.prompt_manager = get_global_prompt_manager()
        
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
            self.logger.info(f"{self.name}: No qafiya refinement needed")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
        
        # Apply refinement using the same logic as async refine
        try:
            refined_poem = self._apply_sync_refinement(poem, constraints, evaluation)
            
            return {
                'poem': refined_poem,
                'refined': True,
                'refinement_iterations': 1
            }
        except Exception as e:
            self.logger.error(f"{self.name}: Qafiya refinement failed: {e}")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
    
    def _apply_sync_refinement(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """
        Apply qafiya refinement synchronously - same logic as async refine method.
        """
        if not evaluation.qafiya_validation:
            return poem
        
        # Get verses with wrong qafiya from evaluation - same logic as async method
        wrong_qafiya_bait = self._identify_broken_bait(evaluation.qafiya_validation)
        
        if not wrong_qafiya_bait:
            return poem
        
        self.logger.info(f"Fixing qafiya for {len(wrong_qafiya_bait)} baits")
        
        # Fix each bait with wrong qafiya - same logic as async method
        fixed_verses = poem.verses.copy()
        for bait_index, issue in wrong_qafiya_bait:
            bait = "#".join(poem.verses[bait_index*2:bait_index*2+2])
            entire_poem = create_entire_poem(fixed_verses)
            
            fixed_bait = self._fix_single_bait_sync(
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
    
    def _identify_broken_bait(self, qafiya_validation) -> List[tuple]:
        """Identify verses with qafiya violations - same logic as _identify_wrong_qafiya_bait"""
        wrong_qafiya_bait = []
        
        if not hasattr(qafiya_validation, 'bait_results') or qafiya_validation.bait_results is None:
            return wrong_qafiya_bait
        
        for i, bait_result in enumerate(qafiya_validation.bait_results):
            if not bait_result.is_valid:
                # Get error details from bait result, default to empty string if missing
                issue = getattr(bait_result, 'error_details', '') or ''
                wrong_qafiya_bait.append((i, issue))
        
        return wrong_qafiya_bait
    
    def _fix_single_bait_sync(self, verse: str, constraints: Constraints, error_details: str, entire_poem: str) -> List[str]:
        """Fix a single bait's qafiya synchronously - returns list of 2 verses"""
        # Format prompt for fixing verse
        formatted_prompt = self.prompt_manager.format_prompt(
            'qafiya_refinement',
            qafiya=constraints.qafiya or "غير محدد",
            qafiya_type=constraints.qafiya_type or "غير محدد",
            qafiya_type_description_and_examples=constraints.qafiya_type_description_and_examples or "غير محدد",
            qafiya_harakah=constraints.qafiya_harakah or "",
            meter=constraints.meter or "غير محدد",
            meeter_tafeelat=constraints.meeter_tafeelat or "غير محدد",
            entire_poem=entire_poem,
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            bad_bait=verse,
            context=f"إصلاح القافية. المشكلة: {error_details}"
        )
        
        # Generate fixed verse
        response = self.llm.generate(formatted_prompt)
        fixed_verses = self._parse_verses_from_response(response)
        
        return fixed_verses
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints', 'evaluation']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations']
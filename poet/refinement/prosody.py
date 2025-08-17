# poet/refinement/prosody.py

import logging
from typing import Optional, Dict, Any, List
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.prompts import get_global_prompt_manager
from poet.llm.base_llm import BaseLLM
from poet.core.node import Node
from poet.models.quality import QualityAssessment


class ProsodyRefiner(Node):
    """Fixes meter violations in verses"""
    
    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_manager = get_global_prompt_manager()
        self.logger.setLevel(logging.INFO)
    
    @property
    def name(self) -> str:
        return "prosody_refiner"
    
    @name.setter
    def name(self, value: str):
        """Set the refiner name (ignored, always returns custom name)"""
        pass
    
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
                fixed_bait = await self._fix_single_bait(
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
    
    async def _fix_single_bait(self, verse: str, constraints: Constraints, error_details: str) -> str:
        """Fix a single verse's meter"""
        # Format prompt for fixing verse
        formatted_prompt = self.prompt_manager.format_prompt(
            'prosody_refinement',
            meter=constraints.meter or "غير محدد",
            meeter_tafeelat=constraints.meeter_tafeelat or "غير محدد",
            qafiya=constraints.qafiya or "غير محدد",
            qafiya_type=constraints.qafiya_type or "غير محدد",
            qafiya_type_description_and_examples=constraints.qafiya_type_description_and_examples or "غير محدد",
            qafiya_harakah=constraints.qafiya_harakah or "",
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
        Execute the prosody refiner node.
        
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
            self.logger.info(f"{self.name}: No prosody refinement needed")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
        
        # Apply refinement
        try:
            refined_poem = self._apply_sync_refinement(poem, constraints, evaluation)
            
            self.logger.info(f"{self.name}: Prosody refinement applied successfully")
            return {
                'poem': refined_poem,
                'refined': True,
                'refinement_iterations': 1
            }
        except Exception as e:
            self.logger.error(f"{self.name}: Prosody refinement failed: {e}")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
    
    def _apply_sync_refinement(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """
        Apply prosody refinement synchronously.
        """
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
            fixed_bait = self._fix_single_bait_sync(
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
    
    def _fix_single_bait_sync(self, verse: str, constraints: Constraints, error_details: str) -> List[str]:
        """Fix a single verse's meter synchronously"""
        # Format prompt for fixing verse
        formatted_prompt = self.prompt_manager.format_prompt(
            'prosody_refinement',  # Use the correct template name from YAML
            meter=constraints.meter or "غير محدد",
            meeter_tafeelat=constraints.meeter_tafeelat or "غير محدد",
            qafiya=constraints.qafiya or "غير محدد",
            qafiya_type=constraints.qafiya_type or "غير محدد",
            qafiya_type_description_and_examples=constraints.qafiya_type_description_and_examples or "غير محدد",
            qafiya_harakah=constraints.qafiya_harakah or "",
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            existing_verses=verse,
            context=f"إصلاح الوزن العروضي للبيت. المشكلة: {error_details}"
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
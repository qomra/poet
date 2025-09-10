# poet/refinement/prosody.py

import logging
from typing import Dict, Any, Optional
from poet.core.node import Node
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.prompts import get_global_prompt_manager


class ProsodyRefiner(Node):
    """
    Refines poem prosody (meter) to improve rhythmic consistency.
    
    Supports iteration context for refinement pipelines.
    """
    
    def __init__(self, llm, iteration: int = None, **kwargs):
        super().__init__(**kwargs)
        
        self.llm = llm
        self.prompt_manager = get_global_prompt_manager()
        self.iteration = iteration
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine poem prosody.
        
        Args:
            input_data: Input data containing poem, constraints, and evaluation
            context: Pipeline context
            
        Returns:
            Output data with refined poem
        """
        # Validate inputs
        poem = input_data.get('poem')
        constraints = input_data.get('constraints')
        evaluation = input_data.get('evaluation')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Check if prosody refinement is needed
        if not self.should_refine(evaluation):
            self.logger.info("ℹ️ Prosody refinement not needed")
            return {
                'poem': poem,
                'refined': False,
                'refiner_used': 'prosody_refiner'
            }
        
        # Perform prosody refinement
        refined_poem = self._refine_prosody(poem, constraints, evaluation)
        
        # Store harmony data
        output_data = {
            'poem': refined_poem,
            'refined': True,
            'refiner_used': 'prosody_refiner',
            'iteration': self.iteration
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        """Check if prosody refinement is needed."""
        if not evaluation or not evaluation.prosody_validation:
            return False
        
        return not evaluation.prosody_validation.overall_valid
    
    def _refine_prosody(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """Refine the poem's prosody."""
        if not evaluation.prosody_validation:
            return poem

        # Get broken verses from evaluation
        broken_bait = self._identify_broken_bait(evaluation.prosody_validation)

        if not broken_bait:
            return poem

        self.logger.info(f"🎵 Fixing prosody for {len(broken_bait)} broken verses")

        # Fix each broken verse
        fixed_verses = poem.verses.copy()
        for bait_index, error_details in broken_bait:
            bait = "#".join(poem.verses[bait_index*2:bait_index*2+2])
            fixed_bait = self._fix_single_bait(
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
        refined_poem = LLMPoem(
            verses=fixed_verses,
            llm_provider=poem.llm_provider,
            model_name=poem.model_name,
            constraints=poem.constraints,
            generation_timestamp=poem.generation_timestamp
        )
        
        self.logger.info(f"✅ Prosody refinement completed")
        self.logger.info(f"🔍 Original verses: {poem.verses}")
        self.logger.info(f"🔍 Fixed verses: {refined_poem.verses}")
        return refined_poem
    
    def _identify_broken_bait(self, prosody_validation) -> list:
        """Identify verses with meter violations"""
        broken_bait = []

        if not hasattr(prosody_validation, 'bait_results') or prosody_validation.bait_results is None:
            return broken_bait

        for i, bait_result in enumerate(prosody_validation.bait_results):
            if not bait_result.is_valid:
                broken_bait.append((i, bait_result.error_details))

        return broken_bait

    def _fix_single_bait(self, verse: str, constraints: Constraints, error_details: str) -> list:
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
        self.logger.info(f"🔍 LLM response for prosody fix: {response[:500]}...")
        fixed_verses = self._parse_verses_from_response(response)
        self.logger.info(f"🔍 Parsed fixed verses: {fixed_verses}")

        return fixed_verses

    def _parse_verses_from_response(self, response: str) -> list:
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
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            self.logger.info(f"Fallback parsing result: {lines}")
            return lines
        except Exception as e:
            self.logger.error(f"Failed to parse verses from response: {e}")
            # Fallback: split by newlines
            return [line.strip() for line in response.split('\n') if line.strip()]
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        poem = self.harmony_data['input'].get('poem')
        if poem:
            return f"Refined prosody for poem with {len(poem.verses)} verses"
        return "Refined prosody"
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        refined = self.harmony_data['output'].get('refined', False)
        return f"Prosody refinement: {'Applied' if refined else 'Not needed'}"
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints', 'evaluation']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refiner_used', 'iteration']
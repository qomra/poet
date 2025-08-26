# poet/analysis/constraint_parser.py

import logging
from typing import Dict, Any, Optional
from poet.core.node import Node
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.prompts import get_global_prompt_manager


class ConstraintParser(Node):
    """
    Parses user prompts to extract poetry constraints.
    
    Supports harmony generation for constraint analysis.
    """
    
    def __init__(self, llm, **kwargs):
        super().__init__(**kwargs)
        
        self.llm = llm
        self.prompt_manager = get_global_prompt_manager()
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse user prompt to extract constraints.
        
        Args:
            input_data: Input data containing user prompt
            context: Pipeline context
            
        Returns:
            Output data with parsed constraints
        """
        # Validate inputs
        user_prompt = input_data.get('user_prompt')
        if not user_prompt:
            raise ValueError("user_prompt not found in input_data")
        
        # Parse constraints from user prompt
        constraints = self._parse_constraints(user_prompt)
        
        # Store harmony data
        output_data = {
            'constraints': constraints,
            'parsed': True
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def _parse_constraints(self, user_prompt: str) -> Constraints:
        """Parse constraints from user prompt."""
        self.logger.info("🔍 Parsing constraints from user prompt")
        
        # Create parsing prompt
        parsing_prompt = self.prompt_manager.format_prompt(
            'constraint_parsing',
            user_prompt=user_prompt
        )
        
        # Get LLM response
        response = self.llm.generate(parsing_prompt)
        
        # Parse the structured response
        constraints_data = self._parse_llm_response(response)
        
        # Debug: Log what was parsed
        self.logger.info(f"🔍 Parsed constraints data: {constraints_data}")
        
        # Create Constraints object
        constraints = Constraints(**constraints_data)
        
        self.logger.info("✅ Constraints parsed successfully")
        return constraints
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into constraint data."""
        import json
        import re
        
        try:
            # Try to find JSON in the response
            json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
            json_blocks = re.findall(json_block_pattern, response, re.DOTALL)
            
            if json_blocks:
                json_str = json_blocks[0]
            else:
                # Fallback: find first { and last }
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = response[json_start:json_end]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = ['theme', 'meter', 'qafiya']
            for field in required_fields:
                if field not in data:
                    data[field] = None
            
            return data
            
        except Exception as e:
            self.logger.error(f"📄 Failed to parse LLM response: {e}")
            # Return default constraints
            return {
                'theme': 'general',
                'meter': None,
                'qafiya': None,
                'line_count': 4
            }
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """Generate natural reasoning for this parser node."""
        user_prompt = input_data.get('user_prompt', '')
        constraints = output_data.get('constraints')
        
        reasoning = f"I analyzed the user prompt: '{user_prompt[:100]}...'"
        
        if constraints:
            reasoning += f" I extracted the following constraints: theme='{constraints.theme}', meter='{constraints.meter}', qafiya='{constraints.qafiya}'"
        
        reasoning += ". These constraints will guide the poem generation process."
        
        return reasoning
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        user_prompt = self.harmony_data['input'].get('user_prompt', '')
        return f"Parsed constraints from prompt: {user_prompt[:50]}..."
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        constraints = self.harmony_data['output'].get('constraints')
        if constraints:
            return f"Extracted constraints: theme={constraints.theme}, meter={constraints.meter}, qafiya={constraints.qafiya}"
        return "Constraints extracted"
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['user_prompt']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['constraints', 'parsed']

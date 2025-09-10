# poet/analysis/constraint_parser.py

import json
import logging
from typing import Dict, Any, Optional
from poet.core.node import Node
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.prompts import get_global_prompt_manager
from poet.llm.base_llm import BaseLLM


class ConstraintParsingError(Exception):
    """Raised when constraint parsing fails"""
    pass


class ConstraintParser(Node):
    """
    Extracts and parses poetry constraints from natural language user input.
    
    Uses LLM-powered analysis to understand user requirements and convert them
    into structured Constraints objects. Handles ambiguities and provides
    clarification requests when needed. Supports harmony generation for constraint analysis.
    """
    
    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(**kwargs)
        
        self.llm = llm
        # Use global prompt manager instead of creating new instance
        self.prompt_manager = get_global_prompt_manager()
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the constraint parsing node.
        
        Args:
            input_data: Input data containing user_prompt or prompt
            context: Pipeline context
            
        Returns:
            Output data with parsed constraints
        """
        # Validate inputs - handle both 'user_prompt' and 'prompt' keys
        user_prompt = input_data.get('user_prompt') or input_data.get('prompt')
        poem_id = input_data.get('poem_id')
        if not user_prompt:
            raise ValueError("user_prompt or prompt not found in input_data")
        
        # Parse constraints from user prompt
        constraints = self.parse_constraints(user_prompt, poem_id)
        
        # Store harmony data
        output_data = {
            'constraints': constraints,
            'parsed': True
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def parse_constraints(self, user_prompt: str, poem_id: int) -> Constraints:
        """
        Parse constraints from user input using LLM analysis.
        
        Args:
            user_prompt: Natural language description of poetry requirements
            
        Returns:
            Constraints object with extracted constraints
            
        Raises:
            ConstraintParsingError: If parsing fails or response is invalid
        """
        try:
            # Format the unified extraction prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'unified_extraction',
                user_prompt=user_prompt
            )
            
            # Get LLM response
            response = self.llm.generate(formatted_prompt)
            # Parse the structured response
            constraints_data = self._parse_llm_response(response)
            
            # Create Constraints object
            constraints = self._create_constraints(constraints_data, user_prompt, poem_id)
            
            # Handle ambiguities and clarifications
            if constraints.has_ambiguities():
                self._handle_ambiguities(constraints, constraints_data)
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"Failed to parse constraints: {e}")
            raise ConstraintParsingError(f"Constraint parsing failed: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract JSON constraints.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed constraints data
            
        Raises:
            ConstraintParsingError: If JSON parsing fails
        """
        try:
            # Look for JSON code blocks first (most reliable)
            import re
            
            # Pattern to match JSON code blocks
            json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
            json_blocks = re.findall(json_block_pattern, response, re.DOTALL)
            
            if json_blocks:
                # Use the first JSON block found
                json_str = json_blocks[0].strip()
                self.logger.debug(f"Found JSON in code block: {json_str[:100]}...")
            else:
                # Fallback: extract JSON from response (handle markdown code blocks)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start == -1 or json_end == 0:
                    raise ValueError("No JSON found in response")
                
                json_str = response[json_start:json_end]
                self.logger.debug(f"Extracted JSON from response: {json_str[:100]}...")
            
            # Clean up the JSON string
            json_str = json_str.strip()
            
            # Remove any trailing text after the JSON
            # Find the last complete JSON object
            brace_count = 0
            last_complete_end = -1
            
            for i, char in enumerate(json_str):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_complete_end = i + 1
            
            if last_complete_end > 0:
                json_str = json_str[:last_complete_end]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate required structure
            self._validate_response_structure(data)
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            self.logger.error(f"Raw JSON string: {json_str}")
            raise ConstraintParsingError(f"Invalid JSON response: {e}")
        except ValueError as e:
            self.logger.error(f"Invalid response format: {e}")
            raise ConstraintParsingError(f"Invalid response format: {e}")
    
    def _validate_response_structure(self, data: Dict[str, Any]):
        """
        Validate that the response has the expected structure.
        
        Args:
            data: Parsed JSON data
            
        Raises:
            ValueError: If structure is invalid
        """
        required_fields = [
            'meter', 'qafiya', 'line_count', 'theme', 'tone',
            'imagery', 'keywords', 'register', 'era', 'poet_style', 'sections',
            'ambiguities', 'suggestions', 'reasoning'
        ]
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate list fields
        list_fields = ['imagery', 'keywords', 'sections', 'ambiguities']
        for field in list_fields:
            if data[field] is not None and not isinstance(data[field], list):
                raise ValueError(f"Field '{field}' must be a list or null")
    
    def _create_constraints(self, data: Dict[str, Any], original_prompt: str, poem_id: int) -> Constraints:
        """
        Create Constraints object from parsed data.
        
        Args:
            data: Parsed constraints data
            original_prompt: Original user prompt for reference
            poem_id: Poem ID
            
        Returns:
            Constraints object
        """
        # Convert null values to None and handle type conversions
        constraints = Constraints(
            poem_id=poem_id,
            meter=data.get('meter') if data.get('meter') != 'null' else None,
            qafiya=data.get('qafiya') if data.get('qafiya') != 'null' else None,
            line_count=data.get('line_count') if data.get('line_count') != 'null' else None,
            theme=data.get('theme') if data.get('theme') != 'null' else None,
            tone=data.get('tone') if data.get('tone') != 'null' else None,
            imagery=data.get('imagery', []) or [],
            keywords=data.get('keywords', []) or [],
            register=data.get('register') if data.get('register') != 'null' else None,
            era=data.get('era') if data.get('era') != 'null' else None,
            poet_style=data.get('poet_style') if data.get('poet_style') != 'null' else None,
            sections=data.get('sections', []) or [],
            ambiguities=data.get('ambiguities', []) or []
        )
        
        # Store additional metadata (handle "null" strings)
        constraints.llm_suggestions = data.get('suggestions') if data.get('suggestions') != 'null' else None
        constraints.llm_reasoning = data.get('reasoning') if data.get('reasoning') != 'null' else None
        constraints.original_prompt = original_prompt
        
        return constraints
    
    def _handle_ambiguities(self, constraints: Constraints, data: Dict[str, Any]):
        """
        Handle ambiguities and provide clarification guidance.
        
        Args:
            constraints: Constraints object with ambiguities
            data: Original parsed data with suggestions
        """
        if data.get('suggestions'):
            # Add suggestions to ambiguities for user clarification
            suggestion_text = f"اقتراح: {data['suggestions']}"
            if suggestion_text not in constraints.ambiguities:
                constraints.ambiguities.append(suggestion_text)
        
        # Log ambiguities for debugging
        if constraints.ambiguities:
            self.logger.info(f"Constraints parsed with ambiguities: {constraints.ambiguities}")
    
    def get_clarification_prompt(self, constraints: Constraints) -> Optional[str]:
        """
        Generate a clarification prompt for ambiguous constraints.
        
        Args:
            constraints: Constraints with ambiguities
            
        Returns:
            Clarification prompt in Arabic, or None if no clarification needed
        """
        if not constraints.has_ambiguities():
            return None
        
        clarifications = []
        
        for ambiguity in constraints.ambiguities:
            if "اقتراح:" in ambiguity:
                clarifications.append(ambiguity)
            elif "غير واضح" in ambiguity or "غموض" in ambiguity:
                clarifications.append(f"يرجى توضيح: {ambiguity}")
        
        if not clarifications:
            return None
        
        prompt = "لتحسين جودة الشعر، يرجى توضيح النقاط التالية:\n\n"
        prompt += "\n".join(f"• {clarification}" for clarification in clarifications)
        prompt += "\n\nيمكنك الإجابة على ما تشاء من هذه النقاط."
        
        return prompt
    
    def refine_constraints(self, constraints: Constraints, user_clarification: str) -> Constraints:
        """
        Refine constraints based on user clarification.
        
        Args:
            constraints: Original constraints with ambiguities
            user_clarification: User's clarification response
            
        Returns:
            Refined Constraints object
        """
        # Create a combined prompt with original and clarification
        combined_prompt = f"""
        الطلب الأصلي: {constraints.original_prompt}
        
        التوضيحات الإضافية: {user_clarification}
        """
        
        # Re-parse with the combined prompt
        return self.parse_constraints(combined_prompt.strip())
    
    def validate_constraints(self, constraints: Constraints) -> bool:
        """
        Validate extracted constraints for completeness and compatibility.
        
        Args:
            constraints: Constraints to validate
            
        Returns:
            True if constraints are valid
            
        Raises:
            ConstraintParsingError: If validation fails
        """
        # Basic validation is now done in Constraints.__post_init__
        # Just return True since constraints are validated on creation
        return True

    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        # Handle both 'user_prompt' and 'prompt' keys
        user_prompt = self.harmony_data['input'].get('user_prompt') or self.harmony_data['input'].get('prompt', '')
        
        if isinstance(user_prompt, str):
            return f"Parsed constraints from prompt: {user_prompt[:50]}..."
        else:
            return f"Parsed constraints from prompt: {str(user_prompt)[:50]}..."
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        constraints = self.harmony_data['output'].get('constraints')
        if constraints:
            summary = f"Extracted constraints: theme={constraints.theme}, meter={constraints.meter}, qafiya={constraints.qafiya}"
            if constraints.has_ambiguities():
                summary += f" ({len(constraints.ambiguities)} ambiguities)"
            return summary
        return "Constraints extracted"
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['user_prompt']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['constraints', 'parsed']

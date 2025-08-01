# poet/analysis/constraint_parser.py

import json
import logging
from typing import Optional, Dict, Any
from poet.models.constraints import UserConstraints
from poet.prompts.prompt_manager import PromptManager
from poet.llm.base_llm import BaseLLM


class ConstraintParsingError(Exception):
    """Raised when constraint parsing fails"""
    pass


class ConstraintParser:
    """
    Extracts and parses poetry constraints from natural language user input.
    
    Uses LLM-powered analysis to understand user requirements and convert them
    into structured UserConstraints objects. Handles ambiguities and provides
    clarification requests when needed.
    """
    
    def __init__(self, llm_provider: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm_provider
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(__name__)
    
    def parse_constraints(self, user_prompt: str) -> UserConstraints:
        """
        Parse constraints from user input using LLM analysis.
        
        Args:
            user_prompt: Natural language description of poetry requirements
            
        Returns:
            UserConstraints object with extracted constraints
            
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
            
            # Create UserConstraints object
            constraints = self._create_constraints(constraints_data, user_prompt)
            
            # Handle ambiguities and clarifications
            if constraints.has_ambiguities():
                self._handle_ambiguities(constraints, constraints_data)
            
            return constraints
            
        except Exception as e:
            self.logger.error(f"Failed to parse constraints: {e}")
            raise ConstraintParsingError(f"Constraint parsing failed: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the structured JSON response from the LLM.
        
        Args:
            response: Raw LLM response containing JSON
            
        Returns:
            Parsed constraints dictionary
            
        Raises:
            ConstraintParsingError: If JSON parsing fails
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate required structure
            self._validate_response_structure(data)
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
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
            'meter', 'rhyme', 'line_count', 'theme', 'tone',
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
    
    def _create_constraints(self, data: Dict[str, Any], original_prompt: str) -> UserConstraints:
        """
        Create UserConstraints object from parsed data.
        
        Args:
            data: Parsed constraints data
            original_prompt: Original user prompt for reference
            
        Returns:
            UserConstraints object
        """
        # Convert null values to None and handle type conversions
        constraints = UserConstraints(
            meter=data.get('meter') if data.get('meter') != 'null' else None,
            rhyme=data.get('rhyme') if data.get('rhyme') != 'null' else None,
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
    
    def _handle_ambiguities(self, constraints: UserConstraints, data: Dict[str, Any]):
        """
        Handle ambiguities and provide clarification guidance.
        
        Args:
            constraints: UserConstraints object with ambiguities
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
    
    def get_clarification_prompt(self, constraints: UserConstraints) -> Optional[str]:
        """
        Generate a clarification prompt for ambiguous constraints.
        
        Args:
            constraints: UserConstraints with ambiguities
            
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
    
    def refine_constraints(self, constraints: UserConstraints, user_clarification: str) -> UserConstraints:
        """
        Refine constraints based on user clarification.
        
        Args:
            constraints: Original constraints with ambiguities
            user_clarification: User's clarification response
            
        Returns:
            Refined UserConstraints object
        """
        # Create a combined prompt with original and clarification
        combined_prompt = f"""
        الطلب الأصلي: {constraints.original_prompt}
        
        التوضيحات الإضافية: {user_clarification}
        """
        
        # Re-parse with the combined prompt
        return self.parse_constraints(combined_prompt.strip())
    
    def validate_constraints(self, constraints: UserConstraints) -> bool:
        """
        Validate extracted constraints for completeness and compatibility.
        
        Args:
            constraints: UserConstraints to validate
            
        Returns:
            True if constraints are valid
            
        Raises:
            ConstraintParsingError: If validation fails
        """
        # Basic validation is now done in UserConstraints.__post_init__
        # Just return True since constraints are validated on creation
        return True

# poet/planning/qafiya_selector.py

import json
import logging
from typing import Optional, Dict, Any
from poet.models.constraints import Constraints, QafiyaType, QafiyaTypeDescriptionAndExamples
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager
from poet.core.node import Node


class QafiyaSelectionError(Exception):
    """Raised when qafiya selection fails"""
    pass


class QafiyaSelector(Node):
    """
    Selects and enriches qafiya specifications for poem generation.
    
    Analyzes user constraints and original prompt to determine the most appropriate
    qafiya letter, harakah, type, and pattern for the requested poem.
    """
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
    
    def select_qafiya(self, constraints: Constraints, original_prompt: str) -> Constraints:
        """
        Select and enrich qafiya specification for the given constraints.
        
        Args:
            constraints: Parsed user constraints
            original_prompt: Original user prompt text
            
        Returns:
            Enhanced Constraints with complete qafiya specification
            
        Raises:
            QafiyaSelectionError: If qafiya selection fails
        """
        try:
            # Check if qafiya is already fully specified
            if self._is_qafiya_complete(constraints):
                self.logger.info("Qafiya already fully specified, validating...")
                return self._validate_existing_qafiya(constraints)
            
            # Get missing components
            missing_components = self._get_missing_qafiya_components(constraints)
            self.logger.info(f"Missing qafiya components: {missing_components}")
            
            # Fill missing components using LLM
            qafiya_spec = self._fill_missing_qafiya_components(constraints, original_prompt, missing_components)
            # Update constraints with filled qafiya specification
            enhanced_constraints = self._enhance_constraints(constraints, qafiya_spec)
            
            return enhanced_constraints
            
        except Exception as e:
            self.logger.error(f"Failed to select qafiya: {e}")
            raise QafiyaSelectionError(f"Qafiya selection failed: {e}")
    
    def _is_qafiya_complete(self, constraints: Constraints) -> bool:
        """Check if qafiya specification is complete"""
        return (
            constraints.qafiya is not None and
            constraints.qafiya_harakah is not None and
            constraints.qafiya_type is not None 
        )
    
    def _get_missing_qafiya_components(self, constraints: Constraints) -> list:
        """Get list of missing qafiya components"""
        missing = []
        if not constraints.qafiya:
            missing.append('qafiya_letter')
        if not constraints.qafiya_harakah:
            missing.append('qafiya_harakah')
        if not constraints.qafiya_type:
            missing.append('qafiya_type')
        return missing
    
    def _validate_existing_qafiya(self, constraints: Constraints) -> Constraints:
        """Validate existing qafiya specification"""
        # For now, just return as-is. Could add validation logic here later
        return constraints
    
    def _fill_missing_qafiya_components(self, constraints: Constraints, original_prompt: str, missing_components: list) -> Dict[str, Any]:
        """Fill missing qafiya components using LLM"""
        try:
            # Format the qafiya completion prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'qafiya_completion',
                original_prompt=original_prompt,
                meter=constraints.meter or "غير محدد",
                theme=constraints.theme or "غير محدد",
                tone=constraints.tone or "غير محدد",
                era=constraints.era or "غير محدد",
                existing_qafiya=constraints.qafiya or "غير محدد",   
                existing_harakah=constraints.qafiya_harakah or "غير محدد",
                existing_type=constraints.qafiya_type or "غير محدد",
                existing_type_description_and_examples=constraints.qafiya_type_description_and_examples or "غير محدد",
                missing_components=", ".join(missing_components)
            )
            # Get LLM response
            response = self.llm.generate(formatted_prompt)
            # Parse the structured response
            qafiya_spec = self._parse_llm_response(response)
            
            # Preserve existing components
            if constraints.qafiya:
                qafiya_spec['qafiya_letter'] = constraints.qafiya
            if constraints.qafiya_harakah:
                qafiya_spec['qafiya_harakah'] = constraints.qafiya_harakah
            if constraints.qafiya_type:
                qafiya_spec['qafiya_type'] = constraints.qafiya_type
                qafiya_spec['qafiya_type_description_and_examples'] = constraints.qafiya_type_description_and_examples

            return qafiya_spec
            
        except Exception as e:
            self.logger.error(f"Failed to fill missing qafiya components: {e}")
            raise QafiyaSelectionError(f"Failed to fill missing qafiya components: {e}")
    
    def _select_qafiya_with_llm(self, constraints: Constraints, original_prompt: str) -> Dict[str, Any]:
        """Use LLM to select appropriate qafiya"""
        
        # Format the qafiya selection prompt
        formatted_prompt = self.prompt_manager.format_prompt(
            'qafiya_selection',
            original_prompt=original_prompt,
            meter=constraints.meter or "غير محدد",
            theme=constraints.theme or "غير محدد",
            tone=constraints.tone or "غير محدد",
            era=constraints.era or "غير محدد",
            existing_qafiya=constraints.qafiya or "غير محدد"
        )
        
        # Get LLM response
        response = self.llm.generate(formatted_prompt)
        
        # Parse the structured response
        qafiya_spec = self._parse_llm_response(response)
        
        return qafiya_spec
    

    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse the structured JSON response from the LLM"""
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
            #self._validate_response_structure(data)
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            raise QafiyaSelectionError(f"Invalid JSON response: {e}")
        except ValueError as e:
            self.logger.error(f"Invalid response format: {e}")
            raise QafiyaSelectionError(f"Invalid response format: {e}")
    
    def _validate_response_structure(self, data: Dict[str, Any]):
        """Validate that the response has the expected structure"""
        required_fields = ['qafiya_letter', 'qafiya_harakah', 'qafiya_type']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Validate qafiya_type is one of the enum values
        valid_types = [qtype.value for qtype in QafiyaType]
        if data['qafiya_type'] not in valid_types:
            raise ValueError(f"Invalid qafiya_type: {data['qafiya_type']}. Must be one of: {valid_types}")
    
    def _enhance_constraints(self, constraints: Constraints, qafiya_spec: Dict[str, Any]) -> Constraints:
        """Enhance constraints with selected qafiya specification"""
        # Create new constraints with enhanced qafiya
        enhanced_constraints = Constraints(
            meter=constraints.meter,
            qafiya=qafiya_spec['qafiya_letter'],
            qafiya_harakah=qafiya_spec['qafiya_harakah'],
            qafiya_type=QafiyaType(qafiya_spec['qafiya_type']),
            line_count=constraints.line_count,
            theme=constraints.theme,
            tone=constraints.tone,
            imagery=constraints.imagery,
            keywords=constraints.keywords,
            sections=constraints.sections,
            register=constraints.register,
            era=constraints.era,
            poet_style=constraints.poet_style,
            ambiguities=constraints.ambiguities,
            original_prompt=constraints.original_prompt
        )
        
        return enhanced_constraints
    
    def _get_harakah_symbol(self, harakah: str) -> str:
        """Convert harakah name to symbol"""
        harakah_map = {
            "مفتوح": "َ",
            "مكسور": "ِ", 
            "مضموم": "ُ",
            "ساكن": "ْ"
        }
        return harakah_map.get(harakah, "") 
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the qafiya selection node.
        
        Args:
            input_data: Input data containing constraints and user_prompt
            context: Pipeline context with LLM and prompt_manager
            
        Returns:
            Output data with enhanced constraints
        """
        # Set up context
        self.llm = context.get('llm')
        self.prompt_manager = context.get('prompt_manager') or PromptManager()
        
        if not self.llm:
            raise ValueError("LLM not provided in context")
        
        # Extract required data
        constraints = input_data.get('constraints')
        user_prompt = input_data.get('user_prompt')
        
        if not constraints:
            raise ValueError("constraints not found in input_data")
        if not user_prompt:
            raise ValueError("user_prompt not found in input_data")
        
        # Select qafiya
        enhanced_constraints = self.select_qafiya(constraints, user_prompt)
        
        return {
            'constraints': enhanced_constraints,
            'qafiya_selected': True
        }
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['constraints', 'user_prompt']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['constraints', 'qafiya_selected'] 
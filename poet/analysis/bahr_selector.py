# poet/analysis/bahr_selector.py

import json
import logging
from typing import Optional, Dict, Any, List
from poet.models.constraints import Constraints
from poet.llm.base_llm import BaseLLM
from poet.prompts import get_global_prompt_manager
from poet.data.bohour_meters import BohourMetersManager
from poet.core.node import Node



class BahrSelectionError(Exception):
    """Raised when bahr selection fails"""
    pass


class BahrSelector(Node):
    """
    Selects and standardizes bahr (meter) specifications for poem generation.
    
    Analyzes user constraints and original prompt to determine the most appropriate
    bahr name, format it correctly, and suggest sub-bahrs if applicable.
    """
    
    def __init__(self, llm: BaseLLM, **kwargs):
        super().__init__(**kwargs)
        self.llm = llm
        self.prompt_manager = get_global_prompt_manager()
        self.meters_manager = BohourMetersManager()
    
    def select_bahr(self, constraints: Constraints, original_prompt: str) -> Constraints:
        """
        Select and enhance bahr specification for the given constraints.
        
        Args:
            constraints: Current constraints object
            original_prompt: Original user prompt for context
            
        Returns:
            Enhanced Constraints with standardized bahr specification
            
        Raises:
            BahrSelectionError: If bahr selection fails
        """
        try:
            # Check if bahr is already properly specified
            if self._is_bahr_complete(constraints) and self._validate_existing_bahr(constraints):
                self.logger.info("Bahr already properly specified, validating...")
                # add tafeelat to constraints
                constraints.meeter_tafeelat = " ".join(self.meters_manager.get_meter_tafeelat(constraints.meter))
                return constraints
            
            # Get missing components
            missing_components = self._get_missing_bahr_components(constraints)
            self.logger.info(f"Missing bahr components: {missing_components}")
            
            # Fill missing components using LLM
            bahr_spec = self._fill_missing_bahr_components(constraints, original_prompt, missing_components)
            
            # Update constraints with filled bahr specification
            enhanced_constraints = self._enhance_constraints(constraints, bahr_spec)
            
            return enhanced_constraints
            
        except Exception as e:
            self.logger.error(f"Failed to select bahr: {e}")
            raise BahrSelectionError(f"Bahr selection failed: {e}")
    
    def _is_bahr_complete(self, constraints: Constraints) -> bool:
        """Check if bahr specification is complete and valid"""
        if not constraints.meter:
            return False
        
        # Check if the meter name is recognized
        return self.meters_manager.validate_meter(constraints.meter)
    
    def _get_missing_bahr_components(self, constraints: Constraints) -> list:
        """Get list of missing bahr components"""
        missing = []
        
        if not constraints.meter:
            missing.append('meter_name')
        elif not self.meters_manager.validate_meter(constraints.meter):
            self.logger.info(f"Meter {constraints.meter} is not recognized, adding meter_standardization")
            missing.append('meter_standardization')
        
        return missing
    
    def _validate_existing_bahr(self, constraints: Constraints) -> Constraints:
        """Validate existing bahr specification"""
        # For now, just return as-is. Could add validation logic here later
        return constraints
    
    def _fill_missing_bahr_components(self, constraints: Constraints, original_prompt: str, missing_components: list) -> Dict[str, Any]:
        """Fill missing bahr components using LLM"""
        try:
            # Get available meters information for the prompt
            available_meters = self._get_relevant_meters_info(constraints)
            
            # Format the bahr selection prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'bahr_selection',
                original_prompt=original_prompt,
                current_meter=constraints.meter or "غير محدد",
                theme=constraints.theme or "غير محدد",
                tone=constraints.tone or "غير محدد",
                line_count=constraints.line_count or "غير محدد",
                available_meters=available_meters,
                missing_components=", ".join(missing_components)
            )
            
            # Get LLM response
            response = self.llm.generate(formatted_prompt)
            
            # Parse the response
            bahr_spec = self._parse_llm_response(response)
            
            # Validate the response structure
            self._validate_response_structure(bahr_spec)
            
            return bahr_spec
            
        except Exception as e:
            self.logger.error(f"Failed to fill missing bahr components: {e}")
            raise BahrSelectionError(f"Failed to fill missing bahr components: {e}")
    
    def _get_relevant_meters_info(self, constraints: Constraints) -> str:
        """Get relevant meters information for the prompt"""
        meters_info = []
        
        # Get meters by theme if specified
        if constraints.theme:
            theme_meters = self.meters_manager.get_meters_by_theme(constraints.theme)
            if theme_meters:
                meters_info.append(f"مقترحات للبحر حسب الموضوع '{constraints.theme}':")
                for meter in theme_meters[:5]:  # Limit to 5 suggestions
                    meters_info.append(f"  - {meter.arabic_name} (صعوبة: {meter.difficulty_level})")
        
        # Get easy meters if no theme or for beginners
        if not constraints.theme or constraints.tone in ["بسيط", "سهل", "مبتدئ"]:
            easy_meters = self.meters_manager.get_meters_by_difficulty("easy")
            if easy_meters:
                meters_info.append("بحور سهلة للمبتدئين:")
                for meter in easy_meters[:3]:
                    meters_info.append(f"  - {meter.arabic_name}")
        
        # Get medium difficulty meters
        medium_meters = self.meters_manager.get_meters_by_difficulty("medium")
        if medium_meters:
            meters_info.append("بحور متوسطة الصعوبة:")
            for meter in medium_meters[:3]:
                meters_info.append(f"  - {meter.arabic_name}")
        
        return "\n".join(meters_info) if meters_info else "جميع البحور متاحة"
    
    def _select_bahr_with_llm(self, constraints: Constraints, original_prompt: str) -> Dict[str, Any]:
        """Select bahr using LLM"""
        try:
            # Get available meters information
            available_meters = self._get_relevant_meters_info(constraints)
            
            # Format the prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                'bahr_selection',
                original_prompt=original_prompt,
                current_meter=constraints.meter or "غير محدد",
                theme=constraints.theme or "غير محدد",
                tone=constraints.tone or "غير محدد",
                line_count=constraints.line_count or "غير محدد",
                available_meters=available_meters,
                missing_components="meter_name"
            )
            
            # Get LLM response
            response = self.llm.generate(formatted_prompt)
            
            # Parse the response
            bahr_spec = self._parse_llm_response(response)
            
            # Validate the response structure
            self._validate_response_structure(bahr_spec)
            
            return bahr_spec
            
        except Exception as e:
            self.logger.error(f"Failed to select bahr with LLM: {e}")
            raise BahrSelectionError(f"Failed to select bahr with LLM: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract bahr specification"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise BahrSelectionError("No JSON found in LLM response")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from LLM response: {e}")
            raise BahrSelectionError(f"Failed to parse JSON from LLM response: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            raise BahrSelectionError(f"Failed to parse LLM response: {e}")
    
    def _validate_response_structure(self, data: Dict[str, Any]):
        """Validate the structure of the LLM response"""
        required_fields = ['meter_name']
        
        for field in required_fields:
            if field not in data:
                raise BahrSelectionError(f"Missing required field: {field}")
            
            if data[field] is None:
                raise BahrSelectionError(f"Field {field} cannot be null")
            
            if not isinstance(data[field], str):
                raise BahrSelectionError(f"Field {field} must be a string")
        
        # Validate meter name
        meter_name = data['meter_name']
        if not self.meters_manager.validate_meter(meter_name):
            # Try to find a similar meter
            similar_meters = self.meters_manager.search_meters(meter_name)
            if similar_meters:
                suggestions = [meter.arabic_name for meter in similar_meters[:3]]
                raise BahrSelectionError(f"Invalid meter name: {meter_name}. Similar meters: {', '.join(suggestions)}")
            else:
                raise BahrSelectionError(f"Invalid meter name: {meter_name}")
    
    def _enhance_constraints(self, constraints: Constraints, bahr_spec: Dict[str, Any]) -> Constraints:
        """Enhance constraints with bahr specification"""
        # Create new constraints with updated bahr
        enhanced_constraints = Constraints(
            meter=bahr_spec['meter_name'],
            qafiya=constraints.qafiya,
            qafiya_harakah=constraints.qafiya_harakah,
            qafiya_type=constraints.qafiya_type,
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
    
    def suggest_sub_bahrs(self, constraints: Constraints) -> List[str]:
        """Suggest sub-bahrs for the given bahr"""
        if not constraints.meter:
            return []
        
        return self.meters_manager.get_sub_bahrs(constraints.meter)
    
    def get_bahr_info(self, bahr_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a bahr"""
        meter_info = self.meters_manager.get_meter_info(bahr_name)
        if not meter_info:
            return None
        
        return {
            'name': meter_info.name,
            'arabic_name': meter_info.arabic_name,
            'tafeelat': meter_info.tafeelat,
            'sub_bahrs': meter_info.sub_bahrs,
            'description': meter_info.description,
            'common_themes': meter_info.common_themes,
            'difficulty_level': meter_info.difficulty_level
        } 
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the bahr selection node.
        
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
        
        # Select bahr
        enhanced_constraints = self.select_bahr(constraints, user_prompt)
        
        return {
            'constraints': enhanced_constraints,
            'bahr_selected': True
        }
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['constraints', 'user_prompt']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['constraints', 'bahr_selected'] 


 
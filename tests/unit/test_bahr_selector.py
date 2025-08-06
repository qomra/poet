# tests/unit/test_bahr_selector.py

import pytest
from unittest.mock import Mock, patch
from poet.analysis.bahr_selector import BahrSelector, BahrSelectionError
from poet.models.constraints import Constraints
from poet.llm.base_llm import LLMConfig
from poet.prompts.prompt_manager import PromptManager


class TestBahrSelector:
    """Unit tests for BahrSelector"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_llm.config = LLMConfig(model_name="test-model")
        self.selector = BahrSelector(self.mock_llm)
        self.prompt_manager = PromptManager()
    
    def test_initialization(self):
        """Test BahrSelector initialization"""
        selector = BahrSelector(self.mock_llm)
        assert selector.llm == self.mock_llm
        assert selector.prompt_manager is not None
        assert selector.meters_manager is not None
    
    def test_initialization_with_custom_prompt_manager(self):
        """Test BahrSelector initialization with custom prompt manager"""
        custom_prompt_manager = PromptManager()
        selector = BahrSelector(self.mock_llm, custom_prompt_manager)
        assert selector.prompt_manager == custom_prompt_manager
    
    def test_is_bahr_complete_with_valid_meter(self):
        """Test _is_bahr_complete with valid meter"""
        constraints = Constraints(meter="بحر الطويل")
        assert self.selector._is_bahr_complete(constraints) is True
    
    def test_is_bahr_complete_with_invalid_meter(self):
        """Test _is_bahr_complete with invalid meter"""
        constraints = Constraints(meter="بحر غير موجود")
        assert self.selector._is_bahr_complete(constraints) is False
    
    def test_is_bahr_complete_without_meter(self):
        """Test _is_bahr_complete without meter"""
        constraints = Constraints()
        assert self.selector._is_bahr_complete(constraints) is False
    
    def test_get_missing_bahr_components_no_meter(self):
        """Test _get_missing_bahr_components when no meter is specified"""
        constraints = Constraints()
        missing = self.selector._get_missing_bahr_components(constraints)
        assert missing == ['meter_name']
    
    def test_get_missing_bahr_components_invalid_meter(self):
        """Test _get_missing_bahr_components with invalid meter"""
        constraints = Constraints(meter="بحر غير موجود")
        missing = self.selector._get_missing_bahr_components(constraints)
        assert missing == ['meter_standardization']
    
    def test_get_missing_bahr_components_valid_meter(self):
        """Test _get_missing_bahr_components with valid meter"""
        constraints = Constraints(meter="بحر الطويل")
        missing = self.selector._get_missing_bahr_components(constraints)
        assert missing == []
    
    def test_validate_existing_bahr(self):
        """Test _validate_existing_bahr"""
        constraints = Constraints(meter="بحر الطويل")
        result = self.selector._validate_existing_bahr(constraints)
        assert result == constraints
    
    def test_get_relevant_meters_info_with_theme(self):
        """Test _get_relevant_meters_info with theme"""
        constraints = Constraints(theme="غزل")
        info = self.selector._get_relevant_meters_info(constraints)
        assert "مقترحات للبحر حسب الموضوع" in info
        assert "غزل" in info
    
    def test_get_relevant_meters_info_without_theme(self):
        """Test _get_relevant_meters_info without theme"""
        constraints = Constraints()
        info = self.selector._get_relevant_meters_info(constraints)
        assert "بحور سهلة للمبتدئين" in info
    
    def test_get_relevant_meters_info_with_easy_tone(self):
        """Test _get_relevant_meters_info with easy tone"""
        constraints = Constraints(tone="بسيط")
        info = self.selector._get_relevant_meters_info(constraints)
        assert "بحور سهلة للمبتدئين" in info
    
    def test_parse_llm_response_valid_json(self):
        """Test _parse_llm_response with valid JSON"""
        response = '''```json
        {
            "meter_name": "بحر الطويل",
            "explanation": "مناسب للغزل"
        }
        ```'''
        
        result = self.selector._parse_llm_response(response)
        assert result["meter_name"] == "بحر الطويل"
        assert result["explanation"] == "مناسب للغزل"
    
    def test_parse_llm_response_plain_json(self):
        """Test _parse_llm_response with plain JSON"""
        response = '{"meter_name": "بحر الكامل", "explanation": "مناسب للمدح"}'
        
        result = self.selector._parse_llm_response(response)
        assert result["meter_name"] == "بحر الكامل"
        assert result["explanation"] == "مناسب للمدح"
    
    def test_parse_llm_response_no_json(self):
        """Test _parse_llm_response with no JSON"""
        response = "This is not JSON"
        
        with pytest.raises(BahrSelectionError, match="No JSON found in LLM response"):
            self.selector._parse_llm_response(response)
    
    def test_parse_llm_response_invalid_json(self):
        """Test _parse_llm_response with invalid JSON"""
        response = '{"meter_name": "بحر الطويل", "explanation":}'
        
        with pytest.raises(BahrSelectionError, match="Failed to parse JSON"):
            self.selector._parse_llm_response(response)
    
    def test_validate_response_structure_valid(self):
        """Test _validate_response_structure with valid data"""
        data = {"meter_name": "بحر الطويل"}
        # Should not raise exception
        self.selector._validate_response_structure(data)
    
    def test_validate_response_structure_missing_field(self):
        """Test _validate_response_structure with missing field"""
        data = {"explanation": "test"}
        
        with pytest.raises(BahrSelectionError, match="Missing required field: meter_name"):
            self.selector._validate_response_structure(data)
    
    def test_validate_response_structure_null_field(self):
        """Test _validate_response_structure with null field"""
        data = {"meter_name": None}
        
        with pytest.raises(BahrSelectionError, match="Field meter_name cannot be null"):
            self.selector._validate_response_structure(data)
    
    def test_validate_response_structure_wrong_type(self):
        """Test _validate_response_structure with wrong field type"""
        data = {"meter_name": 123}
        
        with pytest.raises(BahrSelectionError, match="Field meter_name must be a string"):
            self.selector._validate_response_structure(data)
    
    def test_validate_response_structure_invalid_meter(self):
        """Test _validate_response_structure with invalid meter name"""
        data = {"meter_name": "بحر غير موجود"}
        
        with pytest.raises(BahrSelectionError, match="Invalid meter name"):
            self.selector._validate_response_structure(data)
    
    def test_enhance_constraints(self):
        """Test _enhance_constraints"""
        original_constraints = Constraints(
            qafiya="ل",
            line_count=4,
            theme="غزل"
        )
        
        bahr_spec = {"meter_name": "بحر الطويل"}
        
        enhanced = self.selector._enhance_constraints(original_constraints, bahr_spec)
        
        assert enhanced.meter == "بحر الطويل"
        assert enhanced.qafiya == "ل"
        assert enhanced.line_count == 4
        assert enhanced.theme == "غزل"
    
    def test_suggest_sub_bahrs(self):
        """Test suggest_sub_bahrs"""
        constraints = Constraints(meter="بحر الكامل")
        sub_bahrs = self.selector.suggest_sub_bahrs(constraints)
        
        assert len(sub_bahrs) > 0
        assert "بحر الكامل المجزوء" in sub_bahrs
    
    def test_suggest_sub_bahrs_no_meter(self):
        """Test suggest_sub_bahrs without meter"""
        constraints = Constraints()
        sub_bahrs = self.selector.suggest_sub_bahrs(constraints)
        
        assert sub_bahrs == []
    
    def test_get_bahr_info(self):
        """Test get_bahr_info"""
        info = self.selector.get_bahr_info("بحر الطويل")
        
        assert info is not None
        assert info["name"] == "Taweel"
        assert info["arabic_name"] == "بحر الطويل"
        assert len(info["tafeelat"]) > 0
        # Note: بحر الطويل doesn't have sub-bahrs in current implementation
        assert isinstance(info["sub_bahrs"], list)
        assert info["difficulty_level"] in ["easy", "medium", "hard"]
    
    def test_get_bahr_info_unknown_meter(self):
        """Test get_bahr_info with unknown meter"""
        info = self.selector.get_bahr_info("بحر غير موجود")
        assert info is None
    
    def test_select_bahr_already_complete(self):
        """Test select_bahr when bahr is already complete"""
        constraints = Constraints(meter="بحر الطويل")
        original_prompt = "اكتب قصيدة على بحر الطويل"
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الطويل"
        # Should not call LLM since bahr is already complete
        self.mock_llm.generate.assert_not_called()
    
    def test_select_bahr_missing_meter(self):
        """Test select_bahr when meter is missing"""
        constraints = Constraints(theme="غزل")
        original_prompt = "اكتب قصيدة غزل"
        
        # Mock LLM response
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الكامل",
            "explanation": "مناسب للغزل"
        }
        ```'''
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الكامل"
        assert result.theme == "غزل"
        self.mock_llm.generate.assert_called_once()
    
    def test_select_bahr_invalid_meter(self):
        """Test select_bahr with invalid meter"""
        constraints = Constraints(meter="بحر غير موجود")
        original_prompt = "اكتب قصيدة"
        
        # Mock LLM response to standardize the meter
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الطويل",
            "explanation": "توحيد اسم البحر"
        }
        ```'''
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الطويل"
        self.mock_llm.generate.assert_called_once()
    
    def test_select_bahr_llm_error(self):
        """Test select_bahr when LLM raises error"""
        constraints = Constraints(theme="غزل")
        original_prompt = "اكتب قصيدة غزل"
        
        self.mock_llm.generate.side_effect = Exception("LLM error")
        
        with pytest.raises(BahrSelectionError, match="Bahr selection failed"):
            self.selector.select_bahr(constraints, original_prompt)
    
    def test_select_bahr_invalid_json_response(self):
        """Test select_bahr with invalid JSON response"""
        constraints = Constraints(theme="غزل")
        original_prompt = "اكتب قصيدة غزل"
        
        self.mock_llm.generate.return_value = "Invalid response"
        
        with pytest.raises(BahrSelectionError, match="Bahr selection failed"):
            self.selector.select_bahr(constraints, original_prompt)
    
    def test_select_bahr_with_prompt_manager_injection(self):
        """Test select_bahr with custom prompt manager"""
        custom_prompt_manager = PromptManager()
        selector = BahrSelector(self.mock_llm, custom_prompt_manager)
        
        constraints = Constraints(theme="غزل")
        original_prompt = "اكتب قصيدة غزل"
        
        # Mock LLM response
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الكامل",
            "explanation": "مناسب للغزل"
        }
        ```'''
        
        result = selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الكامل"
        # Verify custom prompt manager was used
        assert selector.prompt_manager == custom_prompt_manager
    
    def test_select_bahr_preserves_other_constraints(self):
        """Test that select_bahr preserves other constraint fields"""
        constraints = Constraints(
            qafiya="ل",
            line_count=4,
            theme="غزل",
            tone="حزينة",
            imagery=["الدموع", "الفراق"],
            keywords=["حب", "شوق"],
            register="فصيح",
            era="كلاسيكي",
            poet_style="المتنبي"
        )
        original_prompt = "اكتب قصيدة غزل حزينة"
        
        # Mock LLM response
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الكامل",
            "explanation": "مناسب للغزل الحزين"
        }
        ```'''
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        # Meter should be updated
        assert result.meter == "بحر الكامل"
        
        # Other fields should be preserved
        assert result.qafiya == "ل"
        assert result.line_count == 4
        assert result.theme == "غزل"
        assert result.tone == "حزينة"
        assert result.imagery == ["الدموع", "الفراق"]
        assert result.keywords == ["حب", "شوق"]
        assert result.register == "فصيح"
        assert result.era == "كلاسيكي"
        assert result.poet_style == "المتنبي"
    
    def test_select_bahr_with_sub_bahr(self):
        """Test select_bahr with sub-bahr specification"""
        constraints = Constraints(meter="بحر الكامل المجزوء")
        original_prompt = "اكتب قصيدة على بحر الكامل المجزوء"
        
        # Mock LLM response in case sub-bahr validation fails
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الكامل المجزوء",
            "explanation": "توحيد اسم البحر المجزوء"
        }
        ```'''
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الكامل المجزوء"
        # May or may not call LLM depending on validation
    
    def test_select_bahr_standardizes_meter_name(self):
        """Test that select_bahr standardizes meter names"""
        # Test various input formats
        test_cases = [
            ("طويل", "بحر الطويل"),
            ("الكامل", "بحر الكامل"),
            ("الوافر", "بحر الوافر"),
            ("بحر الطويل", "بحر الطويل"),  # Already correct
        ]
        
        for input_meter, expected_meter in test_cases:
            constraints = Constraints(meter=input_meter)
            original_prompt = f"اكتب قصيدة على {input_meter}"
            
            # Mock LLM response to standardize
            self.mock_llm.generate.return_value = f'''```json
            {{
                "meter_name": "{expected_meter}",
                "explanation": "توحيد اسم البحر"
            }}
            ```'''
            
            result = self.selector.select_bahr(constraints, original_prompt)
            assert result.meter == expected_meter
    
    def test_select_bahr_handles_theme_based_selection(self):
        """Test that select_bahr considers theme for meter selection"""
        constraints = Constraints(theme="مدح")
        original_prompt = "اكتب قصيدة مدح"
        
        # Mock LLM response
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الوافر",
            "explanation": "مناسب للمدح والفخر"
        }
        ```'''
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الوافر"
        assert result.theme == "مدح"
        
        # Verify that theme information was included in the prompt
        call_args = self.mock_llm.generate.call_args[0][0]
        assert "مدح" in call_args
        assert "الموضوع" in call_args
    
    def test_select_bahr_handles_tone_based_selection(self):
        """Test that select_bahr considers tone for meter selection"""
        constraints = Constraints(tone="حزينة")
        original_prompt = "اكتب قصيدة حزينة"
        
        # Mock LLM response
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الكامل",
            "explanation": "مناسب للقصائد الحزينة"
        }
        ```'''
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الكامل"
        assert result.tone == "حزينة"
        
        # Verify that tone information was included in the prompt
        call_args = self.mock_llm.generate.call_args[0][0]
        assert "حزينة" in call_args
        assert "النبرة" in call_args
    
    def test_select_bahr_handles_line_count_consideration(self):
        """Test that select_bahr considers line count for meter selection"""
        constraints = Constraints(line_count=2)
        original_prompt = "اكتب قصيدة من بيتين"
        
        # Mock LLM response
        self.mock_llm.generate.return_value = '''```json
        {
            "meter_name": "بحر الوافر",
            "explanation": "مناسب للقصائد القصيرة"
        }
        ```'''
        
        result = self.selector.select_bahr(constraints, original_prompt)
        
        assert result.meter == "بحر الوافر"
        assert result.line_count == 2
        
        # Verify that line count information was included in the prompt
        call_args = self.mock_llm.generate.call_args[0][0]
        assert "2" in call_args
        assert "عدد الأبيات" in call_args 
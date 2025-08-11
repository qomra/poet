# tests/unit/test_qafiya_selector.py

import pytest
from unittest.mock import Mock, patch
from poet.analysis.qafiya_selector import QafiyaSelector, QafiyaSelectionError
from poet.models.constraints import Constraints, QafiyaType
from poet.llm.base_llm import MockLLM, LLMConfig


class TestQafiyaSelector:
    """Test cases for QafiyaSelector"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM"""
        config = LLMConfig(model_name="test-model")
        return MockLLM(config)
    
    @pytest.fixture
    def qafiya_selector(self, mock_llm):
        """Create a QafiyaSelector instance"""
        return QafiyaSelector(mock_llm)
    
    @pytest.fixture
    def basic_constraints(self):
        """Create basic user constraints"""
        return Constraints(
            meter="بحر الطويل",
            theme="غزل",
            tone="حزينة",
            era="كلاسيكي"
        )
    
    def test_is_qafiya_complete_with_complete_spec(self, qafiya_selector):
        """Test that complete qafiya specification is detected"""
        constraints = Constraints(
            qafiya="ع",
            qafiya_harakah="مكسور",
            qafiya_type=QafiyaType.MUTAWATIR,
        )
        
        assert qafiya_selector._is_qafiya_complete(constraints) is True
    
    def test_is_qafiya_complete_with_incomplete_spec(self, qafiya_selector, basic_constraints):
        """Test that incomplete qafiya specification is detected"""
        assert qafiya_selector._is_qafiya_complete(basic_constraints) is False
    
    def test_is_qafiya_complete_with_partial_spec(self, qafiya_selector):
        """Test that partial qafiya specification is detected as incomplete"""
        constraints = Constraints(qafiya="ع")  # Only letter specified
        assert qafiya_selector._is_qafiya_complete(constraints) is False
    
    def test_get_missing_qafiya_components(self, qafiya_selector):
        """Test identification of missing qafiya components"""
        # No qafiya specified
        constraints = Constraints()
        missing = qafiya_selector._get_missing_qafiya_components(constraints)
        assert set(missing) == {'qafiya_letter', 'qafiya_harakah', 'qafiya_type'}
        
        # Only letter specified
        constraints = Constraints(qafiya="ع")
        missing = qafiya_selector._get_missing_qafiya_components(constraints)
        assert set(missing) == {'qafiya_harakah', 'qafiya_type'}
        
        # Letter and harakah specified
        constraints = Constraints(qafiya="ع", qafiya_harakah="مكسور")
        missing = qafiya_selector._get_missing_qafiya_components(constraints)
        assert set(missing) == {'qafiya_type'}
        
        # Complete specification
        constraints = Constraints(
            qafiya="ع",
            qafiya_harakah="مكسور",
            qafiya_type=QafiyaType.MUTAWATIR
        )
        missing = qafiya_selector._get_missing_qafiya_components(constraints)
        assert missing == []
    
    def test_select_qafiya_with_complete_spec(self, qafiya_selector):
        """Test that complete qafiya specification is returned as-is"""
        constraints = Constraints(
            qafiya="ع",
            qafiya_harakah="مكسور",
            qafiya_type=QafiyaType.MUTAWATIR
        )
        
        result = qafiya_selector.select_qafiya(constraints, "test prompt")
        
        assert result.qafiya == "ع"
        assert result.qafiya_harakah == "مكسور"
        assert result.qafiya_type == QafiyaType.MUTAWATIR
    
    def test_select_qafiya_with_no_spec(self, qafiya_selector, basic_constraints, mock_llm):
        """Test qafiya selection when no qafiya is specified"""
        # Mock LLM response for complete qafiya selection
        mock_response = '''
        ```json
        {
            "qafiya_letter": "ع",
            "qafiya_harakah": "مكسور",
            "qafiya_type": "متواتر",
            "explanation": "قافية العين المكسورة من نوع المتواتر، مناسبة للغزل والحب"
        }
        ```
        '''
        mock_llm.responses = [mock_response]
        mock_llm.reset()
        
        result = qafiya_selector.select_qafiya(basic_constraints, "أريد قصيدة غزل حزينة")
        
        assert result.qafiya == "ع"
        assert result.qafiya_harakah == "مكسور"
        assert result.qafiya_type == QafiyaType.MUTAWATIR
    
    def test_select_qafiya_with_partial_spec_letter_only(self, qafiya_selector, mock_llm):
        """Test qafiya selection with only letter specified"""
        # User specified only the letter
        constraints = Constraints(
            meter="بحر الطويل",
            theme="غزل",
            qafiya="ع"  # Only letter specified
        )
        
        # Mock LLM response for filling missing components
        mock_response = '''
        ```json
        {
            "qafiya_letter": "ع",
            "qafiya_harakah": "مكسور",
            "qafiya_type": "متواتر",
            "explanation": "تم إكمال القافية بملء المكونات المفقودة"
        }
        ```
        '''
        mock_llm.responses = [mock_response]
        mock_llm.reset()
        
        result = qafiya_selector.select_qafiya(constraints, "أريد قصيدة غزل حزينة")
        
        # Should preserve the specified letter and fill the rest
        assert result.qafiya == "ع"
        assert result.qafiya_harakah == "مكسور"
        assert result.qafiya_type == QafiyaType.MUTAWATIR
    
    def test_select_qafiya_with_partial_spec_letter_and_harakah(self, qafiya_selector, mock_llm):
        """Test qafiya selection with letter and harakah specified"""
        # User specified letter and harakah
        constraints = Constraints(
            meter="بحر الطويل",
            theme="هجاء",
            qafiya="ق",
            qafiya_harakah="مضموم"
        )
        
        # Mock LLM response for filling missing components
        mock_response = '''
        ```json
        {
            "qafiya_letter": "ق",
            "qafiya_harakah": "مضموم",
            "qafiya_type": "متواتر",
            "explanation": "تم إكمال القافية بملء المكونات المفقودة"
        }
        ```
        '''
        mock_llm.responses = [mock_response]
        mock_llm.reset()
        
        result = qafiya_selector.select_qafiya(constraints, "أريد قصيدة هجاء")
        
        # Should preserve the specified letter and harakah
        assert result.qafiya == "ق"
        assert result.qafiya_harakah == "مضموم"
        assert result.qafiya_type == QafiyaType.MUTAWATIR
    
    def test_enhance_constraints(self, qafiya_selector, basic_constraints):
        """Test constraint enhancement with qafiya specification"""
        qafiya_spec = {
            "qafiya_letter": "ر",
            "qafiya_harakah": "مضموم",
            "qafiya_type": "متواتر",
        }
        
        result = qafiya_selector._enhance_constraints(basic_constraints, qafiya_spec)
        
        assert result.meter == "بحر الطويل"
        assert result.theme == "غزل"
        assert result.qafiya == "ر"
        assert result.qafiya_harakah == "مضموم"
        assert result.qafiya_type == QafiyaType.MUTAWATIR
    
    def test_get_harakah_symbol(self, qafiya_selector):
        """Test harakah symbol conversion"""
        assert qafiya_selector._get_harakah_symbol("مفتوح") == "َ"
        assert qafiya_selector._get_harakah_symbol("مكسور") == "ِ"
        assert qafiya_selector._get_harakah_symbol("مضموم") == "ُ"
        assert qafiya_selector._get_harakah_symbol("ساكن") == "ْ"
        assert qafiya_selector._get_harakah_symbol("غير معروف") == ""
    
    def test_parse_llm_response_valid_json(self, qafiya_selector):
        """Test parsing valid LLM response"""
        response = '''
        ```json
        {
            "qafiya_letter": "ع",
            "qafiya_harakah": "مكسور",
            "qafiya_type": "متواتر",
            "qafiya_type_description_and_examples": "متواتر: متحرك واحد بين ساكنين"
        }
        ```
        '''
        
        result = qafiya_selector._parse_llm_response(response)
        
        assert result["qafiya_letter"] == "ع"
        assert result["qafiya_harakah"] == "مكسور"
        assert result["qafiya_type"] == "متواتر"
        assert result["qafiya_type_description_and_examples"] == "متواتر: متحرك واحد بين ساكنين"
        
    def test_parse_llm_response_invalid_json(self, qafiya_selector):
        """Test parsing invalid LLM response"""
        response = "invalid json response"
        
        with pytest.raises(QafiyaSelectionError):
            qafiya_selector._parse_llm_response(response)
    
    def test_parse_llm_response_no_json(self, qafiya_selector):
        """Test parsing response with no JSON"""
        response = "This is a response without JSON"
        
        with pytest.raises(QafiyaSelectionError):
            qafiya_selector._parse_llm_response(response)
    
    def test_validate_response_structure_missing_fields(self, qafiya_selector):
        """Test validation of response structure with missing fields"""
        data = {"qafiya_letter": "ع"}  # Missing required fields
        
        with pytest.raises(ValueError, match="Missing required fields"):
            qafiya_selector._validate_response_structure(data)
    
    def test_validate_response_structure_invalid_type(self, qafiya_selector):
        """Test validation of response structure with invalid qafiya type"""
        data = {
            "qafiya_letter": "ع",
            "qafiya_harakah": "مكسور",
            "qafiya_type": "نوع غير صحيح",
        }
        
        with pytest.raises(ValueError, match="Invalid qafiya_type.*Must be one of"):
            qafiya_selector._validate_response_structure(data)
    
    def test_validate_response_structure_valid(self, qafiya_selector):
        """Test validation of response structure with valid data"""
        data = {
            "qafiya_letter": "ع",
            "qafiya_harakah": "مكسور",
            "qafiya_type": "متواتر",
        }
        
        # Should not raise any exception
        qafiya_selector._validate_response_structure(data)
    
    def test_fill_missing_qafiya_components_preserves_existing(self, qafiya_selector, mock_llm):
        """Test that existing qafiya components are preserved when filling missing ones"""
        constraints = Constraints(
            meter="بحر الطويل",
            theme="غزل",
            qafiya="ع",
            qafiya_harakah="مكسور"
        )
        
        # Mock LLM response
        mock_response = '''
        ```json
        {
            "qafiya_letter": "ر",
            "qafiya_harakah": "مفتوح",
            "qafiya_type": "متواتر",
            "explanation": "test"
        }
        ```
        '''
        mock_llm.responses = [mock_response]
        mock_llm.reset()
        
        result = qafiya_selector._fill_missing_qafiya_components(
            constraints, "test prompt", ["qafiya_type"]
        )
        
        # Should preserve existing components
        assert result["qafiya_letter"] == "ع"
        assert result["qafiya_harakah"] == "مكسور"
        # Should fill missing components
        assert result["qafiya_type"] == "متواتر"
    
    def test_qafiya_selector_error_handling(self, qafiya_selector, mock_llm):
        """Test error handling in qafiya selector"""
        constraints = Constraints(theme="غزل")
        
        # Mock LLM to raise an exception
        mock_llm.generate = Mock(side_effect=Exception("LLM error"))
        
        with pytest.raises(QafiyaSelectionError, match="Qafiya selection failed"):
            qafiya_selector.select_qafiya(constraints, "test prompt") 
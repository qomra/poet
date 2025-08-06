# tests/integration/test_qafiya_selector_integration.py

import os
import pytest
import json
from unittest.mock import Mock, patch
from poet.analysis.qafiya_selector import QafiyaSelector, QafiyaSelectionError
from poet.analysis.constraint_parser import ConstraintParser
from poet.models.constraints import UserConstraints, QafiyaType
from poet.llm.base_llm import MockLLM, LLMConfig
from poet.prompts.prompt_manager import PromptManager


@pytest.mark.integration
@pytest.mark.real_data
class TestQafiyaSelectorIntegration:
    """Integration tests for QafiyaSelector with constraint parser using real test data"""
    
    @pytest.fixture(scope="class")
    def prompt_manager(self):
        """Create a PromptManager instance"""
        return PromptManager()
    
    @pytest.fixture(scope="class")
    def mock_llm(self):
        """Create a mock LLM"""
        config = LLMConfig(model_name="test-model")
        return MockLLM(config)
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Load test data"""
        import json
        from pathlib import Path
        test_file = Path(__file__).parent.parent / "fixtures" / "test_data.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def real_llm(self):
        """Get real LLM if available - only when needed"""
        # Only initialize if TEST_REAL_LLMS is set
        if not os.getenv("TEST_REAL_LLMS"):
            return None
        from poet.llm.llm_factory import get_real_llm_from_env
        return get_real_llm_from_env()
    
    def _should_skip_test(self, llm_type):
        """Determine if test should be skipped based on environment variables"""
        test_real_llms = os.getenv("TEST_REAL_LLMS")
        
        # Check if environment variable is actually set to truthy value
        test_real_llms_enabled = test_real_llms and test_real_llms.lower() not in ['0', 'false', 'no', '']
        
        # Default behavior: only run mock when no environment variable is set
        if not test_real_llms_enabled:
            if llm_type != "mock":
                return True, "Only running mock tests when no environment variables are set"
        
        # If TEST_REAL_LLMS is set, only run real LLM tests
        elif test_real_llms_enabled:
            if llm_type != "real":
                return True, "Only running real tests when TEST_REAL_LLMS is set"
        
        return False, None
    
    def _create_components(self, llm_type, mock_llm, real_llm, prompt_manager):
        """Create ConstraintParser and QafiyaSelector with appropriate LLM"""
        if llm_type == "mock":
            llm = mock_llm
        elif llm_type == "real":
            if real_llm is None:
                pytest.skip("Real LLM not available")
            llm = real_llm
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")
        
        # Create components with the same LLM
        constraint_parser = ConstraintParser(llm, prompt_manager)
        qafiya_selector = QafiyaSelector(llm, prompt_manager)
        
        return constraint_parser, qafiya_selector
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    def test_qafiya_selection_example_1(self, llm_type, mock_llm, real_llm, prompt_manager, test_data):
        """Test qafiya selection for example 1 (غزل poem with قافية القاف)"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Create components
        constraint_parser, qafiya_selector = self._create_components(llm_type, mock_llm, real_llm, prompt_manager)
        
        # Set up mock responses for constraint parsing and qafiya selection
        if llm_type == "mock":
            # Mock response for constraint parsing
            constraint_response = '''```json
{
    "meter": "بحر الكامل",
    "qafiya": "ق",
    "line_count": 2,
    "theme": "غزل",
    "tone": "حزينة",
    "imagery": ["الدموع", "الفراق", "القلب", "الألم العاطفي"],
    "keywords": ["متيم", "فراق", "فؤاد", "دمع", "أجفان", "قلب"],
    "register": "فصيح",
    "era": "كلاسيكي",
    "poet_style": null,
    "sections": [],
    "ambiguities": [],
    "suggestions": null,
    "reasoning": "النص واضح جداً في تحديد جميع المتطلبات"
}
```'''
            
            # Mock response for qafiya selection (completing the ق with harakah and type)
            qafiya_response = '''```json
{
    "qafiya_letter": "ق",
    "qafiya_harakah": "مضموم",
    "qafiya_type": "متواتر",
    "qafiya_pattern": "قُ",
    "explanation": "قافية القاف المضمومة من نوع المتواتر، مناسبة للغزل الحزين"
}
```'''
            
            mock_llm.responses = [constraint_response, qafiya_response]
            mock_llm.reset()
        
        # Parse constraints from user prompt
        constraints = constraint_parser.parse_constraints(user_prompt)
        
        # Verify basic constraint parsing
        assert constraints.meter in ["بحر الكامل", "الكامل"]  # Real LLM might format differently
        assert constraints.qafiya == "ق"
        assert constraints.line_count == 2

        
        # Apply qafiya selection to complete the specification
        enhanced_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
        
        # Verify qafiya specification is complete
        assert enhanced_constraints.qafiya == "ق"  # Letter preserved
        assert enhanced_constraints.qafiya_harakah is not None
        assert enhanced_constraints.qafiya_type is not None
        assert enhanced_constraints.qafiya_pattern is not None
        
        # Verify qafiya is appropriate for ghazal theme
        assert enhanced_constraints.qafiya in ["ق", "ع", "ر", "ل", "ن", "م"]
        assert enhanced_constraints.qafiya_harakah in ["مكسور", "مضموم", "مفتوح"]
        assert enhanced_constraints.qafiya_type in QafiyaType
        
        # Verify pattern matches letter and harakah
        expected_pattern = f"ق{qafiya_selector._get_harakah_symbol(enhanced_constraints.qafiya_harakah)}"
        assert enhanced_constraints.qafiya_pattern == expected_pattern
        
        print(f"Example 1 - Qafiya: {enhanced_constraints.qafiya} ({enhanced_constraints.qafiya_harakah}) - {enhanced_constraints.qafiya_type.value} - {enhanced_constraints.qafiya_pattern}")
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    def test_qafiya_selection_example_2(self, llm_type, mock_llm, real_llm, prompt_manager, test_data):
        """Test qafiya selection for example 2 (هجاء poem with قافية العين)"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[1]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Create components
        constraint_parser, qafiya_selector = self._create_components(llm_type, mock_llm, real_llm, prompt_manager)
        
        # Set up mock responses for constraint parsing and qafiya selection
        if llm_type == "mock":
            # Mock response for constraint parsing
            constraint_response = '''```json
{
    "meter": "بحر الطويل",
    "qafiya": "ع",
    "line_count": 6,
    "theme": "هجاء",
    "tone": "حزينة",
    "imagery": ["الدهر", "القسوة", "الخيبة", "الأمل الضائع", "اللؤم"],
    "keywords": ["دهر", "زمن", "قسوة", "لئيم", "بخل", "أمل", "خيبة"],
    "register": "فصيح",
    "era": "كلاسيكي",
    "poet_style": null,
    "sections": [],
    "ambiguities": ["النبرة مركبة: حزينة وساخرة - تم اختيار الحزينة كنبرة أساسية"],
    "suggestions": "يمكن دمج السخرية في الأسلوب البلاغي مع الحفاظ على النبرة الحزينة الأساسية",
    "reasoning": "النص محدد بوضوح: البحر (الطويل)، القافية (العين)، عدد الأبيات (6)، الموضوع (هجاء الزمن)"
}
```'''
            
            # Mock response for qafiya selection (completing the ع with harakah and type)
            qafiya_response = '''```json
{
    "qafiya_letter": "ع",
    "qafiya_harakah": "مكسور",
    "qafiya_type": "متواتر",
    "qafiya_pattern": "عِ",
    "explanation": "قافية العين المكسورة من نوع المتواتر، مناسبة للهجاء الحزين"
}
```'''
            
            mock_llm.responses = [constraint_response, qafiya_response]
            mock_llm.reset()
        
        # Parse constraints from user prompt
        constraints = constraint_parser.parse_constraints(user_prompt)
        
        # Verify basic constraint parsing
        assert constraints.meter in ["بحر الطويل", "الطويل"]  # Real LLM might format differently
        assert constraints.qafiya == "ع"
        assert constraints.line_count == 6
       
        
        # Apply qafiya selection to complete the specification
        enhanced_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
        
        # Verify qafiya specification is complete
        assert enhanced_constraints.qafiya == "ع"  # Letter preserved
        assert enhanced_constraints.qafiya_harakah is not None
        assert enhanced_constraints.qafiya_type is not None
        assert enhanced_constraints.qafiya_pattern is not None
        
        # Verify qafiya is appropriate for hijaa theme
        assert enhanced_constraints.qafiya in ["ع", "ق", "ح", "خ", "غ"]
        assert enhanced_constraints.qafiya_harakah in ["مكسور", "مضموم", "مفتوح", "ساكن"]
        assert enhanced_constraints.qafiya_type in QafiyaType
        
        # Verify pattern matches letter and harakah
        expected_pattern = f"ع{qafiya_selector._get_harakah_symbol(enhanced_constraints.qafiya_harakah)}"
        assert enhanced_constraints.qafiya_pattern == expected_pattern
        
        print(f"Example 2 - Qafiya: {enhanced_constraints.qafiya} ({enhanced_constraints.qafiya_harakah}) - {enhanced_constraints.qafiya_type.value} - {enhanced_constraints.qafiya_pattern}")
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    def test_qafiya_selection_without_specified_qafiya(self, llm_type, mock_llm, real_llm, prompt_manager):
        """Test qafiya selection when no qafiya is specified in the prompt"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type)
        if should_skip:
            pytest.skip(reason)
        
        # Create a prompt without specified qafiya
        user_prompt = "أريد قصيدة غزل حزينة على بحر الطويل من 4 أبيات"
        
        # Create components
        constraint_parser, qafiya_selector = self._create_components(llm_type, mock_llm, real_llm, prompt_manager)
        
        # Set up mock responses
        if llm_type == "mock":
            # Mock response for constraint parsing (no qafiya specified)
            constraint_response = '''```json
{
    "meter": "بحر الطويل",
    "qafiya": null,
    "line_count": 4,
    "theme": "غزل",
    "tone": "حزينة",
    "imagery": ["الحب", "الفراق", "الحزن"],
    "keywords": ["غزل", "حب", "فراق", "حزن"],
    "register": "فصيح",
    "era": "كلاسيكي",
    "poet_style": null,
    "sections": [],
    "ambiguities": [],
    "suggestions": null,
    "reasoning": "النص يحدد البحر والموضوع والنبرة لكن لا يحدد قافية"
}
```'''
            
            # Mock response for qafiya selection (LLM chooses appropriate qafiya)
            qafiya_response = '''```json
{
    "qafiya_letter": "ر",
    "qafiya_harakah": "مضموم",
    "qafiya_type": "متواتر",
    "qafiya_pattern": "رُ",
    "explanation": "قافية الراء المضمومة من نوع المتواتر، مناسبة للغزل الحزين"
}
```'''
            
            mock_llm.responses = [constraint_response, qafiya_response]
            mock_llm.reset()
        
        # Parse constraints from user prompt
        constraints = constraint_parser.parse_constraints(user_prompt)
        
        # Verify constraint parsing
        assert constraints.meter in ["بحر الطويل", "الطويل"]  # Real LLM might format differently
        assert constraints.qafiya is None  # No qafiya specified
        assert constraints.line_count == 4
        assert constraints.theme in ["غزل", "قصيدة غزل"]  # Real LLM might format differently
        assert constraints.tone == "حزينة"
        
        # Apply qafiya selection to choose appropriate qafiya
        enhanced_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
        
        # Verify qafiya specification is complete
        assert enhanced_constraints.qafiya is not None
        assert enhanced_constraints.qafiya_harakah is not None
        assert enhanced_constraints.qafiya_type is not None
        assert enhanced_constraints.qafiya_pattern is not None
        
        # Verify qafiya is appropriate for ghazal theme
        assert enhanced_constraints.qafiya in ["ع", "ر", "ل", "ن", "م"]
        assert enhanced_constraints.qafiya_harakah in ["مكسور", "مضموم", "مفتوح"]
        assert enhanced_constraints.qafiya_type in QafiyaType
        
        print(f"No qafiya specified - Selected: {enhanced_constraints.qafiya} ({enhanced_constraints.qafiya_harakah}) - {enhanced_constraints.qafiya_type.value} - {enhanced_constraints.qafiya_pattern}") 
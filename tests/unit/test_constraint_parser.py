import pytest
import json
from unittest.mock import Mock, MagicMock
from poet.analysis.constraint_parser import ConstraintParser, ConstraintParsingError
from poet.models.constraints import Constraints
from poet.prompts.prompt_manager import PromptManager
from poet.llm.base_llm import BaseLLM, MockLLM, LLMConfig
from poet.llm.llm_factory import get_real_llm_from_env


class TestConstraintParser:
    
    @pytest.fixture
    def constraint_parser(self, mock_llm, prompt_manager):
        """ConstraintParser instance with mocked LLM"""
        return ConstraintParser(mock_llm, prompt_manager)
    
    @pytest.fixture
    def real_llm(self):
        """Real LLM instance if available, otherwise None"""
        return get_real_llm_from_env()
    
    @pytest.fixture
    def constraint_parser_real(self, real_llm, prompt_manager):
        """ConstraintParser instance with real LLM if available"""
        if real_llm is None:
            pytest.skip("Real LLM not available")
        return ConstraintParser(real_llm, prompt_manager)
    
    @pytest.fixture(params=["mock", "real"])
    def llm_type(self, request):
        """Parametrize tests to run with both mock and real LLMs"""
        return request.param
    
    @pytest.fixture
    def constraint_parser_parametrized(self, llm_type, mock_llm, real_llm, prompt_manager):
        """ConstraintParser instance - mock or real based on parameter"""
        if llm_type == "mock":
            return ConstraintParser(mock_llm, prompt_manager)
        elif llm_type == "real":
            if real_llm is None:
                pytest.skip("Real LLM not available")
            return ConstraintParser(real_llm, prompt_manager)
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")
    
    def test_parse_constraints_example_1(self, constraint_parser, mock_llm, test_data):
        """Test parsing constraints for the first example (غزل poem)"""
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Set up MockLLM with expected JSON response
        mock_response = f"""
        أنت خبير في الشعر العربي والعروض. بعد تحليل النص المقدم:
        
        ```json
        {json.dumps(expected_constraints, ensure_ascii=False, indent=2)}
        ```
        """
        
        mock_llm.responses = [mock_response]
        
        # Parse constraints
        result = constraint_parser.parse_constraints(user_prompt)
        
        # Verify LLM was called with formatted prompt
        assert mock_llm.call_count == 1
        assert 'خبير في الشعر العربي والعروض' in mock_llm.last_prompt
        assert user_prompt in mock_llm.last_prompt
        
        # Verify parsed constraints
        assert result.meter == expected_constraints["meter"]
        assert result.qafiya == expected_constraints["qafiya"]
        assert result.line_count == expected_constraints["line_count"]
        assert result.theme == expected_constraints["theme"]
        assert result.tone == expected_constraints["tone"]
        assert result.imagery == expected_constraints["imagery"]
        assert result.keywords == expected_constraints["keywords"]
        assert result.register == expected_constraints["register"]
        assert result.era == expected_constraints["era"]
        assert result.poet_style == expected_constraints["poet_style"]
        assert result.sections == expected_constraints["sections"]
        assert result.ambiguities == expected_constraints["ambiguities"]
        
        # Verify metadata
        assert result.llm_suggestions == expected_constraints["suggestions"]
        assert result.llm_reasoning == expected_constraints["reasoning"]
        assert result.original_prompt == user_prompt
        
        # Verify no ambiguities for this clear example
        assert not result.has_ambiguities()
    
    def test_parse_constraints_example_2(self, constraint_parser, mock_llm, test_data):
        """Test parsing constraints for the second example (هجاء poem)"""
        example = test_data[1]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Set up MockLLM with expected JSON response
        mock_response = f"""
        بعد تحليل النص المقدم من المستخدم:
        
        ```json
        {json.dumps(expected_constraints, ensure_ascii=False, indent=2)}
        ```
        """
        
        mock_llm.responses = [mock_response]
        
        # Parse constraints
        result = constraint_parser.parse_constraints(user_prompt)
        
        # Verify parsed constraints
        assert result.meter == expected_constraints["meter"]
        assert result.qafiya == expected_constraints["qafiya"]
        assert result.line_count == expected_constraints["line_count"]
        assert result.theme == expected_constraints["theme"]
        assert result.tone == expected_constraints["tone"]
        assert result.imagery == expected_constraints["imagery"]
        assert result.keywords == expected_constraints["keywords"]
        
        # Verify ambiguities are detected
        assert result.has_ambiguities()
        assert len(result.ambiguities) >= 1
        # Check if original ambiguity is present
        original_ambiguity_found = any("النبرة مركبة" in amb for amb in result.ambiguities)
        assert original_ambiguity_found
        
        # Verify suggestions
        assert result.llm_suggestions == expected_constraints["suggestions"]
        assert "السخرية في الأسلوب البلاغي" in result.llm_suggestions
    
    def test_parse_constraints_invalid_json(self, constraint_parser, mock_llm):
        """Test handling of invalid JSON response"""
        mock_llm.responses = ["Invalid response without JSON"]
        
        with pytest.raises(ConstraintParsingError, match="Invalid response format"):
            constraint_parser.parse_constraints("test prompt")
    
    def test_parse_constraints_malformed_json(self, constraint_parser, mock_llm):
        """Test handling of malformed JSON"""
        mock_response = """
        ```json
        {
            "meter": "بحر الكامل",
            "qafiya": "ق"
            // missing comma and closing brace
        }
        ```
        """
        
        mock_llm.responses = [mock_response]
        
        with pytest.raises(ConstraintParsingError, match="Invalid JSON response|Constraint parsing failed"):
            constraint_parser.parse_constraints("test prompt")
    
    def test_parse_constraints_missing_fields(self, constraint_parser, mock_llm):
        """Test handling of response with missing required fields"""
        incomplete_response = {
            "meter": "بحر الكامل",
            "qafiya": "ق"
            # missing many required fields
        }
        
        mock_response = f"""
        ```json
        {json.dumps(incomplete_response, ensure_ascii=False)}
        ```
        """
        
        mock_llm.responses = [mock_response]
        
        with pytest.raises(ConstraintParsingError, match="Invalid response format"):
            constraint_parser.parse_constraints("test prompt")
    
    def test_get_clarification_prompt_no_ambiguities(self, constraint_parser):
        """Test clarification prompt when no ambiguities exist"""
        constraints = Constraints(
            meter="بحر الكامل",
            theme="غزل",
            ambiguities=[]
        )
        
        result = constraint_parser.get_clarification_prompt(constraints)
        assert result is None
    
    def test_get_clarification_prompt_with_ambiguities(self, constraint_parser):
        """Test clarification prompt generation with ambiguities"""
        constraints = Constraints(
            meter="بحر الكامل",
            theme="غزل",
            ambiguities=[
                "النبرة غير واضحة",
                "اقتراح: يمكن استخدام نبرة حزينة",
                "عدد الأبيات غير محدد"
            ]
        )
        
        result = constraint_parser.get_clarification_prompt(constraints)
        
        assert result is not None
        assert "لتحسين جودة الشعر" in result
        assert "يرجى توضيح" in result
        assert "النبرة غير واضحة" in result
        assert "اقتراح: يمكن استخدام نبرة حزينة" in result
    
    def test_refine_constraints(self, constraint_parser, mock_llm):
        """Test constraint refinement with user clarification"""
        original_constraints = Constraints(
            meter="بحر الكامل",
            theme="غزل",
            ambiguities=["النبرة غير واضحة"]
        )
        original_constraints.original_prompt = "أريد قصيدة غزل"
        
        user_clarification = "أريد النبرة حزينة"
        
        # Mock refined response
        refined_constraints = {
            "meter": "بحر الكامل",
            "qafiya": None,
            "line_count": None,
            "theme": "غزل",
            "tone": "حزينة",
            "imagery": [],
            "keywords": [],
            "register": None,
            "era": None,
            "poet_style": None,
            "sections": [],
            "ambiguities": [],
            "suggestions": None,
            "reasoning": "تم توضيح النبرة كحزينة"
        }
        
        mock_response = f"""
        ```json
        {json.dumps(refined_constraints, ensure_ascii=False, indent=2)}
        ```
        """
        
        mock_llm.responses = [mock_response]
        
        # Refine constraints
        result = constraint_parser.refine_constraints(original_constraints, user_clarification)
        
        # Verify the combined prompt was used
        assert "أريد قصيدة غزل" in mock_llm.last_prompt
        assert "أريد النبرة حزينة" in mock_llm.last_prompt
        
        # Verify refined result
        assert result.tone == "حزينة"
        # Should have no ambiguities since the refined response has empty ambiguities
        assert not result.has_ambiguities()
    
    def test_validate_constraints(self, constraint_parser):
        """Test constraint validation"""
        valid_constraints = Constraints(
            meter="بحر الكامل",
            theme="غزل",
            line_count=4
        )
        
        # Should not raise any exception
        assert constraint_parser.validate_constraints(valid_constraints) is True
        
        # Test invalid constraints - should raise ValueError during construction
        with pytest.raises(ValueError, match="Line count must be positive"):
            Constraints(
                meter="بحر الكامل",
                theme="غزل",
                line_count=-1  # Invalid line count
            )
    
    def test_handle_null_values(self, constraint_parser, mock_llm):
        """Test handling of null values in LLM response"""
        response_with_nulls = {
            "meter": "بحر الكامل",
            "qafiya": "null",
            "line_count": "null",
            "theme": "غزل",
            "tone": "null",
            "imagery": [],
            "keywords": [],
            "register": "null",
            "era": "null",
            "poet_style": "null",
            "sections": [],
            "ambiguities": [],
            "suggestions": "null",
            "reasoning": "بعض القيم غير محددة"
        }
        
        mock_response = f"""
        ```json
        {json.dumps(response_with_nulls, ensure_ascii=False, indent=2)}
        ```
        """
        
        mock_llm.responses = [mock_response]
        
        result = constraint_parser.parse_constraints("test prompt")
        
                 # Verify null values are converted to None
        assert result.meter == "بحر الكامل"
        assert result.qafiya is None
        assert result.line_count is None
        assert result.theme == "غزل"
        assert result.tone is None
        assert result.register is None
        assert result.era is None
        assert result.poet_style is None
        assert result.llm_suggestions is None
    
    def test_mock_llm_capabilities(self, prompt_manager):
        """Test MockLLM specific capabilities for testing"""
        # Create MockLLM with predefined responses
        config = LLMConfig(model_name="test-model", temperature=0.5)
        responses = [
            "First response",
            "Second response", 
            "Third response"
        ]
        mock_llm = MockLLM(config, responses)
        
        constraint_parser = ConstraintParser(mock_llm, prompt_manager)
        
        # Test cycling through responses
        assert mock_llm.call_count == 0
        assert mock_llm.last_prompt is None
        
        # First call
        result1 = mock_llm.generate("prompt 1")
        assert result1 == "First response"
        assert mock_llm.call_count == 1
        assert mock_llm.last_prompt == "prompt 1"
        
        # Second call
        result2 = mock_llm.generate("prompt 2")
        assert result2 == "Second response"
        assert mock_llm.call_count == 2
        assert mock_llm.last_prompt == "prompt 2"
        
        # Third call
        result3 = mock_llm.generate("prompt 3")
        assert result3 == "Third response"
        assert mock_llm.call_count == 3
        
        # Fourth call - should cycle back to first response
        result4 = mock_llm.generate("prompt 4")
        assert result4 == "First response"
        assert mock_llm.call_count == 4
        
        # Test reset functionality
        mock_llm.reset()
        assert mock_llm.call_count == 0
        assert mock_llm.last_prompt is None
        
        # Test generate_with_metadata
        response_obj = mock_llm.generate_with_metadata("test prompt")
        assert response_obj.content == "First response"
        assert response_obj.model == "test-model"
        assert response_obj.usage['prompt_tokens'] == 2  # "test prompt" = 2 tokens
        assert response_obj.metadata['mock'] is True
        assert response_obj.metadata['call_count'] == 1
        
        # Test availability
        assert mock_llm.is_available() is True
        
        # Test model info
        info = mock_llm.get_model_info()
        assert info['provider'] == 'MockLLM'
        assert info['model'] == 'test-model'
        assert info['temperature'] == 0.5
        assert info['available'] is True

    def test_parse_constraints_example_1_parametrized(self, constraint_parser_parametrized, llm_type, test_data):
        """Test parsing constraints for example 1 with both mock and real LLMs"""
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Set up mock response if using mock LLM
        if llm_type == "mock":
            mock_response = json.dumps(expected_constraints, ensure_ascii=False)
            constraint_parser_parametrized.llm.responses = [mock_response]
            constraint_parser_parametrized.llm.reset()
        
        # Parse constraints
        result = constraint_parser_parametrized.parse_constraints(user_prompt)
        
        # Basic assertions that work for both mock and real LLMs
        assert result is not None
        assert result.original_prompt == user_prompt
        
        if llm_type == "mock":
            # For mock LLM, we can assert exact matches
            assert result.meter == expected_constraints["meter"]
            assert result.qafiya == expected_constraints["qafiya"]
            assert result.line_count == expected_constraints["line_count"]
            assert result.theme == expected_constraints["theme"]
            assert result.tone == expected_constraints["tone"]
            assert result.imagery == expected_constraints["imagery"]
            assert result.keywords == expected_constraints["keywords"]
        else:
            # For real LLM, check that some constraints were extracted
            constraint_fields = [
                result.meter, result.qafiya, result.theme, result.tone, 
                result.imagery, result.keywords
            ]
            populated_fields = sum(1 for field in constraint_fields if field)
            assert populated_fields > 0, "Real LLM should extract at least some constraints"
            
            # Check LLM metadata is populated
            assert result.llm_reasoning is not None

    def test_parse_constraints_example_2_parametrized(self, constraint_parser_parametrized, llm_type, test_data):
        """Test parsing constraints for example 2 with both mock and real LLMs"""
        example = test_data[1]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Set up mock response if using mock LLM
        if llm_type == "mock":
            mock_response = json.dumps(expected_constraints, ensure_ascii=False)
            constraint_parser_parametrized.llm.responses = [mock_response]
            constraint_parser_parametrized.llm.reset()
        
        # Parse constraints
        result = constraint_parser_parametrized.parse_constraints(user_prompt)
        
        # Basic assertions that work for both mock and real LLMs
        assert result is not None
        assert result.original_prompt == user_prompt
        
        if llm_type == "mock":
            # For mock LLM, we can assert exact matches
            assert result.meter == expected_constraints["meter"]
            assert result.qafiya == expected_constraints["qafiya"]
            assert result.line_count == expected_constraints["line_count"]
            assert result.theme == expected_constraints["theme"]
            assert result.tone == expected_constraints["tone"]
            assert result.imagery == expected_constraints["imagery"]
            assert result.keywords == expected_constraints["keywords"]
        else:
            # For real LLM, check that some constraints were extracted
            constraint_fields = [
                result.meter, result.qafiya, result.theme, result.tone, 
                result.imagery, result.keywords
            ]
            populated_fields = sum(1 for field in constraint_fields if field)
            assert populated_fields > 0, "Real LLM should extract at least some constraints"
            
            # Check LLM metadata is populated
            assert result.llm_reasoning is not None 
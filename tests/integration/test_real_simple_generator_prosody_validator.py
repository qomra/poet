# tests/integration/test_real_simple_generator_prosody_validator.py

import os
import pytest
import json
from unittest.mock import patch
from poet.generation.poem_generator import SimplePoemGenerator
from poet.evaluation.prosody_validator import ProsodyValidator
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.prompts.prompt_manager import PromptManager
from poet.llm.base_llm import MockLLM, LLMConfig


@pytest.mark.integration
@pytest.mark.real_data
class TestRealSimpleGeneratorProsodyValidator:
    """Integration tests for SimplePoemGenerator and ProsodyValidator with real/mock LLM combinations"""
    
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
        """Create SimplePoemGenerator and ProsodyValidator with appropriate LLM"""
        if llm_type == "mock":
            llm = mock_llm
        elif llm_type == "real":
            if real_llm is None:
                pytest.skip("Real LLM not available")
            llm = real_llm
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")
        
        # Create both components with the same LLM
        poem_generator = SimplePoemGenerator(llm, prompt_manager)
        prosody_validator = ProsodyValidator()
        
        return poem_generator, prosody_validator
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    def test_integration_example_1(self, llm_type, mock_llm, real_llm, prompt_manager, test_data):
        """Test integration for example 1 (غزل poem) - generate and validate with real/mock LLM"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Create components
        poem_generator, prosody_validator = self._create_components(llm_type, mock_llm, real_llm, prompt_manager)
        
        # Set up mock responses for consistent testing when using mock LLM
        if llm_type == "mock":
            # Mock response for poem generation (4 verses for example 1 - 2 baits)
            mock_poem_response = '''
            ```json
            {
                "verses": [
                    "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                    "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                    "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                    "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
                ]
            }
            ```
            '''
            
            # Mock response for diacritics (tashkeel)
            mock_tashkeel_response = '''
            ```json
            {
                "diacritized_verses": [
                    "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                    "بِسِقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                    "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                    "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
                ]
            }
            ```
            '''
            
            # Set up mock responses
            mock_llm.responses = [mock_poem_response, mock_tashkeel_response, mock_tashkeel_response]
            mock_llm.reset()
        
        # Create constraints from example
        constraints = Constraints(
            meter=expected_constraints["meter"],
            qafiya=expected_constraints["qafiya"],
            line_count=expected_constraints["line_count"],
            theme=expected_constraints["theme"],
            tone=expected_constraints["tone"]
        )
        
        # Step 1: Generate poem
        print(f"\nExample 1 - LLM Type: {llm_type}")
        print(f"Generating poem with constraints: {constraints}")
        
        poem = poem_generator.generate_poem(constraints)
        
        # Verify poem generation
        assert isinstance(poem, LLMPoem)
        # line_count represents baits, so actual verses should be line_count * 2
        expected_verses = constraints.line_count * 2
        # For mock LLM, expect exact line count; for real LLM, be more flexible
        if llm_type == "mock":
            assert len(poem.verses) == expected_verses
        else:
            # Real LLM may not follow exact line count, but should generate some verses
            assert len(poem.verses) > 0
            assert len(poem.verses) % 2 == 0  # Should be even number (complete baits)
        assert all(isinstance(verse, str) for verse in poem.verses)
        assert all(len(verse.strip()) > 0 for verse in poem.verses)
        assert poem.llm_provider == (mock_llm.__class__.__name__ if llm_type == "mock" else real_llm.__class__.__name__)
        assert poem.constraints == constraints.to_dict()
        
        print(f"Generated poem:")
        print(f"  LLM Provider: {poem.llm_provider}")
        print(f"  Model: {poem.model_name}")
        print(f"  Verses:")
        for i, verse in enumerate(poem.verses, 1):
            print(f"    {i}. {verse}")
        
        # Step 2: Validate prosody
        print(f"\nValidating prosody for meter: {constraints.meter}")
        
        validation_result = prosody_validator.validate_poem(poem, constraints.meter)
        
        # Verify validation results
        assert validation_result.prosody_validation is not None
        assert validation_result.prosody_validation.bahr_used == constraints.meter
        assert validation_result.prosody_validation.total_baits == len(poem.verses) // 2
        # Note: Quality assessment is now handled by PoemEvaluator, not ProsodyValidator
        
        print(f"Prosody validation results:")
        print(f"  Bahr used: {validation_result.prosody_validation.bahr_used}")
        print(f"  Total baits: {validation_result.prosody_validation.total_baits}")
        print(f"  Valid baits: {validation_result.prosody_validation.valid_baits}")
        print(f"  Invalid baits: {validation_result.prosody_validation.invalid_baits}")
        print(f"  Overall valid: {validation_result.prosody_validation.overall_valid}")
        print(f"  Validation summary: {validation_result.prosody_validation.validation_summary}")
        
        # Note: Quality assessment would be done by PoemEvaluator
        print(f"Quality assessment: (handled by PoemEvaluator)")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert constraints.meter == expected_constraints["meter"]
            assert constraints.theme == expected_constraints["theme"]
            assert constraints.line_count == expected_constraints["line_count"]
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    def test_integration_example_2(self, llm_type, mock_llm, real_llm, prompt_manager, test_data):
        """Test integration for example 2 (هجاء poem) - generate and validate with real/mock LLM"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[1]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Create components
        poem_generator, prosody_validator = self._create_components(llm_type, mock_llm, real_llm, prompt_manager)
        
        # Set up mock responses for consistent testing when using mock LLM
        if llm_type == "mock":
            # Mock response for poem generation (12 verses for example 2 - 6 baits)
            mock_poem_response = '''
            ```json
            {
                "verses": [
                    "تَمَكَّنَ هَذا الدَهرُ مِمّا يَسوءُني",
                    "وَلَجَّ فَما يَخلي صَفاتِيَ مِن قَرعِ",
                    "وَأَبلَيتُ آمالي بِوَصلٍ يَكُدُّها",
                    "فَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ",
                    "وَأَصبَحتُ أَشكو ما أَلاقي مِن دَفعِ"
                ]
            }
            ```
            '''
            
            # Mock response for diacritics (tashkeel)
            mock_tashkeel_response = '''
            ```json
            {
                "diacritized_verses": [
                    "تَمَكَّنَ هَذا الدَّهْرُ مِمَّا يَسُوءُني",
                    "وَلَجَّ فَما يَخْلي صَفاتِيَ مِن قَرْعِ",
                    "وَأَبْلَيْتُ آمالي بِوَصْلٍ يَكُدُّها",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "فَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ"
                ]
            }
            ```
            '''
            
            # Set up mock responses
            mock_llm.responses = [mock_poem_response, mock_tashkeel_response, mock_tashkeel_response]
            mock_llm.reset()
        
        # Create constraints from example
        constraints = Constraints(
            meter=expected_constraints["meter"],
            qafiya=expected_constraints["qafiya"],
            line_count=expected_constraints["line_count"],
            theme=expected_constraints["theme"],
            tone=expected_constraints["tone"]
        )
        
        # Step 1: Generate poem
        print(f"\nExample 2 - LLM Type: {llm_type}")
        print(f"Generating poem with constraints: {constraints}")
        
        poem = poem_generator.generate_poem(constraints)
        
        # Verify poem generation
        assert isinstance(poem, LLMPoem)
        # line_count represents baits, so actual verses should be line_count * 2
        expected_verses = constraints.line_count * 2
        # For mock LLM, expect exact line count; for real LLM, be more flexible
        if llm_type == "mock":
            assert len(poem.verses) == expected_verses
        else:
            # Real LLM may not follow exact line count, but should generate some verses
            assert len(poem.verses) > 0
            assert len(poem.verses) % 2 == 0  # Should be even number (complete baits)
        assert all(isinstance(verse, str) for verse in poem.verses)
        assert all(len(verse.strip()) > 0 for verse in poem.verses)
        assert poem.llm_provider == (mock_llm.__class__.__name__ if llm_type == "mock" else real_llm.__class__.__name__)
        assert poem.constraints == constraints.to_dict()
        
        print(f"Generated poem:")
        print(f"  LLM Provider: {poem.llm_provider}")
        print(f"  Model: {poem.model_name}")
        print(f"  Verses:")
        for i, verse in enumerate(poem.verses, 1):
            print(f"    {i}. {verse}")
        
        # Step 2: Validate prosody
        print(f"\nValidating prosody for meter: {constraints.meter}")
        
        validation_result = prosody_validator.validate_poem(poem, constraints.meter)
        
        # Verify validation results
        assert validation_result.prosody_validation is not None
        assert validation_result.prosody_validation.bahr_used == constraints.meter
        assert validation_result.prosody_validation.total_baits == len(poem.verses) // 2
        # Note: Quality assessment is now handled by PoemEvaluator, not ProsodyValidator
        
        print(f"Prosody validation results:")
        print(f"  Bahr used: {validation_result.prosody_validation.bahr_used}")
        print(f"  Total baits: {validation_result.prosody_validation.total_baits}")
        print(f"  Valid baits: {validation_result.prosody_validation.valid_baits}")
        print(f"  Invalid baits: {validation_result.prosody_validation.invalid_baits}")
        print(f"  Overall valid: {validation_result.prosody_validation.overall_valid}")
        print(f"  Validation summary: {validation_result.prosody_validation.validation_summary}")
        
        # Note: Quality assessment would be done by PoemEvaluator
        print(f"Quality assessment: (handled by PoemEvaluator)")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert constraints.meter == expected_constraints["meter"]
            assert constraints.theme == expected_constraints["theme"]
            assert constraints.line_count == expected_constraints["line_count"]
        
        # Additional analysis for real LLM tests
        if llm_type == "real":
            print(f"\nReal LLM Analysis:")
            print(f"  Poem follows meter: {constraints.meter}")
            print(f"  Poem follows theme: {constraints.theme}")
            print(f"  Poem follows qafiya: {constraints.qafiya}")
            print(f"  Prosody compliance: {validation_result.prosody_validation.overall_valid}")
            # Note: Quality score would be available from PoemEvaluator
            print(f"  Quality score: (available from PoemEvaluator)")
            
            # Provide insights on generation quality
            if validation_result.prosody_validation.overall_valid:
                print(f"  [SUCCESS] Generated poem successfully follows {constraints.meter} meter")
            else:
                print(f"  [WARNING] Generated poem has prosody issues with {constraints.meter} meter")
            
            # Note: Quality assessment would be done by PoemEvaluator
            print(f"  [INFO] Quality assessment would be done by PoemEvaluator")

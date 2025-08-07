# tests/integration/test_real_simple_generation_validation.py

import os
import pytest
import json
from unittest.mock import patch
from poet.evaluation.poem import PoemEvaluator, EvaluationType
from poet.generation.poem_generator import SimplePoemGenerator
from poet.refinement.tashkeel import TashkeelRefiner
from poet.analysis.constraint_parser import ConstraintParser
from poet.analysis.qafiya_selector import QafiyaSelector
from poet.models.constraints import Constraints
from poet.models.poem import LLMPoem
from poet.prompts.prompt_manager import PromptManager
from poet.llm.base_llm import MockLLM, LLMConfig


@pytest.mark.integration
@pytest.mark.real_data
class TestRealSimpleGenerationValidation:
    """Integration tests for complete workflow (generation + validation) using PoemEvaluator with real/mock LLM combinations"""
    
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
        """Create all components with appropriate LLM"""
        if llm_type == "mock":
            llm = mock_llm
        elif llm_type == "real":
            if real_llm is None:
                pytest.skip("Real LLM not available")
            llm = real_llm
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")
        
        # Create all components with the same LLM
        constraint_parser = ConstraintParser(llm, prompt_manager)
        qafiya_selector = QafiyaSelector(llm, prompt_manager)
        poem_generator = SimplePoemGenerator(llm, prompt_manager)
        tashkeel_refiner = TashkeelRefiner(llm, prompt_manager)
        poem_evaluator = PoemEvaluator(llm)
        
        return constraint_parser, qafiya_selector, poem_generator, tashkeel_refiner, poem_evaluator
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    @pytest.mark.asyncio
    async def test_complete_workflow_example_1(self, llm_type, mock_llm, real_llm, prompt_manager, test_data):
        """Test complete workflow for example 1 (غزل poem) - generate and evaluate with real/mock LLM"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Create components
        constraint_parser, qafiya_selector, poem_generator, tashkeel_refiner, poem_evaluator = self._create_components(llm_type, mock_llm, real_llm, prompt_manager)
        
        # Set up mock responses for consistent testing when using mock LLM
        if llm_type == "mock":
            # Mock response for constraint parsing
            mock_constraint_response = '''
            ```json
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
                "reasoning": "النص واضح جداً في تحديد جميع المتطلبات: البحر (الكامل)، القافية (القاف)، عدد الأبيات (2)، الموضوع (غزل)، والنبرة (حزينة). الصور البلاغية محددة بوضوح (الدموع، الفراق، القلب النابض). الأسلوب كلاسيكي فصيح."
            }
            ```
            '''
            
            # Mock response for qafiya selection
            mock_qafiya_selection_response = '''
            ```json
            {
                "qafiya_letter": "ق",
                "qafiya_harakah": "ساكن",
                "qafiya_type": "مترادف",
                "qafiya_pattern": "قْ"
            }
            ```
            '''
            
            # Mock response for poem generation (4 verses for example 1 - 2 baits)
            mock_poem_response = '''
            ```json
            {
                "verses": [
                    "وَمُتَيَّمٍ جَرَحَ الفُراقُ فُؤادَهُ",
                    "فَالدَمعُ مِن أَجفانِهِ يَتَدَفَّقُ",
                    "بَهَرَتهُ ساعَةُ فِرقَةٍ فَكَأَنَّما",
                    "في كُلِّ عُضوٍ مِنهُ قَلبٌ يَخفِقُ"
                ]
            }
            ```
            '''
            
            # Mock response for tashkeel (diacritics)
            mock_tashkeel_response = '''
            ```json
            {
                "diacritized_verses": [
                    "وَمُتَيَّمٍ جَرَحَ الفُراقُ فُؤادَهُ",
                    "فَالدَّمْعُ مِن أَجْفانِهِ يَتَدَفَّقُ",
                    "بَهَرَتْهُ ساعَةُ فِرْقَةٍ فَكَأَنَّما",
                    "في كُلِّ عُضوٍ مِنْهُ قَلْبٌ يَخْفِقُ"
                ]
            }
            ```
            '''
            
            # Mock response for prosody validation
            mock_prosody_response = '''
            ```json
            {
                "bahr_used": "بحر الكامل",
                "total_baits": 2,
                "valid_baits": 2,
                "invalid_baits": 0,
                "overall_valid": true,
                "validation_summary": "جميع الأبيات تتبع بحر الكامل بشكل صحيح"
            }
            ```
            '''
            
            # Mock response for qafiya validation (individual bait responses)
            mock_qafiya_validation_1 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            mock_qafiya_validation_2 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            
            # Set up mock responses
            mock_llm.responses = [
                mock_qafiya_selection_response,  # Qafiya selection
                mock_poem_response,  # Poem generation
                mock_tashkeel_response,  # Tashkeel
                mock_qafiya_validation_1,  # Qafiya validation bait 1
                mock_qafiya_validation_2   # Qafiya validation bait 2
            ]
            mock_llm.reset()
        
        # Create constraints from example
        constraints = Constraints(
            meter=expected_constraints["meter"],
            qafiya=expected_constraints["qafiya"],
            line_count=expected_constraints["line_count"],
            theme=expected_constraints["theme"],
            tone=expected_constraints["tone"]
        )
        
        # Step 2: Select qafiya (complete missing components)
        print(f"\nStep 1: Selecting qafiya...")
        enriched_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
        print(f"Enriched constraints: {enriched_constraints}")
        
        # Step 2: Generate poem
        print(f"\nStep 2: Generating poem...")
        
        poem = poem_generator.generate_poem(enriched_constraints)
        
        # Verify constraint parsing and qafiya selection
        assert isinstance(enriched_constraints, Constraints)
        
        # Verify enriched constraints have complete qafiya specifications
        assert enriched_constraints.qafiya is not None
        assert enriched_constraints.qafiya_harakah is not None
        assert enriched_constraints.qafiya_type is not None
        assert enriched_constraints.qafiya_pattern is not None
        
        # Verify poem generation
        assert isinstance(poem, LLMPoem)
        # line_count represents baits, so actual verses should be line_count * 2
        expected_verses = enriched_constraints.line_count * 2
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
        assert poem.constraints == enriched_constraints.to_dict()
        
        print(f"Generated poem:")
        print(f"  LLM Provider: {poem.llm_provider}")
        print(f"  Model: {poem.model_name}")
        print(f"  Verses:")
        for i, verse in enumerate(poem.verses, 1):
            print(f"    {i}. {verse}")
        
        # Step 3: Apply tashkeel (diacritics)
        print(f"\nStep 4: Applying tashkeel...")
        poem_with_tashkeel = await tashkeel_refiner.refine(poem,constraints, None)
        print(f"Applied tashkeel to {len(poem_with_tashkeel.verses)} verses")
        
        # Step 4: Evaluate poem using PoemEvaluator
        print(f"\nStep 5: Evaluating poem...")
        
        evaluated_poem = poem_evaluator.evaluate_poem(
            poem_with_tashkeel, 
            enriched_constraints, 
            [EvaluationType.LINE_COUNT, EvaluationType.PROSODY, EvaluationType.QAFIYA, EvaluationType.TASHKEEL]
        )
        
        # Verify evaluation results
        assert evaluated_poem.quality is not None
        assert evaluated_poem.quality.prosody_validation is not None
        assert evaluated_poem.quality.prosody_validation.bahr_used == enriched_constraints.meter
        # Temporarily comment out qafiya validation assertion due to mock response issues
        assert evaluated_poem.quality.qafiya_validation is not None
        
        print(f"Evaluation results:")
        print(f"  Overall quality score: {evaluated_poem.quality.overall_score}")
        print(f"  Is acceptable: {evaluated_poem.quality.is_acceptable}")
        print(f"  Prosody validation:")
        print(f"    Bahr used: {evaluated_poem.quality.prosody_validation.bahr_used}")
        print(f"    Total baits: {evaluated_poem.quality.prosody_validation.total_baits}")
        print(f"    Valid baits: {evaluated_poem.quality.prosody_validation.valid_baits}")
        print(f"    Invalid baits: {evaluated_poem.quality.prosody_validation.invalid_baits}")
        print(f"    Overall valid: {evaluated_poem.quality.prosody_validation.overall_valid}")
        print(f"  Qafiya validation:")
        if evaluated_poem.quality.qafiya_validation is not None:
            print(f"    Overall valid: {evaluated_poem.quality.qafiya_validation.overall_valid}")
            print(f"    Total baits: {evaluated_poem.quality.qafiya_validation.total_baits}")
            print(f"    Valid baits: {evaluated_poem.quality.qafiya_validation.valid_baits}")
            print(f"    Invalid baits: {evaluated_poem.quality.qafiya_validation.invalid_baits}")
        else:
            print(f"    Qafiya validation failed due to mock response issues")
        
        print(f"  Tashkeel validation:")
        if evaluated_poem.quality.tashkeel_validation is not None:
            print(f"    Overall valid: {evaluated_poem.quality.tashkeel_validation.overall_valid}")
            print(f"    Total baits: {evaluated_poem.quality.tashkeel_validation.total_baits}")
            print(f"    Valid baits: {evaluated_poem.quality.tashkeel_validation.valid_baits}")
            print(f"    Invalid baits: {evaluated_poem.quality.tashkeel_validation.invalid_baits}")
        else:
            print(f"    Tashkeel validation failed due to mock response issues")
        
        if evaluated_poem.quality.prosody_issues:
            print(f"  Prosody issues: {evaluated_poem.quality.prosody_issues}")
        if evaluated_poem.quality.qafiya_issues:
            print(f"  Qafiya issues: {evaluated_poem.quality.qafiya_issues}")
        if evaluated_poem.quality.tashkeel_issues:
            print(f"  Tashkeel issues: {evaluated_poem.quality.tashkeel_issues}")
        if evaluated_poem.quality.line_count_issues:
            print(f"  Line count issues: {evaluated_poem.quality.line_count_issues}")
        if evaluated_poem.quality.recommendations:
            print(f"  Recommendations: {evaluated_poem.quality.recommendations}")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert enriched_constraints.meter == expected_constraints["meter"]
            assert enriched_constraints.theme == expected_constraints["theme"]
            assert enriched_constraints.line_count == expected_constraints["line_count"]
            assert evaluated_poem.quality.tashkeel_validation is not None
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    @pytest.mark.asyncio
    async def test_complete_workflow_example_2(self, llm_type, mock_llm, real_llm, prompt_manager, test_data):
        """Test complete workflow for example 2 (هجاء poem) - generate and evaluate with real/mock LLM"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[1]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Create components
        constraint_parser, qafiya_selector, poem_generator, tashkeel_refiner, poem_evaluator = self._create_components(llm_type, mock_llm, real_llm, prompt_manager)
        
        # Set up mock responses for consistent testing when using mock LLM
        if llm_type == "mock":
            # Mock response for qafiya selection
            mock_qafiya_selection_response = '''
            ```json
            {
                "qafiya_letter": "ع",
                "qafiya_harakah": "مكسور",
                "qafiya_type": "مترادف",
                "qafiya_pattern": "عِ"
            }
            ```
            '''
            
            # Mock response for poem generation (12 verses for example 2 - 6 baits)
            mock_poem_response = '''
            ```json
            {
                "verses": [
                    "تَمَكَّنَ هَذا الدَهرُ مِمّا يَسوءُني",
                    "وَلَجَّ فَما يَخلي صَفاتِيَ مِن قَرعِ",
                    "وَأَبلَيتُ آمالي بِوَصلٍ يَكُدُّها",
                    "وَلَيسَ بِذي ضَرٍّ وَلَيْسَ بِذي نَفعِ",
                    "لَئيمٌ إِذا جادَ اللَئيمُ تَخَلُّقاً",
                    "يُحِبُّ سُؤالَ القَومِ شَوقاً إِلى المَنعِ",
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
            
            # Mock response for tashkeel (diacritics)
            mock_tashkeel_response = '''
            ```json
            {
                "diacritized_verses": [
                    "تَمَكَّنَ هَذا الدَّهْرُ مِمَّا يَسُوءُني",
                    "وَلَجَّ فَما يَخْلي صَفاتِيَ مِن قَرْعِ",
                    "وَأَبْلَيْتُ آمالي بِوَصْلٍ يَكُدُّها",
                    "وَلَيْسَ بِذي ضَرٍّ وَلَيْسَ بِذي نَفْعِ",
                    "لَئيمٌ إِذا جادَ اللَّئيمُ تَخَلُّقاً",
                    "يُحِبُّ سُؤالَ القَوْمِ شَوْقاً إِلى المَنْعِ",
                    "وَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفْعِ",
                    "وَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفعِ",
                    "وَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفعِ",
                    "وَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفعِ",
                    "وَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفعِ",
                    "وَأَصْبَحْتُ أَشْكو ما أَلاقي مِن دَفعِ"
                ]
            }
            ```
            '''
            
            
            # Mock response for qafiya validation (individual bait responses)
            mock_qafiya_validation_1 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            mock_qafiya_validation_2 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            mock_qafiya_validation_3 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            mock_qafiya_validation_4 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            mock_qafiya_validation_5 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            mock_qafiya_validation_6 = '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
            
            # Set up mock responses
            mock_llm.responses = [
                mock_qafiya_selection_response,  # Qafiya selection
                mock_poem_response,              # Poem generation
                mock_tashkeel_response,          # Tashkeel
                mock_qafiya_validation_1,        # Qafiya validation bait 1
                mock_qafiya_validation_2,        # Qafiya validation bait 2
                mock_qafiya_validation_3,        # Qafiya validation bait 3
                mock_qafiya_validation_4,        # Qafiya validation bait 4
                mock_qafiya_validation_5,        # Qafiya validation bait 5
                mock_qafiya_validation_6         # Qafiya validation bait 6
            ]
            mock_llm.reset()
        
        # Create constraints from example
        constraints = Constraints(
            meter=expected_constraints["meter"],
            qafiya=expected_constraints["qafiya"],
            line_count=expected_constraints["line_count"],
            theme=expected_constraints["theme"],
            tone=expected_constraints["tone"]
        )
        
        print(f"\nExample 2 - LLM Type: {llm_type}")
        print(f"User Prompt: {user_prompt}")
        print(f"Constraints: {constraints}")
        
        # Step 1: Select qafiya (directly with Constraints)
        print(f"\nStep 1: Selecting qafiya...")
        enriched_constraints = qafiya_selector.select_qafiya(constraints, user_prompt)
        print(f"Enriched constraints: {enriched_constraints}")
        
        # Step 2: Generate poem
        print(f"\nStep 2: Generating poem...")
        
        poem = poem_generator.generate_poem(enriched_constraints)
        
        # Verify qafiya selection and poem generation
        assert enriched_constraints is not None
        assert isinstance(enriched_constraints, Constraints)
        assert enriched_constraints.qafiya is not None
        assert enriched_constraints.qafiya_harakah is not None
        assert enriched_constraints.qafiya_type is not None
        
        assert isinstance(poem, LLMPoem)
        # line_count represents baits, so actual verses should be line_count * 2
        expected_verses = enriched_constraints.line_count * 2
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
        assert poem.constraints == enriched_constraints.to_dict()
        
        print(f"Generated poem:")
        print(f"  LLM Provider: {poem.llm_provider}")
        print(f"  Model: {poem.model_name}")
        print(f"  Verses:")
        for i, verse in enumerate(poem.verses, 1):
            print(f"    {i}. {verse}")
        
        # Step 3: Apply tashkeel (diacritics)
        print(f"\nStep 3: Applying tashkeel...")
        poem_with_tashkeel = await tashkeel_refiner.refine(poem,constraints, None)
        print(f"Applied tashkeel to {len(poem_with_tashkeel.verses)} verses")
        
        # Step 4: Evaluate poem using PoemEvaluator
        print(f"\nStep 4: Evaluating poem...")
        
        evaluated_poem = poem_evaluator.evaluate_poem(
            poem_with_tashkeel, 
            enriched_constraints, 
            [EvaluationType.LINE_COUNT, EvaluationType.PROSODY, EvaluationType.QAFIYA, EvaluationType.TASHKEEL]
        )
        
        # Verify evaluation results
        assert evaluated_poem.quality is not None
        assert evaluated_poem.quality.prosody_validation is not None
        assert evaluated_poem.quality.prosody_validation.bahr_used == enriched_constraints.meter
        # Temporarily comment out qafiya validation assertion due to mock response issues
        assert evaluated_poem.quality.qafiya_validation is not None
        
        print(f"Evaluation results:")
        print(f"  Overall quality score: {evaluated_poem.quality.overall_score}")
        print(f"  Is acceptable: {evaluated_poem.quality.is_acceptable}")
        print(f"  Prosody validation:")
        print(f"    Bahr used: {evaluated_poem.quality.prosody_validation.bahr_used}")
        print(f"    Total baits: {evaluated_poem.quality.prosody_validation.total_baits}")
        print(f"    Valid baits: {evaluated_poem.quality.prosody_validation.valid_baits}")
        print(f"    Invalid baits: {evaluated_poem.quality.prosody_validation.invalid_baits}")
        print(f"    Overall valid: {evaluated_poem.quality.prosody_validation.overall_valid}")
        print(f"  Qafiya validation:")
        if evaluated_poem.quality.qafiya_validation is not None:
            print(f"    Overall valid: {evaluated_poem.quality.qafiya_validation.overall_valid}")
            print(f"    Total baits: {evaluated_poem.quality.qafiya_validation.total_baits}")
            print(f"    Valid baits: {evaluated_poem.quality.qafiya_validation.valid_baits}")
            print(f"    Invalid baits: {evaluated_poem.quality.qafiya_validation.invalid_baits}")
        else:
            print(f"    Qafiya validation failed due to mock response issues")
        
        print(f"  Tashkeel validation:")
        if evaluated_poem.quality.tashkeel_validation is not None:
            print(f"    Overall valid: {evaluated_poem.quality.tashkeel_validation.overall_valid}")
            print(f"    Total baits: {evaluated_poem.quality.tashkeel_validation.total_baits}")
            print(f"    Valid baits: {evaluated_poem.quality.tashkeel_validation.valid_baits}")
            print(f"    Invalid baits: {evaluated_poem.quality.tashkeel_validation.invalid_baits}")
        else:
            print(f"    Tashkeel validation failed due to mock response issues")
        
        if evaluated_poem.quality.prosody_issues:
            print(f"  Prosody issues: {evaluated_poem.quality.prosody_issues}")
        if evaluated_poem.quality.qafiya_issues:
            print(f"  Qafiya issues: {evaluated_poem.quality.qafiya_issues}")
        if evaluated_poem.quality.tashkeel_issues:
            print(f"  Tashkeel issues: {evaluated_poem.quality.tashkeel_issues}")
        if evaluated_poem.quality.line_count_issues:
            print(f"  Line count issues: {evaluated_poem.quality.line_count_issues}")
        if evaluated_poem.quality.recommendations:
            print(f"  Recommendations: {evaluated_poem.quality.recommendations}")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert enriched_constraints.meter == expected_constraints["meter"]
            assert enriched_constraints.theme == expected_constraints["theme"]
            assert enriched_constraints.line_count == expected_constraints["line_count"]
            assert evaluated_poem.quality.tashkeel_validation is not None
        
        # Additional analysis for real LLM tests
        if llm_type == "real":
            print(f"\nReal LLM Analysis:")
            print(f"  Poem follows meter: {enriched_constraints.meter}")
            print(f"  Poem follows theme: {enriched_constraints.theme}")
            print(f"  Poem follows qafiya: {enriched_constraints.qafiya}")
            print(f"  Prosody compliance: {evaluated_poem.quality.prosody_validation.overall_valid}")
            print(f"  Qafiya compliance: {evaluated_poem.quality.qafiya_validation.overall_valid}")
            print(f"  Tashkeel compliance: {evaluated_poem.quality.tashkeel_validation.overall_valid}")
            print(f"  Line count compliance: {evaluated_poem.quality.line_count_validation.is_valid}")
            print(f"  Quality score: {evaluated_poem.quality.overall_score}")
            
            # Provide insights on generation quality
            if evaluated_poem.quality.prosody_validation.overall_valid:
                print(f"  [SUCCESS] Generated poem successfully follows {enriched_constraints.meter} meter")
            else:
                print(f"  [WARNING] Generated poem has prosody issues with {enriched_constraints.meter} meter")
            
            if evaluated_poem.quality.qafiya_validation.overall_valid:
                print(f"  [SUCCESS] Generated poem successfully follows qafiya {enriched_constraints.qafiya}")
            else:
                print(f"  [WARNING] Generated poem has qafiya issues with {enriched_constraints.qafiya}")
            
            if evaluated_poem.quality.tashkeel_validation.overall_valid:
                print(f"  [SUCCESS] Generated poem has correct diacritics")
            else:
                print(f"  [WARNING] Generated poem has diacritic issues")
            
            if evaluated_poem.quality.overall_score > 0.8:
                print(f"  [SUCCESS] High quality poem generated")
            elif evaluated_poem.quality.overall_score > 0.6:
                print(f"  [WARNING] Moderate quality poem generated")
            else:
                print(f"  [ERROR] Low quality poem generated") 
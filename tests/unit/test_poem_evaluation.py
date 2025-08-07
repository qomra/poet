import pytest
from unittest.mock import Mock, patch
from poet.evaluation.poem import PoemEvaluator, EvaluationType
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.llm.base_llm import MockLLM, LLMConfig


class TestPoemEvaluator:
    """Unit tests for PoemEvaluator"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM"""
        config = LLMConfig(model_name="test-model")
        return MockLLM(config)
    
    @pytest.fixture
    def poem_evaluator(self, mock_llm):
        """Create PoemEvaluator instance"""
        return PoemEvaluator(mock_llm)
    
    @pytest.fixture
    def sample_poem(self):
        """Create a sample poem"""
        return LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا",
                "لِمَا نَسَجَتْهَا مِنْ جَنُوبٍ وَشَمْأَلِ"
            ],
            llm_provider="mock",
            model_name="test-model"
        )
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints"""
        return Constraints(
            meter="طويل",
            qafiya="ل",
            line_count=2,
            theme="غزل",
            tone="رومانسي"
        )
    
    def test_poem_evaluator_initialization(self, mock_llm):
        """Test PoemEvaluator initialization"""
        evaluator = PoemEvaluator(mock_llm)
        
        assert evaluator.llm == mock_llm
        assert evaluator.line_count_validator is not None
        assert evaluator.prosody_validator is not None
        assert evaluator.qafiya_validator is not None
    
    def test_evaluate_poem_line_count_only(self, poem_evaluator, sample_poem, sample_constraints):
        """Test evaluation with only line count validation"""
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.LINE_COUNT]
        )
        
        assert result == sample_poem
        assert result.quality is not None
        assert result.quality.line_count_issues == []
        assert result.quality.prosody_issues == []
        assert result.quality.qafiya_issues == []
        assert result.quality.overall_score == 1.0
        assert result.quality.is_acceptable is True
    
    def test_evaluate_poem_with_invalid_line_count(self, poem_evaluator, sample_constraints):
        """Test evaluation with invalid line count"""
        invalid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا"
            ],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = poem_evaluator.evaluate_poem(
            invalid_poem, 
            sample_constraints, 
            [EvaluationType.LINE_COUNT]
        )
        
        assert result.quality is not None
        assert len(result.quality.line_count_issues) > 0
        assert result.quality.overall_score < 1.0
        assert result.quality.is_acceptable is False
    
    def test_evaluate_poem_with_prosody_only(self, poem_evaluator, sample_poem, sample_constraints):
        """Test evaluation with prosody validation only"""
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.PROSODY]
        )
        
        assert result == sample_poem
        assert result.quality is not None
        # Should have prosody validation results
        assert result.prosody_validation is not None
        assert result.prosody_validation.bahr_used == "طويل"
    
    def test_evaluate_poem_with_prosody(self, poem_evaluator, sample_poem, sample_constraints):
        """Test evaluation with prosody validation"""
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.PROSODY]
        )
        
        assert result == sample_poem
        assert result.quality is not None
        # Should have prosody validation results (even if there are issues)
        assert result.prosody_validation is not None
        assert result.prosody_validation.bahr_used == "طويل"
    
    def test_evaluate_poem_with_qafiya(self, poem_evaluator, sample_poem, sample_constraints, mock_llm):
        """Test evaluation with qafiya validation"""
        # Set up mock qafiya response for individual bait validation
        mock_llm.responses = ['''
        ```json
        {
            "is_valid": true,
            "issue": null
        }
        ```
        ''']
        mock_llm.reset()
        
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.QAFIYA]
        )
        
        assert result == sample_poem
        assert result.quality is not None
        # Should have qafiya validation results
        assert result.quality.qafiya_issues == []
        assert result.quality.overall_score == 1.0
    
    def test_evaluate_poem_complete_workflow(self, poem_evaluator, sample_poem, sample_constraints, mock_llm):
        """Test complete evaluation workflow"""
        # Set up mock response for qafiya (individual bait validation)
        mock_llm.responses = [
            # Qafiya response for single bait
            '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
        ]
        mock_llm.reset()
        
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.LINE_COUNT, EvaluationType.PROSODY, EvaluationType.QAFIYA]
        )
        
        assert result == sample_poem
        assert result.quality is not None
        # Should have validation results
        assert result.prosody_validation is not None
        assert result.prosody_validation.bahr_used == "طويل"
    
    def test_evaluate_poem_with_errors(self, poem_evaluator, sample_poem, sample_constraints):
        """Test evaluation with validation errors"""
        # Create constraints with unknown meter to trigger prosody error
        bad_constraints = Constraints(
            meter="unknown_meter",
            qafiya="ل",
            line_count=2,
            theme="غزل",
            tone="رومانسي"
        )
        
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            bad_constraints, 
            [EvaluationType.PROSODY]
        )
        
        assert result.quality is not None
        # Should have prosody validation result even for unknown meter
        assert result.prosody_validation is not None
        assert result.prosody_validation.bahr_used == "unknown_meter"
        assert result.prosody_validation.overall_valid is False
    
    def test_evaluate_poem_error_handling(self, poem_evaluator, sample_poem, sample_constraints):
        """Test error handling when validators fail"""
        # Create a failing LLM
        failing_llm = Mock()
        failing_llm.generate.side_effect = Exception("LLM error")
        failing_llm.config = LLMConfig(model_name="test-model")
        
        failing_evaluator = PoemEvaluator(failing_llm)
        
        result = failing_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.QAFIYA]  # This will fail due to LLM error
        )
        
        assert result == sample_poem
        assert result.quality is not None
        # Should have error in qafiya issues
        assert len(result.quality.qafiya_issues) > 0
    
    def test_evaluate_poem_quality_calculation(self, poem_evaluator, sample_poem, sample_constraints):
        """Test quality score calculation with multiple issues"""
        # Create poem with odd line count to trigger line count issue
        invalid_poem = LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                "فَتُوْضِحَ فَالْمِقْرَاةِ لَمْ يَعْفُ رَسْمُهَا"
            ],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = poem_evaluator.evaluate_poem(
            invalid_poem, 
            sample_constraints, 
            [EvaluationType.LINE_COUNT, EvaluationType.PROSODY]
        )
        
        assert result.quality is not None
        assert len(result.quality.line_count_issues) > 0
        assert result.quality.overall_score < 1.0
        assert result.quality.is_acceptable is False
        assert "تأكد من أن عدد الأبيات زوجي" in result.quality.recommendations
    
    def test_evaluation_type_enum(self):
        """Test EvaluationType enum values"""
        assert EvaluationType.LINE_COUNT.value == "line_count"
        assert EvaluationType.PROSODY.value == "prosody"
        assert EvaluationType.QAFIYA.value == "qafiya"
        assert EvaluationType.TASHKEEL.value == "tashkeel"
    
    def test_evaluate_poem_with_tashkeel_only(self, poem_evaluator, sample_poem, sample_constraints):
        """Test evaluation with tashkeel validation only"""
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.TASHKEEL]
        )
        
        assert result == sample_poem
        assert result.quality is not None
        # Should have tashkeel validation results
        assert result.quality.tashkeel_validation is not None
        assert result.quality.tashkeel_issues == []
        assert result.quality.overall_score == 1.0
    
    def test_evaluate_poem_with_tashkeel_issues(self, poem_evaluator, sample_poem, sample_constraints):
        """Test evaluation with tashkeel validation issues"""
        # Create a poem with tashkeel issues (missing diacritics)
        poem_with_tashkeel_issues = LLMPoem(
            verses=[
                "قفا نبك من ذكرى حبيب ومنزل",  # Missing diacritics
                "بسقط اللوى بين الدخول فحومل",
                "فتوضح فالمقراة لم يعف رسمها",
                "لما نسجتها من جنوب وشمأل"
            ],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = poem_evaluator.evaluate_poem(
            poem_with_tashkeel_issues, 
            sample_constraints, 
            [EvaluationType.TASHKEEL]
        )
        
        assert result.quality is not None
        assert result.quality.tashkeel_validation is not None
        # Should have tashkeel issues
        assert len(result.quality.tashkeel_issues) > 0
        assert result.quality.overall_score < 1.0
    
    def test_evaluate_poem_complete_workflow_with_tashkeel(self, poem_evaluator, sample_poem, sample_constraints, mock_llm):
        """Test complete evaluation workflow including tashkeel"""
        # Set up mock response for qafiya (individual bait validation)
        mock_llm.responses = [
            # Qafiya response for single bait
            '''
            ```json
            {
                "is_valid": true,
                "issue": null
            }
            ```
            '''
        ]
        mock_llm.reset()
        
        result = poem_evaluator.evaluate_poem(
            sample_poem, 
            sample_constraints, 
            [EvaluationType.LINE_COUNT, EvaluationType.PROSODY, EvaluationType.QAFIYA, EvaluationType.TASHKEEL]
        )
        
        assert result == sample_poem
        assert result.quality is not None
        # Should have all validation results
        assert result.prosody_validation is not None
        assert result.prosody_validation.bahr_used == "طويل"
        assert result.quality.tashkeel_validation is not None
        assert result.quality.tashkeel_issues == []
    
    def test_evaluate_poem_quality_calculation_with_tashkeel(self, poem_evaluator, sample_constraints):
        """Test quality score calculation with tashkeel issues"""
        # Create poem with tashkeel issues
        poem_with_issues = LLMPoem(
            verses=[
                "قفا نبك من ذكرى حبيب ومنزل",  # Missing diacritics
                "بسقط اللوى بين الدخول فحومل",
                "فتوضح فالمقراة لم يعف رسمها",
                "لما نسجتها من جنوب وشمأل"
            ],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = poem_evaluator.evaluate_poem(
            poem_with_issues, 
            sample_constraints, 
            [EvaluationType.TASHKEEL]
        )
        
        assert result.quality is not None
        assert len(result.quality.tashkeel_issues) > 0
        assert result.quality.overall_score < 1.0
        assert "راجع التشكيل في الأبيات" in result.quality.recommendations
    
    def test_evaluate_poem_multiple_issues_including_tashkeel(self, poem_evaluator, sample_constraints):
        """Test evaluation with multiple issues including tashkeel"""
        # Create poem with multiple issues
        problematic_poem = LLMPoem(
            verses=[
                "قفا نبك من ذكرى حبيب ومنزل",  # Missing diacritics
                "بسقط اللوى بين الدخول فحومل",
                "فتوضح فالمقراة لم يعف رسمها"  # Odd number of verses
            ],
            llm_provider="mock",
            model_name="test-model"
        )
        
        result = poem_evaluator.evaluate_poem(
            problematic_poem, 
            sample_constraints, 
            [EvaluationType.LINE_COUNT, EvaluationType.TASHKEEL]
        )
        
        assert result.quality is not None
        assert len(result.quality.line_count_issues) > 0
        assert len(result.quality.tashkeel_issues) > 0
        assert result.quality.overall_score < 1.0
        assert result.quality.is_acceptable is False
        assert "تأكد من أن عدد الأبيات زوجي" in result.quality.recommendations
        assert "راجع التشكيل في الأبيات" in result.quality.recommendations 
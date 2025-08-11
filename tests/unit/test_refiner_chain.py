# tests/unit/test_refiner_chain.py

import pytest
from unittest.mock import Mock, AsyncMock
from poet.refinement.refiner_chain import RefinerChain
from poet.refinement.base import BaseRefiner, RefinementStep
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.evaluation.poem import PoemEvaluator, EvaluationType
from poet.llm.base_llm import MockLLM, LLMConfig


class MockRefiner(BaseRefiner):
    """Mock refiner for testing"""
    
    def __init__(self, name, should_refine_result=True, refine_result=None):
        self._name = name
        self.should_refine_result = should_refine_result
        self.refine_result = refine_result or Mock(spec=LLMPoem)
        self.refine_called = False
    
    @property
    def name(self) -> str:
        return self._name
    
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        return self.should_refine_result
    
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        self.refine_called = True
        if self.refine_result is None:
            # Return a modified version of the input poem
            return LLMPoem(
                verses=poem.verses.copy(),
                llm_provider=poem.llm_provider,
                model_name=poem.model_name,
                constraints=poem.constraints,
                generation_timestamp=poem.generation_timestamp,
                quality=poem.quality
            )
        return self.refine_result


class TestRefinerChain:
    """Test RefinerChain functionality"""
    
    @pytest.fixture
    def mock_poem(self):
        """Create mock poem"""
        return Mock(spec=LLMPoem)
    
    @pytest.fixture
    def mock_constraints(self):
        """Create mock constraints"""
        return Mock(spec=Constraints)
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM"""
        config = LLMConfig(model_name="test-model")
        return MockLLM(config)
    
    @pytest.fixture
    def mock_evaluator(self):
        """Create mock evaluator"""
        evaluator = Mock(spec=PoemEvaluator)
        evaluator.evaluate_poem = Mock()  # Not async anymore
        return evaluator
    
    @pytest.fixture
    def sample_poem(self):
        """Create sample poem"""
        return LLMPoem(
            verses=["بيت أول", "بيت ثاني"],
            llm_provider="test_provider",
            model_name="test_model"
        )
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints"""
        return Constraints(
            meter="بحر الطويل",
            qafiya="ق"
        )
    
    def test_refiner_chain_initialization(self, mock_llm):
        """Test refiner chain initialization"""
        refiners = [MockRefiner("refiner1"), MockRefiner("refiner2")]
        chain = RefinerChain(mock_llm, refiners, max_iterations=5)
        
        assert len(chain.refiners) == 2
        assert chain.max_iterations == 5
        assert isinstance(chain.evaluator, PoemEvaluator)
    
    @pytest.mark.asyncio
    async def test_refine_no_refiners_needed(self, sample_poem, sample_constraints, mock_evaluator, mock_llm):
        """Test refine when no refiners are needed"""
        # Create refiners that don't need refinement
        refiners = [
            MockRefiner("refiner1", should_refine_result=False),
            MockRefiner("refiner2", should_refine_result=False)
        ]
        
        # Mock evaluator to return high quality score
        mock_poem = LLMPoem(
            verses=sample_poem.verses,
            llm_provider=sample_poem.llm_provider,
            model_name=sample_poem.model_name
        )
        mock_poem.quality = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=0.9,
            is_acceptable=True,
            recommendations=[]
        )
        mock_evaluator.evaluate_poem.return_value = mock_poem
        
        chain = RefinerChain(mock_llm, refiners, max_iterations=3)
        chain.evaluator = mock_evaluator
        
        result_poem, history = await chain.refine(sample_poem, sample_constraints, target_quality=0.8)
        
        # Should return original poem
        assert result_poem == sample_poem
        assert len(history) == 0
        
        # Evaluator should be called once
        mock_evaluator.evaluate_poem.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_refine_quality_target_reached(self, sample_poem, sample_constraints, mock_evaluator, mock_llm):
        """Test refine when quality target is reached after first iteration"""
        # Create refiners that need refinement
        refiners = [
            MockRefiner("refiner1", should_refine_result=True),
            MockRefiner("refiner2", should_refine_result=True)
        ]
        
        # Mock evaluator to return improving quality scores
        mock_poem1 = LLMPoem(verses=sample_poem.verses, llm_provider=sample_poem.llm_provider, model_name=sample_poem.model_name)
        mock_poem1.quality = QualityAssessment(prosody_issues=[], line_count_issues=[], qafiya_issues=[], tashkeel_issues=[], overall_score=0.5, is_acceptable=False, recommendations=[])
        
        mock_poem2 = LLMPoem(verses=sample_poem.verses, llm_provider=sample_poem.llm_provider, model_name=sample_poem.model_name)
        mock_poem2.quality = QualityAssessment(prosody_issues=[], line_count_issues=[], qafiya_issues=[], tashkeel_issues=[], overall_score=0.9, is_acceptable=True, recommendations=[])
        
        mock_poem3 = LLMPoem(verses=sample_poem.verses, llm_provider=sample_poem.llm_provider, model_name=sample_poem.model_name)
        mock_poem3.quality = QualityAssessment(prosody_issues=[], line_count_issues=[], qafiya_issues=[], tashkeel_issues=[], overall_score=0.9, is_acceptable=True, recommendations=[])
        
        mock_evaluator.evaluate_poem.side_effect = [mock_poem1, mock_poem2, mock_poem3]
        
        chain = RefinerChain(mock_llm, refiners, max_iterations=3)
        chain.evaluator = mock_evaluator
        
        result_poem, history = await chain.refine(sample_poem, sample_constraints, target_quality=0.8)
        
        # Should have applied refiners and stopped when target reached
        assert len(history) > 0
        assert mock_evaluator.evaluate_poem.call_count == 3  # Initial + 1 iteration + final evaluation
        
        # Check that refiners were called
        assert refiners[0].refine_called
        assert refiners[1].refine_called
    
    @pytest.mark.asyncio
    async def test_refine_max_iterations_reached(self, sample_poem, sample_constraints, mock_evaluator, mock_llm):
        """Test refine when max iterations are reached"""
        # Create refiners that always need refinement
        refiners = [
            MockRefiner("refiner1", should_refine_result=True),
            MockRefiner("refiner2", should_refine_result=True)
        ]
        
        # Mock evaluator to always return low quality
        mock_poem = LLMPoem(verses=sample_poem.verses, llm_provider=sample_poem.llm_provider, model_name=sample_poem.model_name)
        mock_poem.quality = QualityAssessment(prosody_issues=[], line_count_issues=[], qafiya_issues=[], tashkeel_issues=[], overall_score=0.3, is_acceptable=False, recommendations=[])
        mock_evaluator.evaluate_poem.return_value = mock_poem
        
        chain = RefinerChain(mock_llm, refiners, max_iterations=2)
        chain.evaluator = mock_evaluator
        
        result_poem, history = await chain.refine(sample_poem, sample_constraints, target_quality=0.8)
        
        # Should have applied refiners for max iterations
        assert len(history) > 0
        assert mock_evaluator.evaluate_poem.call_count == 3  # Initial + 2 iterations
        
        # Check that refiners were called multiple times
        assert refiners[0].refine_called
        assert refiners[1].refine_called
    
    @pytest.mark.asyncio
    async def test_refine_no_refiners_applied(self, sample_poem, sample_constraints, mock_evaluator, mock_llm):
        """Test refine when no refiners are applied in an iteration"""
        # Create refiners that don't need refinement
        refiners = [
            MockRefiner("refiner1", should_refine_result=False),
            MockRefiner("refiner2", should_refine_result=False)
        ]
        
        # Mock evaluator to return low quality
        mock_poem = LLMPoem(verses=sample_poem.verses, llm_provider=sample_poem.llm_provider, model_name=sample_poem.model_name)
        mock_poem.quality = QualityAssessment(prosody_issues=[], line_count_issues=[], qafiya_issues=[], tashkeel_issues=[], overall_score=0.3, is_acceptable=False, recommendations=[])
        mock_evaluator.evaluate_poem.return_value = mock_poem
        
        chain = RefinerChain(mock_llm, refiners, max_iterations=3)
        chain.evaluator = mock_evaluator
        
        result_poem, history = await chain.refine(sample_poem, sample_constraints, target_quality=0.8)
        
        # Should return original poem and stop
        assert result_poem == sample_poem
        assert len(history) == 0
        assert mock_evaluator.evaluate_poem.call_count == 1
    
    def test_calculate_quality_score_perfect(self, mock_llm):
        """Test quality score calculation for perfect poem"""
        chain = RefinerChain([], mock_llm)
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[]
        )
        
        score = chain._calculate_quality_score(evaluation)
        assert score == 1.0
    
    def test_calculate_quality_score_with_line_count_issues(self, mock_llm):
        """Test quality score calculation with line count issues"""
        chain = RefinerChain([], mock_llm)
        
        from poet.models.line_count import LineCountValidationResult
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=["Wrong line count"],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=0.7,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=LineCountValidationResult(
                is_valid=False,
                line_count=3,
                expected_even=True,
                validation_summary="Wrong line count"
            )
        )
        
        score = chain._calculate_quality_score(evaluation)
        assert score == 0.7  # 1.0 - 0.3
    
    def test_calculate_quality_score_with_prosody_issues(self, mock_llm):
        """Test quality score calculation with prosody issues"""
        chain = RefinerChain([], mock_llm)
        
        from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
        
        # Create validation with broken verses
        bait_result = BaitValidationResult(
            bait_text="test bait",
            is_valid=False,
            pattern="test pattern",
            error_details="Broken verse"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=["Broken verse"],
            line_count_issues=[],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            prosody_validation=ProsodyValidationResult(
                overall_valid=False,
                total_baits=1,
                valid_baits=0,
                invalid_baits=1,
                bait_results=[bait_result],
                bahr_used="طويل",
                validation_summary="Broken verse"
            )
        )
        
        # Mock poem for verse count
        evaluation.poem = Mock()
        evaluation.poem.verses = ["verse1", "verse2", "verse3", "verse4"]
        
        score = chain._calculate_quality_score(evaluation)
        assert score < 1.0  # Should be penalized
        assert score >= 0.6  # But not too much for one broken verse
    
    def test_calculate_quality_score_with_qafiya_issues(self, mock_llm):
        """Test quality score calculation with qafiya issues"""
        chain = RefinerChain([], mock_llm)
        
        from poet.models.qafiya import QafiyaValidationResult, QafiyaBaitResult
        
        # Create validation with wrong qafiya verses
        bait_result = QafiyaBaitResult(
            bait_number=0,
            is_valid=False,
            error_details="Wrong qafiya"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=["Wrong qafiya"],
            tashkeel_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            qafiya_validation=QafiyaValidationResult(
                overall_valid=False,
                total_baits=1,
                valid_baits=0,
                invalid_baits=1,
                bait_results=[bait_result],
                validation_summary="Wrong qafiya",
                misaligned_bait_numbers=[]
            )
        )
        
        # Mock poem for verse count
        evaluation.poem = Mock()
        evaluation.poem.verses = ["verse1", "verse2", "verse3", "verse4"]
        
        score = chain._calculate_quality_score(evaluation)
        assert score < 1.0  # Should be penalized
        assert score >= 0.7  # But not too much for one wrong qafiya
    
    def test_calculate_quality_score_with_tashkeel_issues(self, mock_llm):
        """Test quality score calculation with tashkeel issues"""
        chain = RefinerChain([], mock_llm)
        
        from poet.models.tashkeel import TashkeelValidationResult, TashkeelBaitResult
        
        # Create validation with tashkeel issues
        bait_result = TashkeelBaitResult(
            bait_number=0,
            is_valid=False,
            error_details="Missing diacritics"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            tashkeel_issues=["Missing diacritics"],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            tashkeel_validation=TashkeelValidationResult(
                overall_valid=False,
                total_baits=1,
                valid_baits=0,
                invalid_baits=1,
                bait_results=[bait_result],
                validation_summary="Missing diacritics"
            )
        )
        
        # Mock poem for verse count
        evaluation.poem = Mock()
        evaluation.poem.verses = ["verse1", "verse2", "verse3", "verse4"]
        
        score = chain._calculate_quality_score(evaluation)
        assert score < 1.0  # Should be penalized
        assert score >= 0.8  # But not too much for tashkeel issues
    
    def test_calculate_quality_score_minimum_zero(self, mock_llm):
        """Test quality score calculation returns minimum of 0.0"""
        chain = RefinerChain([], mock_llm)
        
        # Create evaluation with many issues
        from poet.models.line_count import LineCountValidationResult
        from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
        from poet.models.qafiya import QafiyaValidationResult, QafiyaBaitResult
        
        evaluation = QualityAssessment(
            prosody_issues=["Many issues"],
            line_count_issues=["Many issues"],
            qafiya_issues=["Many issues"],
            tashkeel_issues=["Many issues"],
            overall_score=0.1,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=LineCountValidationResult(
                is_valid=False,
                line_count=1,
                expected_even=True,
                validation_summary="Many issues"
            ),
            prosody_validation=ProsodyValidationResult(
                overall_valid=False,
                total_baits=1,
                valid_baits=0,
                invalid_baits=1,
                bait_results=[BaitValidationResult(
                    bait_text="test",
                    is_valid=False,
                    pattern="test",
                    error_details="Many issues"
                )],
                bahr_used="طويل",
                validation_summary="Many issues"
            ),
            qafiya_validation=QafiyaValidationResult(
                overall_valid=False,
                total_baits=1,
                valid_baits=0,
                invalid_baits=1,
                bait_results=[QafiyaBaitResult(
                    bait_number=0,
                    is_valid=False,
                    error_details="Many issues"
                )],
                validation_summary="Many issues",
                misaligned_bait_numbers=[]
            )
        )
        
        evaluation.poem = Mock()
        evaluation.poem.verses = ["verse1", "verse2"]
        
        score = chain._calculate_quality_score(evaluation)
        assert score >= 0.0  # Should not go below 0
    
    def test_get_refinement_summary_empty_history(self, mock_llm):
        """Test refinement summary with empty history"""
        chain = RefinerChain([], mock_llm)
        
        summary = chain.get_refinement_summary([])
        
        assert summary["total_steps"] == 0
        assert summary["refiners_used"] == []
        assert summary["quality_improvement"] == 0.0
        assert summary["iterations"] == 0
    
    def test_get_refinement_summary_with_history(self, mock_llm):
        """Test refinement summary with refinement history"""
        chain = RefinerChain([], mock_llm)
        
        # Create mock refinement steps
        step1 = RefinementStep(
            refiner_name="refiner1",
            iteration=0,
            before=Mock(spec=LLMPoem),
            after=Mock(spec=LLMPoem),
            quality_before=0.5,
            quality_after=0.7
        )
        
        step2 = RefinementStep(
            refiner_name="refiner2",
            iteration=1,
            before=Mock(spec=LLMPoem),
            after=Mock(spec=LLMPoem),
            quality_before=0.7,
            quality_after=0.9
        )
        
        history = [step1, step2]
        summary = chain.get_refinement_summary(history)
        
        assert summary["total_steps"] == 2
        assert set(summary["refiners_used"]) == {"refiner1", "refiner2"}
        assert summary["quality_improvement"] == 0.4  # 0.9 - 0.5
        assert summary["iterations"] == 2
    
    def test_get_refinement_summary_missing_quality_scores(self, mock_llm):
        """Test refinement summary when quality scores are missing"""
        chain = RefinerChain([], mock_llm)
        
        step = RefinementStep(
            refiner_name="refiner1",
            iteration=0,
            before=Mock(spec=LLMPoem),
            after=Mock(spec=LLMPoem)
            # Missing quality scores
        )
        
        history = [step]
        summary = chain.get_refinement_summary(history)
        
        assert summary["total_steps"] == 1
        assert summary["refiners_used"] == ["refiner1"]
        assert summary["quality_improvement"] == 0.0  # No scores to calculate from
        assert summary["iterations"] == 1 
# tests/integration/test_refiner_integration.py

import pytest
from unittest.mock import Mock, AsyncMock
from poet.refinement import (
    BaseRefiner, RefinementStep, LineCountRefiner, 
    ProsodyRefiner, QafiyaRefiner, RefinerChain
)
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.models.line_count import LineCountValidationResult
from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
from poet.models.qafiya import QafiyaValidationResult, QafiyaBaitResult
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class TestRefinerIntegration:
    """Integration tests for the refiner system"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM"""
        llm = Mock(spec=BaseLLM)
        llm.generate = Mock(return_value='{"verses": ["بيت مصحح"]}')
        return llm
    
    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager"""
        pm = Mock(spec=PromptManager)
        pm.format_prompt = Mock(return_value="formatted prompt")
        return pm
    
    @pytest.fixture
    def sample_poem(self):
        """Create sample poem with issues"""
        return LLMPoem(
            verses=["بيت أول", "بيت ثاني", "بيت ثالث"],  # Odd number - line count issue
            llm_provider="test_provider",
            model_name="test_model"
        )
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints"""
        return Constraints(
            line_count=4,  # Expecting 4 verses
            meter="بحر الطويل",
            qafiya="ق",
            qafiya_pattern="قَ"
        )
    
    @pytest.fixture
    def problematic_evaluation(self):
        """Create evaluation with multiple issues"""
        # Line count issue
        line_count_validation = LineCountValidationResult(
            is_valid=False,
            expected_count=4,
            actual_count=3,
            validation_summary="Wrong line count"
        )
        
        # Prosody issue
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            bait_results=[
                BaitValidationResult(
                    bait_index=0,
                    is_valid=False,
                    first_verse_valid=False,
                    second_verse_valid=True,
                    error_details="وزن خاطئ في البيت الأول"
                )
            ]
        )
        
        # Qafiya issue
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            bait_results=[
                BaitQafiyaResult(
                    bait_index=0,
                    is_valid=False,
                    first_verse_valid=False,
                    second_verse_valid=True,
                    expected_qafiya="قَ"
                )
            ]
        )
        
        return QualityAssessment(
            line_count_validation=line_count_validation,
            prosody_validation=prosody_validation,
            qafiya_validation=qafiya_validation,
            overall_score=0.3,
            is_acceptable=False
        )
    
    def test_refiner_chain_with_all_refiners(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints, problematic_evaluation):
        """Test refiner chain with all three refiners"""
        # Create all refiners
        line_count_refiner = LineCountRefiner(mock_llm, mock_prompt_manager)
        prosody_refiner = ProsodyRefiner(mock_llm, mock_prompt_manager)
        qafiya_refiner = QafiyaRefiner(mock_llm, mock_prompt_manager)
        
        # Test that all refiners identify issues
        assert line_count_refiner.should_refine(problematic_evaluation) is True
        assert prosody_refiner.should_refine(problematic_evaluation) is True
        assert qafiya_refiner.should_refine(problematic_evaluation) is True
        
        # Test refiner names
        assert line_count_refiner.name == "line_count_refiner"
        assert prosody_refiner.name == "prosody_refiner"
        assert qafiya_refiner.name == "qafiya_refiner"
    
    @pytest.mark.asyncio
    async def test_refiner_chain_sequential_execution(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test that refiners can be executed sequentially"""
        # Create refiners
        refiners = [
            LineCountRefiner(mock_llm, mock_prompt_manager),
            ProsodyRefiner(mock_llm, mock_prompt_manager),
            QafiyaRefiner(mock_llm, mock_prompt_manager)
        ]
        
        # Create chain
        chain = RefinerChain(refiners, max_iterations=2)
        
        # Mock evaluator to simulate improvement
        mock_evaluator = Mock()
        mock_evaluator.evaluate_poem = AsyncMock()
        mock_evaluator.evaluate_poem.side_effect = [
            # First evaluation - low quality
            QualityAssessment(
                line_count_validation=LineCountValidationResult(is_valid=False),
                prosody_validation=ProsodyValidationResult(overall_valid=False),
                qafiya_validation=QafiyaValidationResult(overall_valid=False),
                overall_score=0.3
            ),
            # Second evaluation - improved quality
            QualityAssessment(
                line_count_validation=LineCountValidationResult(is_valid=True),
                prosody_validation=ProsodyValidationResult(overall_valid=True),
                qafiya_validation=QafiyaValidationResult(overall_valid=True),
                overall_score=0.9
            )
        ]
        chain.evaluator = mock_evaluator
        
        # Execute refinement
        result_poem, history = await chain.refine(sample_poem, sample_constraints, target_quality=0.8)
        
        # Verify results
        assert result_poem is not None
        assert len(history) > 0
        assert mock_evaluator.evaluate_poem.call_count == 2
        
        # Verify refinement steps
        for step in history:
            assert isinstance(step, RefinementStep)
            assert step.refiner_name in ["line_count_refiner", "prosody_refiner", "qafiya_refiner"]
            assert step.iteration >= 0
            assert step.before is not None
            assert step.after is not None
    
    @pytest.mark.asyncio
    async def test_refiner_error_handling(self, mock_llm, sample_poem, sample_constraints):
        """Test that refiners handle errors gracefully"""
        # Create refiner with LLM that raises exceptions
        mock_llm.generate.side_effect = Exception("LLM error")
        refiner = LineCountRefiner(mock_llm)
        
        # Create evaluation with line count issue
        evaluation = QualityAssessment(
            line_count_validation=LineCountValidationResult(
                is_valid=False,
                expected_count=4,
                actual_count=3
            )
        )
        
        # Should not raise exception, should return original poem
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        assert result == sample_poem
    
    def test_refiner_chain_quality_calculation(self):
        """Test quality score calculation in refiner chain"""
        chain = RefinerChain([])
        
        # Test perfect poem
        perfect_evaluation = QualityAssessment(
            line_count_validation=LineCountValidationResult(is_valid=True),
            prosody_validation=ProsodyValidationResult(overall_valid=True),
            qafiya_validation=QafiyaValidationResult(overall_valid=True)
        )
        perfect_evaluation.poem = Mock()
        perfect_evaluation.poem.verses = ["verse1", "verse2"]
        
        score = chain._calculate_quality_score(perfect_evaluation)
        assert score == 1.0
        
        # Test poem with issues
        problematic_evaluation = QualityAssessment(
            line_count_validation=LineCountValidationResult(is_valid=False),
            prosody_validation=ProsodyValidationResult(overall_valid=False),
            qafiya_validation=QafiyaValidationResult(overall_valid=False)
        )
        problematic_evaluation.poem = Mock()
        problematic_evaluation.poem.verses = ["verse1", "verse2"]
        
        score = chain._calculate_quality_score(problematic_evaluation)
        assert score < 1.0
        assert score >= 0.0
    
    def test_refinement_summary_statistics(self):
        """Test refinement summary generation"""
        chain = RefinerChain([])
        
        # Create mock refinement history
        history = [
            RefinementStep(
                refiner_name="line_count_refiner",
                iteration=0,
                before=Mock(spec=LLMPoem),
                after=Mock(spec=LLMPoem),
                quality_before=0.3,
                quality_after=0.6
            ),
            RefinementStep(
                refiner_name="prosody_refiner",
                iteration=0,
                before=Mock(spec=LLMPoem),
                after=Mock(spec=LLMPoem),
                quality_before=0.6,
                quality_after=0.8
            ),
            RefinementStep(
                refiner_name="qafiya_refiner",
                iteration=1,
                before=Mock(spec=LLMPoem),
                after=Mock(spec=LLMPoem),
                quality_before=0.8,
                quality_after=0.9
            )
        ]
        
        summary = chain.get_refinement_summary(history)
        
        assert summary["total_steps"] == 3
        assert set(summary["refiners_used"]) == {"line_count_refiner", "prosody_refiner", "qafiya_refiner"}
        assert summary["quality_improvement"] == 0.6  # 0.9 - 0.3
        assert summary["iterations"] == 2  # Max iteration was 1
    
    @pytest.mark.asyncio
    async def test_refiner_metadata_preservation(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test that refiners preserve poem metadata"""
        refiner = LineCountRefiner(mock_llm, mock_prompt_manager)
        
        # Create evaluation with line count issue
        evaluation = QualityAssessment(
            line_count_validation=LineCountValidationResult(
                is_valid=False,
                expected_count=4,
                actual_count=3
            )
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Check metadata preservation
        assert result.llm_provider == sample_poem.llm_provider
        assert result.model_name == sample_poem.model_name
        assert result.constraints == sample_poem.constraints
        assert result.generation_timestamp == sample_poem.generation_timestamp
    
    def test_refiner_interface_compliance(self):
        """Test that all refiners properly implement the BaseRefiner interface"""
        mock_llm = Mock(spec=BaseLLM)
        
        # Test all refiner types
        refiners = [
            LineCountRefiner(mock_llm),
            ProsodyRefiner(mock_llm),
            QafiyaRefiner(mock_llm)
        ]
        
        for refiner in refiners:
            # Check inheritance
            assert isinstance(refiner, BaseRefiner)
            
            # Check required methods exist
            assert hasattr(refiner, 'name')
            assert hasattr(refiner, 'should_refine')
            assert hasattr(refiner, 'refine')
            
            # Check method signatures
            assert callable(refiner.should_refine)
            assert callable(refiner.refine)
            
            # Check name is string
            assert isinstance(refiner.name, str)
            assert len(refiner.name) > 0 
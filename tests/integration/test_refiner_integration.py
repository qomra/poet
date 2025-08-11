# tests/integration/test_refiner_integration.py

import pytest
from unittest.mock import Mock, AsyncMock
from poet.refinement import (
    BaseRefiner, RefinementStep, LineCountRefiner, 
    ProsodyRefiner, QafiyaRefiner, TashkeelRefiner, RefinerChain
)
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.models.line_count import LineCountValidationResult
from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
from poet.models.qafiya import QafiyaValidationResult, QafiyaBaitResult
from poet.models.tashkeel import TashkeelValidationResult, TashkeelBaitResult
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
            qafiya="ق"
        )
    
    @pytest.fixture
    def problematic_evaluation(self):
        """Create evaluation with multiple issues"""
        # Line count issue
        line_count_validation = LineCountValidationResult(
            is_valid=False,
            line_count=3,
            expected_even=True,
            validation_summary="Wrong line count"
        )
        
        # Prosody issue
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[
                BaitValidationResult(
                    bait_text="بيت أول بيت ثاني",
                    is_valid=False,
                    pattern="طويل",
                    error_details="وزن خاطئ في البيت الأول"
                )
            ],
            bahr_used="طويل",
            validation_summary="يوجد أخطاء في الوزن العروضي"
        )
        
        # Qafiya issue
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[
                QafiyaBaitResult(
                    bait_number=0,
                    is_valid=False,
                    error_details="قافية خاطئة"
                )
            ],
            validation_summary="قافية خاطئة",
            misaligned_bait_numbers=[]
        )
        
        # Tashkeel issue
        tashkeel_validation = TashkeelValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[
                TashkeelBaitResult(
                    bait_number=0,
                    is_valid=False,
                    error_details="حرف بدون تشكيل"
                )
            ],
            validation_summary="يوجد أخطاء في التشكيل"
        )
        
        return QualityAssessment(
            prosody_issues=["وزن خاطئ في البيت الأول"],
            line_count_issues=["Wrong line count"],
            qafiya_issues=["قافية خاطئة"],
            tashkeel_issues=["حرف بدون تشكيل"],
            overall_score=0.3,
            is_acceptable=False,
            recommendations=["تأكد من أن عدد الأبيات زوجي", "راجع الأوزان العروضية للأبيات", "راجع القافية في الأبيات", "راجع التشكيل في الأبيات"],
            line_count_validation=line_count_validation,
            prosody_validation=prosody_validation,
            qafiya_validation=qafiya_validation,
            tashkeel_validation=tashkeel_validation
        )
    
    def test_refiner_chain_with_all_refiners(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints, problematic_evaluation):
        """Test refiner chain with all four refiners"""
        # Create all refiners
        line_count_refiner = LineCountRefiner(mock_llm, mock_prompt_manager)
        prosody_refiner = ProsodyRefiner(mock_llm, mock_prompt_manager)
        qafiya_refiner = QafiyaRefiner(mock_llm, mock_prompt_manager)
        tashkeel_refiner = TashkeelRefiner(mock_llm, mock_prompt_manager)
        
        # Test that all refiners identify issues
        assert line_count_refiner.should_refine(problematic_evaluation) is True
        assert prosody_refiner.should_refine(problematic_evaluation) is True
        assert qafiya_refiner.should_refine(problematic_evaluation) is True
        assert tashkeel_refiner.should_refine(problematic_evaluation) is True
        
        # Test refiner names
        assert line_count_refiner.name == "line_count_refiner"
        assert prosody_refiner.name == "prosody_refiner"
        assert qafiya_refiner.name == "qafiya_refiner"
        assert tashkeel_refiner.name == "tashkeel_refiner"
    
    @pytest.mark.asyncio
    async def test_refiner_chain_sequential_execution(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test that refiners can be executed sequentially"""
        # Create refiners
        refiners = [
            LineCountRefiner(mock_llm, mock_prompt_manager),
            ProsodyRefiner(mock_llm, mock_prompt_manager),
            QafiyaRefiner(mock_llm, mock_prompt_manager),
            TashkeelRefiner(mock_llm, mock_prompt_manager)
        ]
        
        # Create chain
        chain = RefinerChain(refiners, mock_llm, max_iterations=2)
        
        # Mock evaluator to simulate improvement
        mock_evaluator = Mock()
        mock_evaluator.evaluate_poem = Mock()
                # Create mock poems with quality assessments
        mock_poem_1 = Mock(spec=LLMPoem)
        mock_poem_1.verses = ["verse1", "verse2", "verse3", "verse4"]
        mock_poem_1.quality = QualityAssessment(
            prosody_issues=["وزن خاطئ"],
            line_count_issues=["عدد خاطئ"],
            qafiya_issues=["قافية خاطئة"],
            tashkeel_issues=["تشكيل خاطئ"],
            overall_score=0.3,
            is_acceptable=False,
            recommendations=["تأكد من أن عدد الأبيات زوجي", "راجع الأوزان العروضية للأبيات", "راجع القافية في الأبيات", "راجع التشكيل في الأبيات"],
            line_count_validation=LineCountValidationResult(is_valid=False, line_count=3, expected_even=True, validation_summary="عدد خاطئ"),
            prosody_validation=ProsodyValidationResult(overall_valid=False, total_baits=1, valid_baits=0, invalid_baits=1, bait_results=[], bahr_used="طويل", validation_summary="وزن خاطئ"),
            qafiya_validation=QafiyaValidationResult(overall_valid=False, total_baits=1, valid_baits=0, invalid_baits=1, bait_results=[], validation_summary="قافية خاطئة", misaligned_bait_numbers=[]),
            tashkeel_validation=TashkeelValidationResult(overall_valid=False, total_baits=1, valid_baits=0, invalid_baits=1, bait_results=[], validation_summary="تشكيل خاطئ")     
        )
        
        mock_poem_2 = Mock(spec=LLMPoem)
        mock_poem_2.verses = ["verse1", "verse2", "verse3", "verse4"]
        mock_poem_2.quality = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=0.9,
            is_acceptable=True,
            recommendations=[],
            line_count_validation=LineCountValidationResult(is_valid=True, line_count=4, expected_even=True, validation_summary="عدد صحيح"),
            prosody_validation=ProsodyValidationResult(overall_valid=True, total_baits=2, valid_baits=2, invalid_baits=0, bait_results=[], bahr_used="طويل", validation_summary="وزن صحيح"),
            qafiya_validation=QafiyaValidationResult(overall_valid=True, total_baits=2, valid_baits=2, invalid_baits=0, bait_results=[], validation_summary="قافية صحيحة", misaligned_bait_numbers=[]),
            tashkeel_validation=TashkeelValidationResult(overall_valid=True, total_baits=2, valid_baits=2, invalid_baits=0, bait_results=[], validation_summary="تشكيل صحيح")      
        )
        
        mock_poem_3 = Mock(spec=LLMPoem)
        mock_poem_3.verses = ["verse1", "verse2", "verse3", "verse4"]
        mock_poem_3.quality = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=0.9,
            is_acceptable=True,
            recommendations=[],
            line_count_validation=LineCountValidationResult(is_valid=True, line_count=4, expected_even=True, validation_summary="عدد صحيح"),
            prosody_validation=ProsodyValidationResult(overall_valid=True, total_baits=2, valid_baits=2, invalid_baits=0, bait_results=[], bahr_used="طويل", validation_summary="وزن صحيح"),
            qafiya_validation=QafiyaValidationResult(overall_valid=True, total_baits=2, valid_baits=2, invalid_baits=0, bait_results=[], validation_summary="قافية صحيحة", misaligned_bait_numbers=[]),
            tashkeel_validation=TashkeelValidationResult(overall_valid=True, total_baits=2, valid_baits=2, invalid_baits=0, bait_results=[], validation_summary="تشكيل صحيح")      
        )
        
        mock_evaluator.evaluate_poem.side_effect = [mock_poem_1, mock_poem_2, mock_poem_3]
        chain.evaluator = mock_evaluator
        
        # Execute refinement
        result_poem, history = await chain.refine(sample_poem, sample_constraints, target_quality=0.8)
        
        # Verify results
        assert result_poem is not None
        assert len(history) > 0
        assert mock_evaluator.evaluate_poem.call_count == 3  # Initial + 1 iteration + final evaluation
        
        # Verify refinement steps
        for step in history:
            assert isinstance(step, RefinementStep)
            assert step.refiner_name in ["line_count_refiner", "prosody_refiner", "qafiya_refiner", "tashkeel_refiner"]
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
            prosody_issues=[],
            line_count_issues=["Wrong line count"],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=0.7,
            is_acceptable=False,
            recommendations=["تأكد من أن عدد الأبيات زوجي"],
            line_count_validation=LineCountValidationResult(
                is_valid=False,
                line_count=3,
                expected_even=True,
                validation_summary="Wrong line count"
            )
        )
        
        # Should not raise exception, should return original poem
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        assert result == sample_poem
    
    def test_refiner_chain_quality_calculation(self):
        """Test quality score calculation in refiner chain"""
        mock_llm = Mock(spec=BaseLLM)
        chain = RefinerChain([], mock_llm)
        
        # Test perfect poem
        perfect_evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[],
            line_count_validation=LineCountValidationResult(is_valid=True, line_count=2, expected_even=True, validation_summary="عدد صحيح"),
            prosody_validation=ProsodyValidationResult(overall_valid=True, total_baits=1, valid_baits=1, invalid_baits=0, bait_results=[], bahr_used="طويل", validation_summary="وزن صحيح"),
            qafiya_validation=QafiyaValidationResult(overall_valid=True, total_baits=1, valid_baits=1, invalid_baits=0, bait_results=[], validation_summary="قافية صحيحة", misaligned_bait_numbers=[]),
            tashkeel_validation=TashkeelValidationResult(overall_valid=True, total_baits=1, valid_baits=1, invalid_baits=0, bait_results=[], validation_summary="تشكيل صحيح")
        )
        perfect_evaluation.poem = Mock()
        perfect_evaluation.poem.verses = ["verse1", "verse2"]
        
        score = chain._calculate_quality_score(perfect_evaluation)
        assert score == 1.0
        
        # Test poem with issues
        problematic_evaluation = QualityAssessment(
            prosody_issues=["وزن خاطئ"],
            line_count_issues=["عدد خاطئ"],
            qafiya_issues=["قافية خاطئة"],
            tashkeel_issues=["تشكيل خاطئ"],
            overall_score=0.3,
            is_acceptable=False,
            recommendations=["تأكد من أن عدد الأبيات زوجي", "راجع الأوزان العروضية للأبيات", "راجع القافية في الأبيات", "راجع التشكيل في الأبيات"],
            line_count_validation=LineCountValidationResult(is_valid=False, line_count=1, expected_even=True, validation_summary="عدد خاطئ"),
            prosody_validation=ProsodyValidationResult(overall_valid=False, total_baits=1, valid_baits=0, invalid_baits=1, bait_results=[], bahr_used="طويل", validation_summary="وزن خاطئ"),
            qafiya_validation=QafiyaValidationResult(overall_valid=False, total_baits=1, valid_baits=0, invalid_baits=1, bait_results=[], validation_summary="قافية خاطئة", misaligned_bait_numbers=[]),
            tashkeel_validation=TashkeelValidationResult(overall_valid=False, total_baits=1, valid_baits=0, invalid_baits=1, bait_results=[], validation_summary="تشكيل خاطئ")
        )
        problematic_evaluation.poem = Mock()
        problematic_evaluation.poem.verses = ["verse1", "verse2"]
        
        score = chain._calculate_quality_score(problematic_evaluation)
        assert score < 1.0
        assert score >= 0.0
    
    def test_refinement_summary_statistics(self):
        """Test refinement summary generation"""
        mock_llm = Mock(spec=BaseLLM)
        chain = RefinerChain([], mock_llm)
        
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
        assert abs(summary["quality_improvement"] - 0.6) < 0.001  # 0.9 - 0.3
        assert summary["iterations"] == 2  # Max iteration was 1
    
    def test_refinement_summary_with_tashkeel(self):
        """Test refinement summary generation including tashkeel refiner"""
        mock_llm = Mock(spec=BaseLLM)
        chain = RefinerChain([], mock_llm)
        
        # Create mock refinement history with tashkeel refiner
        history = [
            RefinementStep(
                refiner_name="tashkeel_refiner",
                iteration=0,
                before=Mock(spec=LLMPoem),
                after=Mock(spec=LLMPoem),
                quality_before=0.4,
                quality_after=0.7
            ),
            RefinementStep(
                refiner_name="line_count_refiner",
                iteration=1,
                before=Mock(spec=LLMPoem),
                after=Mock(spec=LLMPoem),
                quality_before=0.7,
                quality_after=0.9
            )
        ]
        
        summary = chain.get_refinement_summary(history)
        
        assert summary["total_steps"] == 2
        assert set(summary["refiners_used"]) == {"tashkeel_refiner", "line_count_refiner"}
        assert summary["quality_improvement"] == 0.5  # 0.9 - 0.4
        assert summary["iterations"] == 2
    
    @pytest.mark.asyncio
    async def test_refiner_metadata_preservation(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test that refiners preserve poem metadata"""
        refiner = LineCountRefiner(mock_llm, mock_prompt_manager)
        
        # Create evaluation with line count issue
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=["Wrong line count"],
            qafiya_issues=[],
            tashkeel_issues=[],
            overall_score=0.7,
            is_acceptable=False,
            recommendations=["تأكد من أن عدد الأبيات زوجي"],
            line_count_validation=LineCountValidationResult(
                is_valid=False,
                line_count=3,
                expected_even=True,
                validation_summary="Wrong line count"
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
            QafiyaRefiner(mock_llm),
            TashkeelRefiner(mock_llm)
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
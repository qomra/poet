# tests/unit/test_qafiya_refiner.py

import pytest
from unittest.mock import Mock, AsyncMock
from poet.refinement.qafiya_refiner import QafiyaRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.models.qafiya import QafiyaValidationResult, QafiyaBaitResult
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class TestQafiyaRefiner:
    """Test QafiyaRefiner functionality"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM"""
        llm = Mock(spec=BaseLLM)
        llm.generate = Mock(return_value='{"verses": ["بيت بقافية صحيحة"]}')
        return llm
    
    @pytest.fixture
    def mock_prompt_manager(self):
        """Create mock prompt manager"""
        pm = Mock(spec=PromptManager)
        pm.format_prompt = Mock(return_value="formatted prompt")
        return pm
    
    @pytest.fixture
    def sample_poem(self):
        """Create sample poem"""
        return LLMPoem(
            verses=["بيت أول", "بيت ثاني", "بيت ثالث", "بيت رابع"],
            llm_provider="test_provider",
            model_name="test_model"
        )
    
    @pytest.fixture
    def sample_constraints(self):
        """Create sample constraints"""
        return Constraints(
            meter="بحر الطويل",
            qafiya="ق",
            qafiya_pattern="قَ"
        )
    
    def test_qafiya_refiner_initialization(self, mock_llm, mock_prompt_manager):
        """Test refiner initialization"""
        refiner = QafiyaRefiner(mock_llm, mock_prompt_manager)
        
        assert refiner.name == "qafiya_refiner"
        assert refiner.llm == mock_llm
        assert refiner.prompt_manager == mock_prompt_manager
    
    def test_should_refine_with_qafiya_issues(self, mock_llm):
        """Test should_refine when qafiya validation fails"""
        refiner = QafiyaRefiner(mock_llm)
        
        # Create evaluation with qafiya issues
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=[],
            validation_summary="Qafiya issues",
            misaligned_bait_numbers=[]
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=["Qafiya issues"],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            qafiya_validation=qafiya_validation
        )
        
        assert refiner.should_refine(evaluation) is True
    
    def test_should_refine_without_qafiya_issues(self, mock_llm):
        """Test should_refine when qafiya is correct"""
        refiner = QafiyaRefiner(mock_llm)
        
        # Create evaluation without qafiya issues
        qafiya_validation = QafiyaValidationResult(
            overall_valid=True,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=[],
            validation_summary="No qafiya issues",
            misaligned_bait_numbers=[]
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[],
            qafiya_validation=qafiya_validation
        )
        
        assert refiner.should_refine(evaluation) is False
    
    def test_should_refine_without_validation(self, mock_llm):
        """Test should_refine when no qafiya validation exists"""
        refiner = QafiyaRefiner(mock_llm)
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[]
        )
        
        assert refiner.should_refine(evaluation) is False
    
    @pytest.mark.asyncio
    async def test_refine_no_qafiya_validation(self, mock_llm, sample_poem, sample_constraints):
        """Test refine when no qafiya validation exists"""
        refiner = QafiyaRefiner(mock_llm)
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[]
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem unchanged
        assert result == sample_poem
    
    @pytest.mark.asyncio
    async def test_refine_no_wrong_qafiya_verses(self, mock_llm, sample_poem, sample_constraints):
        """Test refine when no verses have wrong qafiya"""
        refiner = QafiyaRefiner(mock_llm)
        
        # Create validation with no wrong qafiya verses
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=[],
            validation_summary="No qafiya issues",
            misaligned_bait_numbers=[]
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            qafiya_validation=qafiya_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem unchanged
        assert result == sample_poem
    
    @pytest.mark.asyncio
    async def test_refine_with_wrong_qafiya_verses(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test refine when verses have wrong qafiya"""
        refiner = QafiyaRefiner(mock_llm, mock_prompt_manager)
        
        # Create validation with wrong qafiya verses
        bait_result = QafiyaBaitResult(
            bait_number=0,
            is_valid=False,
            error_details="قافية خاطئة"
        )
        
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[bait_result],
            validation_summary="Wrong qafiya in first verse",
            misaligned_bait_numbers=[]
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            qafiya_validation=qafiya_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should have fixed the wrong qafiya verse
        assert len(result.verses) == len(sample_poem.verses)
        assert result.verses[0] == "بيت بقافية صحيحة"  # Fixed verse
        assert result.verses[1:] == sample_poem.verses[1:]  # Other verses unchanged
        
        # Check that prompt was formatted correctly
        mock_prompt_manager.format_prompt.assert_called_once()
        call_args = mock_prompt_manager.format_prompt.call_args
        assert call_args[0][0] == 'qafiya_refinement'
        assert 'context' in call_args[1]
    
    @pytest.mark.asyncio
    async def test_refine_multiple_wrong_qafiya_verses(self, mock_llm, sample_poem, sample_constraints):
        """Test refine with multiple wrong qafiya verses"""
        refiner = QafiyaRefiner(mock_llm)
        
        # Create validation with multiple wrong qafiya verses
        bait_result1 = QafiyaBaitResult(
            bait_number=0,
            is_valid=False,
            error_details="قافية خاطئة في البيت الأول"
        )
        
        bait_result2 = QafiyaBaitResult(
            bait_number=1,
            is_valid=False,
            error_details="قافية خاطئة في البيت الثاني"
        )
        
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=2,
            valid_baits=0,
            invalid_baits=2,
            bait_results=[bait_result1, bait_result2],
            validation_summary="Multiple wrong qafiya verses",
            misaligned_bait_numbers=[]
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=["قافية خاطئة في البيت الأول", "قافية خاطئة في البيت الثاني"],
            overall_score=0.6,
            is_acceptable=False,
            recommendations=[],
            qafiya_validation=qafiya_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should have fixed both wrong qafiya verses
        assert len(result.verses) == len(sample_poem.verses)
        assert result.verses[0] == "بيت بقافية صحيحة"  # First verse fixed (bait 0)
        assert result.verses[2] == "بيت بقافية صحيحة"  # Third verse fixed (bait 1)
        assert result.verses[1] == sample_poem.verses[1]  # Second verse unchanged
        assert result.verses[3] == sample_poem.verses[3]  # Fourth verse unchanged
    
    @pytest.mark.asyncio
    async def test_refine_exception_handling(self, mock_llm, sample_poem, sample_constraints):
        """Test refine handles exceptions gracefully"""
        refiner = QafiyaRefiner(mock_llm)
        
        # Make LLM raise an exception
        mock_llm.generate.side_effect = Exception("LLM error")
        
        bait_result = QafiyaBaitResult(
            bait_number=0,
            is_valid=False,
            error_details="قافية خاطئة"
        )
        
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[bait_result],
            validation_summary="قافية خاطئة",
            misaligned_bait_numbers=[]
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=["قافية خاطئة"],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            qafiya_validation=qafiya_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem on error
        assert result == sample_poem
    
    def test_identify_wrong_qafiya_verses(self, mock_llm):
        """Test identifying wrong qafiya verses from validation"""
        refiner = QafiyaRefiner(mock_llm)
        
        # Create validation with wrong qafiya verses
        bait_result1 = QafiyaBaitResult(
            bait_number=0,
            is_valid=False,
            error_details="قافية خاطئة في البيت الأول"
        )
        
        bait_result2 = QafiyaBaitResult(
            bait_number=1,
            is_valid=False,
            error_details="قافية خاطئة في البيت الثاني"
        )
        
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=2,
            valid_baits=0,
            invalid_baits=2,
            bait_results=[bait_result1, bait_result2],
            validation_summary="Multiple wrong qafiya verses",
            misaligned_bait_numbers=[]
        )
        
        wrong_qafiya_verses = refiner._identify_wrong_qafiya_verses(qafiya_validation)
        
        # Should identify both wrong qafiya verses
        assert len(wrong_qafiya_verses) == 2
        assert wrong_qafiya_verses[0] == (0, "")  # First verse (no expected_qafiya in new model)
        assert wrong_qafiya_verses[1] == (2, "")  # Third verse (second bait)
    
    def test_identify_wrong_qafiya_verses_no_bait_results(self, mock_llm):
        """Test identifying wrong qafiya verses when no bait results exist"""
        refiner = QafiyaRefiner(mock_llm)
        
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=None,
            validation_summary="No bait results",
            misaligned_bait_numbers=[]
        )
        
        wrong_qafiya_verses = refiner._identify_wrong_qafiya_verses(qafiya_validation)
        
        assert wrong_qafiya_verses == []
    
    @pytest.mark.asyncio
    async def test_fix_single_verse_qafiya(self, mock_llm, mock_prompt_manager, sample_constraints):
        """Test fixing a single verse's qafiya"""
        refiner = QafiyaRefiner(mock_llm, mock_prompt_manager)
        
        original_verse = "بيت بقافية خاطئة"
        expected_qafiya = "قَ"
        
        fixed_verse = await refiner._fix_single_verse_qafiya(original_verse, sample_constraints, expected_qafiya)
        
        assert fixed_verse == "بيت بقافية صحيحة"
        
        # Check that prompt was formatted correctly
        mock_prompt_manager.format_prompt.assert_called_once()
        call_args = mock_prompt_manager.format_prompt.call_args
        assert call_args[0][0] == 'qafiya_refinement'
        assert call_args[1]['existing_verses'] == original_verse
        assert call_args[1]['context'] == f"إصلاح القافية للبيت. القافية المطلوبة: {expected_qafiya}"
    
    def test_parse_verses_from_response_json(self, mock_llm):
        """Test parsing verses from JSON response"""
        refiner = QafiyaRefiner(mock_llm)
        
        response = '{"verses": ["بيت بقافية صحيحة"]}'
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == ["بيت بقافية صحيحة"]
    
    def test_parse_verses_from_response_newlines(self, mock_llm):
        """Test parsing verses from newline-separated response"""
        refiner = QafiyaRefiner(mock_llm)
        
        response = "بيت بقافية صحيحة\nبيت آخر"
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == ["بيت بقافية صحيحة", "بيت آخر"]
    
    @pytest.mark.asyncio
    async def test_refine_preserves_poem_metadata(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test that refine preserves poem metadata"""
        refiner = QafiyaRefiner(mock_llm, mock_prompt_manager)
        
        bait_result = QafiyaBaitResult(
            bait_number=0,
            is_valid=False,
            error_details="قافية خاطئة"
        )
        
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[bait_result],
            validation_summary="قافية خاطئة",
            misaligned_bait_numbers=[]
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=["قافية خاطئة"],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            qafiya_validation=qafiya_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Check metadata is preserved
        assert result.llm_provider == sample_poem.llm_provider
        assert result.model_name == sample_poem.model_name
        assert result.constraints == sample_poem.constraints
        assert result.generation_timestamp == sample_poem.generation_timestamp
    
    def test_identify_wrong_qafiya_verses_missing_attributes(self, mock_llm):
        """Test identifying wrong qafiya verses when bait results have missing attributes"""
        refiner = QafiyaRefiner(mock_llm)
        
        # Create bait result with missing attributes
        bait_result = Mock()
        bait_result.bait_index = 0
        bait_result.is_valid = False
        bait_result.first_verse_valid = False
        bait_result.second_verse_valid = True
        bait_result.expected_qafiya = ""  # Empty expected_qafiya
        
        qafiya_validation = QafiyaValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[bait_result],
            validation_summary="Missing attributes",
            misaligned_bait_numbers=[]
        )
        
        wrong_qafiya_verses = refiner._identify_wrong_qafiya_verses(qafiya_validation)
        
        # Should handle missing attributes gracefully
        assert len(wrong_qafiya_verses) == 1
        assert wrong_qafiya_verses[0][0] == 0  # verse index
        assert wrong_qafiya_verses[0][1] == ""  # empty expected_qafiya 
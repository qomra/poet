# tests/unit/test_prosody_refiner.py

import pytest
from unittest.mock import Mock, AsyncMock
from poet.refinement.prosody import ProsodyRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class TestProsodyRefiner:
    """Test ProsodyRefiner functionality"""
    
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
            qafiya="ق"
        )
    
    def test_prosody_refiner_initialization(self, mock_llm, mock_prompt_manager):
        """Test refiner initialization"""
        refiner = ProsodyRefiner(mock_llm, mock_prompt_manager)
        
        assert refiner.name == "prosody_refiner"
        assert refiner.llm == mock_llm
        assert refiner.prompt_manager == mock_prompt_manager
    
    def test_should_refine_with_prosody_issues(self, mock_llm):
        """Test should_refine when prosody validation fails"""
        refiner = ProsodyRefiner(mock_llm)
        
        # Create evaluation with prosody issues
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=[],
            bahr_used="طويل",
            validation_summary="Prosody issues"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=["Prosody issues"],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            prosody_validation=prosody_validation
        )
        
        assert refiner.should_refine(evaluation) is True
    
    def test_should_refine_without_prosody_issues(self, mock_llm):
        """Test should_refine when prosody is correct"""
        refiner = ProsodyRefiner(mock_llm)
        
        # Create evaluation without prosody issues
        prosody_validation = ProsodyValidationResult(
            overall_valid=True,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=[],
            bahr_used="طويل",
            validation_summary="No prosody issues"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[],
            prosody_validation=prosody_validation
        )
        
        assert refiner.should_refine(evaluation) is False
    
    def test_should_refine_without_validation(self, mock_llm):
        """Test should_refine when no prosody validation exists"""
        refiner = ProsodyRefiner(mock_llm)
        
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
    async def test_refine_no_prosody_validation(self, mock_llm, sample_poem, sample_constraints):
        """Test refine when no prosody validation exists"""
        refiner = ProsodyRefiner(mock_llm)
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[]
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem unchanged
        assert result == sample_poem
    
    @pytest.mark.asyncio
    async def test_refine_no_broken_verses(self, mock_llm, sample_poem, sample_constraints):
        """Test refine when no verses are broken"""
        refiner = ProsodyRefiner(mock_llm)
        
        # Create validation with no broken verses
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=[],
            bahr_used="طويل",
            validation_summary="No broken verses"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[],
            prosody_validation=prosody_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem unchanged
        assert result == sample_poem
    
    @pytest.mark.asyncio
    async def test_refine_with_broken_verses(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test refine when verses have prosody issues"""
        refiner = ProsodyRefiner(mock_llm, mock_prompt_manager)
        
        # Create validation with broken verses
        bait_result = BaitValidationResult(
            bait_text="test bait",
            is_valid=False,
            pattern="test pattern",
            error_details="وزن خاطئ في البيت الأول"
        )
        
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[bait_result],
            bahr_used="طويل",
            validation_summary="وزن خاطئ في البيت الأول"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=["وزن خاطئ في البيت الأول"],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            prosody_validation=prosody_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original verses when refinement fails (which is correct behavior)
        assert len(result.verses) == len(sample_poem.verses)
        assert result.verses == sample_poem.verses  # Original verses preserved
        
        # Check that prompt was formatted correctly
        mock_prompt_manager.format_prompt.assert_called_once()
        call_args = mock_prompt_manager.format_prompt.call_args
        assert call_args[0][0] == 'prosody_refinement'
        assert 'context' in call_args[1]
    
    @pytest.mark.asyncio
    async def test_refine_multiple_broken_verses(self, mock_llm, sample_poem, sample_constraints):
        """Test refine with multiple broken verses"""
        refiner = ProsodyRefiner(mock_llm)
        
        # Create validation with multiple broken verses
        bait_result1 = BaitValidationResult(
            bait_text="test bait 1",
            is_valid=False,
            pattern="test pattern",
            error_details="وزن خاطئ في البيت الأول"
        )
        
        bait_result2 = BaitValidationResult(
            bait_text="test bait 2",
            is_valid=False,
            pattern="test pattern",
            error_details="وزن خاطئ في البيت الرابع"
        )
        
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=2,
            valid_baits=0,
            invalid_baits=2,
            bait_results=[bait_result1, bait_result2],
            bahr_used="طويل",
            validation_summary="Multiple broken verses"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=["وزن خاطئ في البيت الأول", "وزن خاطئ في البيت الرابع"],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.6,
            is_acceptable=False,
            recommendations=[],
            prosody_validation=prosody_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original verses when refinement fails (which is correct behavior)
        assert len(result.verses) == len(sample_poem.verses)
        assert result.verses == sample_poem.verses  # Original verses preserved
    
    @pytest.mark.asyncio
    async def test_refine_exception_handling(self, mock_llm, sample_poem, sample_constraints):
        """Test refine handles exceptions gracefully"""
        refiner = ProsodyRefiner(mock_llm)
        
        # Make LLM raise an exception
        mock_llm.generate.side_effect = Exception("LLM error")
        
        bait_result = BaitValidationResult(
            bait_text="test bait",
            is_valid=False,
            pattern="test pattern",
            error_details="وزن خاطئ"
        )
        
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[bait_result],
            bahr_used="طويل",
            validation_summary="وزن خاطئ"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=["وزن خاطئ"],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            prosody_validation=prosody_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem on error
        assert result == sample_poem
    
    def test_identify_broken_bait(self, mock_llm):
        """Test identifying broken bait from validation"""
        refiner = ProsodyRefiner(mock_llm)

        # Create validation with broken verses
        bait_result1 = BaitValidationResult(
            bait_text="test bait 1",
            is_valid=False,
            pattern="test pattern",
            error_details="وزن خاطئ في البيت الأول"
        )

        bait_result2 = BaitValidationResult(
            bait_text="test bait 2",
            is_valid=False,
            pattern="test pattern",
            error_details="وزن خاطئ في البيت الرابع"
        )

        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=2,
            valid_baits=0,
            invalid_baits=2,
            bait_results=[bait_result1, bait_result2],
            bahr_used="طويل",
            validation_summary="Multiple broken verses"
        )

        broken_bait = refiner._identify_broken_bait(prosody_validation)
        
        # Should identify both broken bait
        assert len(broken_bait) == 2
        assert broken_bait[0] == (0, "وزن خاطئ في البيت الأول")  # First bait
        assert broken_bait[1] == (1, "وزن خاطئ في البيت الرابع")  # Second bait
    
    def test_identify_broken_bait_no_bait_results(self, mock_llm):
        """Test identifying broken bait when no bait results exist"""
        refiner = ProsodyRefiner(mock_llm)

        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=0,
            valid_baits=0,
            invalid_baits=0,
            bait_results=None,
            bahr_used="طويل",
            validation_summary="No bait results"
        )

        broken_bait = refiner._identify_broken_bait(prosody_validation)
        
        assert broken_bait == []
    
    @pytest.mark.asyncio
    async def test_fix_single_verse(self, mock_llm, mock_prompt_manager, sample_constraints):
        """Test fixing a single verse"""
        refiner = ProsodyRefiner(mock_llm, mock_prompt_manager)
        
        original_verse = "بيت أصلي"
        error_details = "وزن خاطئ"
        
        fixed_verse = await refiner._fix_single_verse(original_verse, sample_constraints, error_details)
        
        assert fixed_verse[0] == "بيت مصحح"
        
        # Check that prompt was formatted correctly
        mock_prompt_manager.format_prompt.assert_called_once()
        call_args = mock_prompt_manager.format_prompt.call_args
        assert call_args[0][0] == 'prosody_refinement'
        assert call_args[1]['existing_verses'] == original_verse
        assert call_args[1]['context'] == f"إصلاح الوزن العروضي للبيت. المشكلة: {error_details}"
    
    def test_parse_verses_from_response_json(self, mock_llm):
        """Test parsing verses from JSON response"""
        refiner = ProsodyRefiner(mock_llm)
        
        response = '{"verses": ["بيت مصحح"]}'
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == ["بيت مصحح"]
    
    def test_parse_verses_from_response_newlines(self, mock_llm):
        """Test parsing verses from newline-separated response"""
        refiner = ProsodyRefiner(mock_llm)
        
        response = "بيت مصحح\nبيت آخر"
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == ["بيت مصحح", "بيت آخر"]
    
    @pytest.mark.asyncio
    async def test_refine_preserves_poem_metadata(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test that refine preserves poem metadata"""
        refiner = ProsodyRefiner(mock_llm, mock_prompt_manager)
        
        bait_result = BaitValidationResult(
            bait_text="test bait",
            is_valid=False,
            pattern="test pattern",
            error_details="وزن خاطئ"
        )
        
        prosody_validation = ProsodyValidationResult(
            overall_valid=False,
            total_baits=1,
            valid_baits=0,
            invalid_baits=1,
            bait_results=[bait_result],
            bahr_used="طويل",
            validation_summary="وزن خاطئ"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=["وزن خاطئ"],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            prosody_validation=prosody_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Check metadata is preserved
        assert result.llm_provider == sample_poem.llm_provider
        assert result.model_name == sample_poem.model_name
        assert result.constraints == sample_poem.constraints
        assert result.generation_timestamp == sample_poem.generation_timestamp 
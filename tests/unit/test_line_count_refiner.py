# tests/unit/test_line_count_refiner.py

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from poet.refinement.line_count_refiner import LineCountRefiner
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.models.line_count import LineCountValidationResult
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager


class TestLineCountRefiner:
    """Test LineCountRefiner functionality"""
    
    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM"""
        llm = Mock(spec=BaseLLM)
        llm.generate = Mock(return_value='{"verses": ["بيت جديد 1", "بيت جديد 2"]}')
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
            line_count=6,
            meter="بحر الطويل",
            qafiya="ق"
        )
    
    def test_line_count_refiner_initialization(self, mock_llm, mock_prompt_manager):
        """Test refiner initialization"""
        refiner = LineCountRefiner(mock_llm, mock_prompt_manager)
        
        assert refiner.name == "line_count_refiner"
        assert refiner.llm == mock_llm
        assert refiner.prompt_manager == mock_prompt_manager
    
    def test_should_refine_with_line_count_issues(self, mock_llm):
        """Test should_refine when line count validation fails"""
        refiner = LineCountRefiner(mock_llm)
        
        # Create evaluation with line count issues
        line_count_validation = LineCountValidationResult(
            is_valid=False,
            line_count=4,
            expected_even=True,
            validation_summary="Wrong line count"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=["Wrong line count"],
            qafiya_issues=[],
            overall_score=0.7,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=line_count_validation
        )
        
        assert refiner.should_refine(evaluation) is True
    
    def test_should_refine_without_line_count_issues(self, mock_llm):
        """Test should_refine when line count is correct"""
        refiner = LineCountRefiner(mock_llm)
        
        # Create evaluation without line count issues
        line_count_validation = LineCountValidationResult(
            is_valid=True,
            line_count=4,
            expected_even=True,
            validation_summary="Correct line count"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[],
            line_count_validation=line_count_validation
        )
        
        assert refiner.should_refine(evaluation) is False
    
    def test_should_refine_without_validation(self, mock_llm):
        """Test should_refine when no line count validation exists"""
        refiner = LineCountRefiner(mock_llm)
        
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
    async def test_refine_no_change_needed(self, mock_llm, sample_poem, sample_constraints):
        """Test refine when line count is already correct"""
        refiner = LineCountRefiner(mock_llm)
        
        # Set constraints to match current poem
        sample_constraints.line_count = 4
        
        line_count_validation = LineCountValidationResult(
            is_valid=True,
            line_count=4,
            expected_even=True,
            validation_summary="Line count is correct"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=line_count_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem unchanged
        assert result == sample_poem
        assert result.verses == sample_poem.verses
    
    @pytest.mark.asyncio
    async def test_refine_add_verses(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test refine when verses need to be added"""
        refiner = LineCountRefiner(mock_llm, mock_prompt_manager)
        
        # Set constraints to require more verses
        sample_constraints.line_count = 6
        
        line_count_validation = LineCountValidationResult(
            is_valid=False,
            line_count=4,
            expected_even=True,
            validation_summary="Need 2 more verses"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=line_count_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should have more verses
        assert len(result.verses) == 6
        assert result.verses[:4] == sample_poem.verses  # Original verses preserved
        assert result.verses[4:] == ["بيت جديد 1", "بيت جديد 2"]  # New verses added
        
        # Check that prompt was formatted correctly
        mock_prompt_manager.format_prompt.assert_called_once()
        call_args = mock_prompt_manager.format_prompt.call_args
        assert call_args[0][0] == 'verse_completion'
        assert call_args[1]['verses_to_add'] == 2
    
    @pytest.mark.asyncio
    async def test_refine_remove_verses(self, mock_llm, sample_poem, sample_constraints):
        """Test refine when verses need to be removed"""
        refiner = LineCountRefiner(mock_llm)
        
        # Set constraints to require fewer verses
        sample_constraints.line_count = 2
        
        line_count_validation = LineCountValidationResult(
            is_valid=False,
            line_count=4,
            expected_even=True,
            validation_summary="Need to remove 2 verses"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=line_count_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should have fewer verses
        assert len(result.verses) == 2
        assert result.verses == sample_poem.verses[:2]  # First verses kept
    
    @pytest.mark.asyncio
    async def test_refine_exception_handling(self, mock_llm, sample_poem, sample_constraints):
        """Test refine handles exceptions gracefully"""
        refiner = LineCountRefiner(mock_llm)
        
        # Make LLM raise an exception
        mock_llm.generate.side_effect = Exception("LLM error")
        
        line_count_validation = LineCountValidationResult(
            is_valid=False,
            line_count=4,
            expected_even=True,
            validation_summary="Need 2 more verses"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=line_count_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Should return original poem on error
        assert result == sample_poem
    
    def test_parse_verses_from_response_json(self, mock_llm):
        """Test parsing verses from JSON response"""
        refiner = LineCountRefiner(mock_llm)
        
        response = '{"verses": ["بيت أول", "بيت ثاني", "بيت ثالث"]}'
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == ["بيت أول", "بيت ثاني", "بيت ثالث"]
    
    def test_parse_verses_from_response_newlines(self, mock_llm):
        """Test parsing verses from newline-separated response"""
        refiner = LineCountRefiner(mock_llm)
        
        response = "بيت أول\nبيت ثاني\nبيت ثالث"
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == ["بيت أول", "بيت ثاني", "بيت ثالث"]
    
    def test_parse_verses_from_response_invalid_json(self, mock_llm):
        """Test parsing verses from invalid JSON response"""
        refiner = LineCountRefiner(mock_llm)
        
        response = "invalid json\nبيت أول\nبيت ثاني"
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == ["invalid json", "بيت أول", "بيت ثاني"]
    
    def test_parse_verses_from_response_empty(self, mock_llm):
        """Test parsing verses from empty response"""
        refiner = LineCountRefiner(mock_llm)
        
        response = ""
        verses = refiner._parse_verses_from_response(response)
        
        assert verses == []
    
    @pytest.mark.asyncio
    async def test_refine_preserves_poem_metadata(self, mock_llm, mock_prompt_manager, sample_poem, sample_constraints):
        """Test that refine preserves poem metadata"""
        refiner = LineCountRefiner(mock_llm, mock_prompt_manager)
        
        line_count_validation = LineCountValidationResult(
            is_valid=False,
            line_count=4,
            expected_even=True,
            validation_summary="Need 2 more verses"
        )
        
        evaluation = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=0.8,
            is_acceptable=False,
            recommendations=[],
            line_count_validation=line_count_validation
        )
        
        result = await refiner.refine(sample_poem, sample_constraints, evaluation)
        
        # Check metadata is preserved
        assert result.llm_provider == sample_poem.llm_provider
        assert result.model_name == sample_poem.model_name
        assert result.constraints == sample_poem.constraints
        assert result.generation_timestamp == sample_poem.generation_timestamp 
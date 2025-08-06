import pytest
import os
from unittest.mock import Mock, patch
from poet.evaluation.prosody_validator import ProsodyValidator
from poet.models.poem import LLMPoem
from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
from poet.models.quality import QualityAssessment


class TestProsodyValidator:
    """Unit tests for ProsodyValidator"""
    
    @pytest.fixture
    def prosody_validator(self):
        """ProsodyValidator instance"""
        return ProsodyValidator()
    
    @pytest.fixture
    def sample_poem(self):
        """Sample LLMPoem with famous bait (pre-diacritized)"""
        return LLMPoem(
            verses=[
                "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ"
            ],
            llm_provider="mock",
            model_name="test-model"
        )


   
      
    @pytest.mark.skipif(
        not os.getenv("TEST_REAL_LLMS"),
        reason="Real LLM tests require TEST_REAL_LLMS environment variable"
    )

    def test_bahr_mapping(self, prosody_validator):
        """Test bahr name mapping to bohour classes"""
        # Test full names
        assert prosody_validator._get_bahr_class("بحر الطويل") is not None
        assert prosody_validator._get_bahr_class("بحر الكامل") is not None
        
        # Test short names
        assert prosody_validator._get_bahr_class("طويل") is not None
        assert prosody_validator._get_bahr_class("كامل") is not None
        
        # Test English names
        assert prosody_validator._get_bahr_class("Taweel") is not None
        assert prosody_validator._get_bahr_class("Kamel") is not None
        
        # Test unknown bahr
        assert prosody_validator._get_bahr_class("unknown") is None
    

    
    def test_validate_bait_success(self, prosody_validator):
        """Test successful bait validation with pre-diacritized input"""
        bait = ("قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ", "بِسَقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ")
        bahr_class = prosody_validator._get_bahr_class("طويل")
        
        result = prosody_validator._validate_bait(bait, bahr_class)
        
        # Verify result structure - pattern will be the real one from get_arudi_style
        assert isinstance(result, BaitValidationResult)
        assert result.bait_text == "#".join(bait)
        assert result.pattern == '110101101010110101101101101011010101101110110'
        assert result.diacritized_text == "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ#بِسَقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
    
    def test_validate_bait_pattern_mismatch(self, prosody_validator):
        """Test bait validation with pattern mismatch"""
        bait = ("قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ", "بِسَقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ")
        bahr_class = prosody_validator._get_bahr_class("كامل")  # Use different bahr to ensure mismatch
        
        result = prosody_validator._validate_bait(bait, bahr_class)
        
        # Verify result - should be invalid since this is Taweel pattern on Kamel bahr
        assert not result.is_valid
        assert result.pattern  # Should have a real pattern
        assert "لا يتطابق" in result.error_details
    
    def test_validate_bait_arudi_error(self, prosody_validator):
        """Test bait validation when arudi style extraction fails"""
        bait = ("invalid text that will cause arudi error",)
        bahr_class = prosody_validator._get_bahr_class("طويل")
        
        result = prosody_validator._validate_bait(bait, bahr_class)
        
        # Verify result - should fail due to invalid text
        assert not result.is_valid
        assert result.pattern == ""
        assert "فشل في استخراج النمط العروضي" in result.error_details
    
    def test_validate_poem_success(self, prosody_validator, sample_poem):
        """Test successful poem validation with pre-diacritized input"""
        with patch.object(prosody_validator, '_validate_bait') as mock_validate_bait:
            # Mock successful bait validation
            mock_validate_bait.return_value = BaitValidationResult(
                bait_text="قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ بِسَقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ",
                is_valid=True,
                pattern='110101101010110101101101101011010101101110110',
                diacritized_text="قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
            )
            
            result = prosody_validator.validate_poem(sample_poem, "طويل")
            
            # Verify poem was updated
            assert result.prosody_validation is not None
            assert result.prosody_validation.overall_valid is True
            assert result.prosody_validation.total_baits == 1
            assert result.prosody_validation.valid_baits == 1
            assert result.prosody_validation.invalid_baits == 0
            assert result.prosody_validation.bahr_used == "طويل"
            assert "جميع الأبيات" in result.prosody_validation.validation_summary
            
            # Verify quality assessment
            assert result.quality is not None
            assert result.quality.overall_score > 0.9
            assert result.quality.is_acceptable is True
            assert len(result.quality.prosody_issues) == 0
    
    def test_validate_poem_invalid_line_count(self, prosody_validator):
        """Test poem validation with invalid line count (should be handled by LineCountValidator)"""
        poem = LLMPoem(
            verses=["قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ"],  # Only one verse
            llm_provider="mock",
            model_name="test-model"
        )
        
        # This test now assumes line count validation is done separately
        # ProsodyValidator should still work with odd number of lines
        # but will only validate complete baits
        result = prosody_validator.validate_poem(poem, "طويل")
        
        # Verify validation was attempted but no complete baits to validate
        assert result.prosody_validation is not None
        assert result.prosody_validation.total_baits == 0
        assert result.prosody_validation.valid_baits == 0
        assert result.prosody_validation.invalid_baits == 0
    
    def test_validate_poem_unknown_bahr(self, prosody_validator, sample_poem):
        """Test poem validation with unknown bahr"""
        result = prosody_validator.validate_poem(sample_poem, "unknown_bahr")
        
        # Verify validation was skipped
        assert result.prosody_validation is None
        assert result.quality is not None
        assert not result.quality.is_acceptable
        assert len(result.quality.prosody_issues) > 0
        assert "غير معروف" in result.quality.prosody_issues[0]
    
    def test_validate_poem_mixed_results(self, prosody_validator, sample_poem):
        """Test poem validation with mixed valid/invalid baits"""
        with patch.object(prosody_validator, '_validate_bait') as mock_validate_bait:
            # Mock mixed results
            mock_validate_bait.side_effect = [
                BaitValidationResult(
                    bait_text="قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ بِسَقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ",
                    is_valid=True,
                    pattern="1010101010101010101010",
                    diacritized_text="قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ بِسِقْطِ اللّوَى بَيْنَ الدّخُولِ فَحَوْمَلِ"
                ),
                BaitValidationResult(
                    bait_text="another bait",
                    is_valid=False,
                    pattern="110111101111011110111101111",
                    error_details="النمط '110111101111011110111101111' لا يتطابق مع أنماط بحر Taweel",
                    diacritized_text="another bait"
                )
            ]
            
            # Create poem with 4 verses (2 baits) - pre-diacritized
            poem = LLMPoem(
                verses=[
                    "قِفَا نَبْكِ مِنْ ذِكْرَى حَبِيبٍ وَمَنْزِلِ",
                    "بِسَقْطِ اللِّوَى بَيْنَ الدَّخُولِ فَحَوْمَلِ",
                    "verse 3",
                    "verse 4"
                ],
                llm_provider="mock",
                model_name="test-model"
            )
            
            result = prosody_validator.validate_poem(poem, "طويل")
            
            # Verify results
            assert result.prosody_validation is not None
            assert result.prosody_validation.overall_valid is False
            assert result.prosody_validation.total_baits == 2
            assert result.prosody_validation.valid_baits == 1
            assert result.prosody_validation.invalid_baits == 1
            assert "1 من 2 أبيات" in result.prosody_validation.validation_summary
            assert "الأبيات الخاطئة: 2" in result.prosody_validation.validation_summary
            
            # Verify quality assessment
            assert result.quality is not None
            assert result.quality.overall_score < 1.0
            assert len(result.quality.prosody_issues) > 0
    
    def test_generate_validation_summary(self, prosody_validator):
        """Test validation summary generation"""
        # Test all valid
        summary = prosody_validator._generate_validation_summary(2, 0, "طويل")
        assert "جميع الأبيات (2) صحيحة" in summary
        
        # Test all invalid
        summary = prosody_validator._generate_validation_summary(0, 2, "طويل")
        assert "جميع الأبيات (2) خاطئة" in summary
        
        # Test mixed with few invalid baits
        bait_results = [
            BaitValidationResult("bait1", True, "pattern1", diacritized_text="bait1"),
            BaitValidationResult("bait2", False, "pattern2", diacritized_text="bait2")
        ]
        summary = prosody_validator._generate_validation_summary(1, 1, "طويل", bait_results)
        assert "1 من 2 أبيات صحيحة" in summary
        assert "الأبيات الخاطئة: 2" in summary
        
        # Test mixed with many invalid baits
        summary = prosody_validator._generate_validation_summary(1, 5, "طويل")
        assert "1 من 6 أبيات صحيحة" in summary
        assert "عدد الأبيات الخاطئة: 5" in summary
    
    def test_update_poem_quality(self, prosody_validator, sample_poem):
        """Test poem quality assessment update"""
        # Test with line count issues
        prosody_validator._update_poem_quality(
            sample_poem, 
            line_count_issues=["عدد الأبيات يجب أن يكون زوجياً"]
        )
        
        assert sample_poem.quality is not None
        assert not sample_poem.quality.is_acceptable
        assert len(sample_poem.quality.line_count_issues) > 0
        assert sample_poem.quality.overall_score < 1.0
        
        # Test with prosody issues
        prosody_validator._update_poem_quality(
            sample_poem,
            prosody_issues=["النمط لا يتطابق"]
        )
        
        assert len(sample_poem.quality.prosody_issues) > 0
        assert "النمط لا يتطابق" in sample_poem.quality.prosody_issues
        assert len(sample_poem.quality.recommendations) > 0 

 

 
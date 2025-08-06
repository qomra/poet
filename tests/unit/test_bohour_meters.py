# tests/unit/test_bohour_meters.py

import pytest
from poet.data.bohour_meters import BohourMetersManager, MeterInfo
from poet.models.constraints import Constraints


class TestBohourMetersManager:
    """Unit tests for BohourMetersManager"""
    
    @pytest.fixture
    def meters_manager(self):
        """Create BohourMetersManager instance"""
        return BohourMetersManager()
    
    def test_initialization(self, meters_manager):
        """Test BohourMetersManager initialization"""
        assert meters_manager is not None
        assert len(meters_manager.get_all_meters()) > 0
    
    def test_get_meter_info_arabic_name(self, meters_manager):
        """Test getting meter info by Arabic name"""
        meter_info = meters_manager.get_meter_info("بحر الطويل")
        
        assert meter_info is not None
        assert meter_info.name == "Taweel"
        assert meter_info.arabic_name == "بحر الطويل"
        assert len(meter_info.tafeelat) > 0
        assert "فعولن" in meter_info.tafeelat
        assert "مفاعيلن" in meter_info.tafeelat
    
    def test_get_meter_info_english_name(self, meters_manager):
        """Test getting meter info by English name"""
        meter_info = meters_manager.get_meter_info("Taweel")
        
        assert meter_info is not None
        assert meter_info.name == "Taweel"
        assert meter_info.arabic_name == "بحر الطويل"
        assert len(meter_info.tafeelat) > 0
    
    def test_get_meter_info_unknown_meter(self, meters_manager):
        """Test getting meter info for unknown meter"""
        meter_info = meters_manager.get_meter_info("Unknown Meter")
        
        assert meter_info is None
    
    def test_get_all_meters(self, meters_manager):
        """Test getting all meters"""
        all_meters = meters_manager.get_all_meters()
        
        assert len(all_meters) >= 16  # Should have at least 16 main meters
        assert all(isinstance(meter, MeterInfo) for meter in all_meters)
        
        # Check for some expected meters
        meter_names = [meter.name for meter in all_meters]
        assert "Taweel" in meter_names
        assert "Kamel" in meter_names
        assert "Wafer" in meter_names
    
    def test_search_meters_by_name(self, meters_manager):
        """Test searching meters by name"""
        results = meters_manager.search_meters("طويل")
        
        assert len(results) > 0
        assert any("طويل" in meter.arabic_name for meter in results)
    
    def test_search_meters_by_theme(self, meters_manager):
        """Test searching meters by theme"""
        results = meters_manager.search_meters("غزل")
        
        assert len(results) > 0
        assert any("غزل" in theme for meter in results for theme in meter.common_themes)
    
    def test_get_meters_by_theme(self, meters_manager):
        """Test getting meters by theme"""
        ghazal_meters = meters_manager.get_meters_by_theme("غزل")
        
        assert len(ghazal_meters) > 0
        for meter in ghazal_meters:
            assert any("غزل" in theme.lower() for theme in meter.common_themes)
    
    def test_get_meters_by_difficulty(self, meters_manager):
        """Test getting meters by difficulty"""
        easy_meters = meters_manager.get_meters_by_difficulty("easy")
        medium_meters = meters_manager.get_meters_by_difficulty("medium")
        hard_meters = meters_manager.get_meters_by_difficulty("hard")
        
        assert len(easy_meters) > 0
        assert len(medium_meters) > 0
        assert len(hard_meters) > 0
        
        assert all(meter.difficulty_level == "easy" for meter in easy_meters)
        assert all(meter.difficulty_level == "medium" for meter in medium_meters)
        assert all(meter.difficulty_level == "hard" for meter in hard_meters)
    
    def test_enrich_constraints_with_meter(self, meters_manager):
        """Test enriching constraints with meter tafeelat"""
        original_constraints = Constraints(
            meter="بحر الطويل",
            qafiya="ل",
            line_count=4,
            theme="غزل"
        )
        
        enriched_constraints = meters_manager.enrich_constraints(original_constraints)
        
        assert enriched_constraints.meter == "بحر الطويل"
        assert enriched_constraints.meeter_tafeelat is not None
        assert len(enriched_constraints.meeter_tafeelat) > 0
        assert "فعولن" in enriched_constraints.meeter_tafeelat
        assert "مفاعيلن" in enriched_constraints.meeter_tafeelat
        assert enriched_constraints.qafiya == "ل"
        assert enriched_constraints.line_count == 4
        assert enriched_constraints.theme == "غزل"
    
    def test_enrich_constraints_without_meter(self, meters_manager):
        """Test enriching constraints without meter"""
        original_constraints = Constraints(
            qafiya="ل",
            line_count=4,
            theme="غزل"
        )
        
        enriched_constraints = meters_manager.enrich_constraints(original_constraints)
        
        assert enriched_constraints.meter is None
        assert enriched_constraints.meeter_tafeelat is None
        assert enriched_constraints.qafiya == "ل"
        assert enriched_constraints.line_count == 4
        assert enriched_constraints.theme == "غزل"
    
    def test_enrich_constraints_unknown_meter(self, meters_manager):
        """Test enriching constraints with unknown meter"""
        original_constraints = Constraints(
            meter="بحر غير موجود",
            qafiya="ل",
            line_count=4
        )
        
        enriched_constraints = meters_manager.enrich_constraints(original_constraints)
        
        assert enriched_constraints.meter == "بحر غير موجود"
        assert enriched_constraints.meeter_tafeelat is None  # Should not be enriched
    
    def test_enrich_constraints_with_sub_bahr(self, meters_manager):
        """Test enriching constraints with sub-bahr"""
        original_constraints = Constraints(
            meter="بحر الكامل المجزوء",
            qafiya="ل",
            line_count=4,
            theme="غزل"
        )
        
        enriched_constraints = meters_manager.enrich_constraints(original_constraints)
        
        assert enriched_constraints.meter == "بحر الكامل المجزوء"
        assert enriched_constraints.meeter_tafeelat is not None
        assert len(enriched_constraints.meeter_tafeelat) > 0
        # Sub-bahr should have different tafeelat than main bahr
        assert "متفاعلن" in enriched_constraints.meeter_tafeelat
        assert enriched_constraints.qafiya == "ل"
        assert enriched_constraints.line_count == 4
        assert enriched_constraints.theme == "غزل"
    
    def test_enrich_constraints_with_another_sub_bahr(self, meters_manager):
        """Test enriching constraints with another sub-bahr"""
        original_constraints = Constraints(
            meter="بحر الوافر المجزوء",
            qafiya="ر",
            line_count=2,
            theme="مدح"
        )
        
        enriched_constraints = meters_manager.enrich_constraints(original_constraints)
        
        assert enriched_constraints.meter == "بحر الوافر المجزوء"
        assert enriched_constraints.meeter_tafeelat is not None
        assert len(enriched_constraints.meeter_tafeelat) > 0
        # Should have tafeelat specific to WaferMajzoo
        assert "مفاعلتن" in enriched_constraints.meeter_tafeelat
        assert enriched_constraints.qafiya == "ر"
        assert enriched_constraints.line_count == 2
        assert enriched_constraints.theme == "مدح"
    
    def test_enrich_constraints_sub_bahr_vs_main_bahr(self, meters_manager):
        """Test that sub-bahr and main bahr have different tafeelat"""
        # Main bahr
        main_constraints = Constraints(meter="بحر الكامل")
        main_enriched = meters_manager.enrich_constraints(main_constraints)
        
        # Sub-bahr
        sub_constraints = Constraints(meter="بحر الكامل المجزوء")
        sub_enriched = meters_manager.enrich_constraints(sub_constraints)
        
        # They should have different tafeelat
        assert main_enriched.meeter_tafeelat != sub_enriched.meeter_tafeelat
        
        # Main bahr should have more tafeelat (longer pattern)
        main_tafeelat_count = len(main_enriched.meeter_tafeelat.split())
        sub_tafeelat_count = len(sub_enriched.meeter_tafeelat.split())
        assert main_tafeelat_count > sub_tafeelat_count
    
    def test_suggest_meter_for_theme(self, meters_manager):
        """Test suggesting meters for a theme"""
        suggestions = meters_manager.suggest_meter_for_theme("غزل")
        
        assert len(suggestions) > 0
        for meter in suggestions:
            assert any("غزل" in theme.lower() for theme in meter.common_themes)
    
    def test_suggest_meter_for_theme_with_difficulty(self, meters_manager):
        """Test suggesting meters for a theme with difficulty filter"""
        easy_suggestions = meters_manager.suggest_meter_for_theme("غزل", "easy")
        
        assert len(easy_suggestions) > 0
        for meter in easy_suggestions:
            assert meter.difficulty_level == "easy"
            assert any("غزل" in theme.lower() for theme in meter.common_themes)
    
    def test_get_meter_tafeelat(self, meters_manager):
        """Test getting tafeelat for a meter"""
        tafeelat = meters_manager.get_meter_tafeelat("بحر الطويل")
        
        assert len(tafeelat) > 0
        assert "فعولن" in tafeelat
        assert "مفاعيلن" in tafeelat
    
    def test_get_sub_bahrs(self, meters_manager):
        """Test getting sub-bahrs for a meter"""
        sub_bahrs = meters_manager.get_sub_bahrs("بحر الكامل")
        
        assert len(sub_bahrs) > 0
        assert "بحر الكامل المجزوء" in sub_bahrs
    
    def test_validate_meter(self, meters_manager):
        """Test meter validation"""
        assert meters_manager.validate_meter("بحر الطويل") is True
        assert meters_manager.validate_meter("Taweel") is True
        assert meters_manager.validate_meter("بحر غير موجود") is False
    
    def test_get_meter_examples(self, meters_manager):
        """Test getting meter examples"""
        examples = meters_manager.get_meter_examples("بحر الطويل")
        
        assert len(examples) > 0
        assert all(isinstance(example, str) for example in examples)
    
    def test_get_meter_themes(self, meters_manager):
        """Test getting meter themes"""
        themes = meters_manager.get_meter_themes("بحر الطويل")
        
        assert len(themes) > 0
        assert all(isinstance(theme, str) for theme in themes)
        assert "ملاحم" in themes
    
    def test_meter_info_structure(self, meters_manager):
        """Test MeterInfo structure"""
        meter_info = meters_manager.get_meter_info("بحر الطويل")
        
        assert hasattr(meter_info, 'name')
        assert hasattr(meter_info, 'arabic_name')
        assert hasattr(meter_info, 'tafeelat')
        assert hasattr(meter_info, 'sub_bahrs')
        assert hasattr(meter_info, 'description')
        assert hasattr(meter_info, 'examples')
        assert hasattr(meter_info, 'common_themes')
        assert hasattr(meter_info, 'difficulty_level')
        
        assert isinstance(meter_info.tafeelat, list)
        assert isinstance(meter_info.sub_bahrs, list)
        assert isinstance(meter_info.examples, list)
        assert isinstance(meter_info.common_themes, list)
        assert meter_info.difficulty_level in ["easy", "medium", "hard"]
    
    def test_tafeelat_consistency(self, meters_manager):
        """Test that tafeelat are consistent across meters"""
        # Test a few known meters
        test_cases = [
            ("بحر الطويل", ["فعولن", "مفاعيلن", "فعولن", "مفاعيلن"]),
            ("بحر الكامل", ["متفاعلن", "متفاعلن", "متفاعلن"]),
            ("بحر الوافر", ["مفاعلتن", "مفاعلتن", "مفاعلتن"]),
        ]
        
        for meter_name, expected_tafeelat in test_cases:
            tafeelat = meters_manager.get_meter_tafeelat(meter_name)
            assert len(tafeelat) == len(expected_tafeelat)
            for expected in expected_tafeelat:
                assert expected in tafeelat
    
    def test_sub_bahrs_consistency(self, meters_manager):
        """Test that sub-bahrs are properly identified"""
        # Test meters with known sub-bahrs
        test_cases = [
            ("بحر الكامل", ["بحر الكامل المجزوء"]),
            ("بحر الوافر", ["بحر الوافر المجزوء"]),
            ("بحر الرجز", ["بحر الرجز المجزوء", "بحر الرجز المشطور", "بحر الرجز المنهوك"]),
        ]
        
        for meter_name, expected_sub_bahrs in test_cases:
            sub_bahrs = meters_manager.get_sub_bahrs(meter_name)
            for expected in expected_sub_bahrs:
                assert expected in sub_bahrs 
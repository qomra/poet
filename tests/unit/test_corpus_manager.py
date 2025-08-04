# tests/unit/test_corpus_manager.py

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
from poet.data.corpus_manager import CorpusManager, PoemRecord, SearchCriteria

class TestPoemRecord:
    """Test PoemRecord class"""
    
    def test_from_dict(self):
        """Test creating PoemRecord from dictionary"""
        data = {
            'poem title': 'Test Poem',
            'poem meter': 'بحر الكامل',
            'poem verses': 'بيت أول\nبيت ثاني',
            'poem qafiya': 'ق',
            'poem theme': 'غزل',
            'poem url': 'http://example.com',
            'poet name': 'شاعر تجريبي',
            'poet description': 'شاعر عربي',
            'poet url': 'http://poet.com',
            'poet era': 'عصر حديث',
            'poet location': 'دمشق',
            'poem description': 'قصيدة تجريبية',
            'poem language type': 'فصحى'
        }
        
        poem = PoemRecord.from_dict(data)
        
        assert poem.title == 'Test Poem'
        assert poem.meter == 'بحر الكامل'
        assert poem.verses == 'بيت أول\nبيت ثاني'
        assert poem.qafiya == 'ق'
        assert poem.theme == 'غزل'
        assert poem.poet_name == 'شاعر تجريبي'
    
    def test_from_dict_missing_fields(self):
        """Test creating PoemRecord with missing fields"""
        data = {'poem title': 'Test Only'}
        
        poem = PoemRecord.from_dict(data)
        
        assert poem.title == 'Test Only'
        assert poem.meter == ''
        assert poem.verses == ''
        assert poem.qafiya == ''
        assert poem.theme == ''
    
    def test_matches_meter(self):
        """Test meter matching"""
        poem = PoemRecord(
            title='Test', meter='بحر الكامل', verses='', qafiya='', theme='',
            url='', poet_name='', poet_description='', poet_url='',
            poet_era='', poet_location='', description='', language_type=''
        )
        
        assert poem.matches_meter('الكامل')
        assert poem.matches_meter('بحر الكامل')
        assert not poem.matches_meter('الطويل')
        assert not poem.matches_meter('')
    
    def test_matches_theme(self):
        """Test theme matching"""
        poem = PoemRecord(
            title='Test', meter='', verses='', qafiya='', theme='غزل وحب',
            url='', poet_name='', poet_description='', poet_url='',
            poet_era='', poet_location='', description='', language_type=''
        )
        
        assert poem.matches_theme('غزل')
        assert poem.matches_theme('حب')
        assert not poem.matches_theme('هجاء')
    
    def test_matches_qafiya(self):
        """Test qafiya matching"""
        poem = PoemRecord(
            title='Test', meter='', verses='', qafiya='ق', theme='',
            url='', poet_name='', poet_description='', poet_url='',
            poet_era='', poet_location='', description='', language_type=''
        )
        
        assert poem.matches_qafiya('ق')
        assert not poem.matches_qafiya('ع')
        assert not poem.matches_qafiya('')
    
    def test_matches_poet(self):
        """Test poet matching"""
        poem = PoemRecord(
            title='Test', meter='', verses='', qafiya='', theme='',
            url='', poet_name='أحمد شوقي', poet_description='', poet_url='',
            poet_era='', poet_location='', description='', language_type=''
        )
        
        assert poem.matches_poet('أحمد')
        assert poem.matches_poet('شوقي')
        assert not poem.matches_poet('المتنبي')
    
    def test_get_verse_count(self):
        """Test verse count calculation"""
        # Test with newlines
        poem1 = PoemRecord(
            title='Test', meter='', verses='بيت أول\nبيت ثاني\nبيت ثالث', qafiya='', theme='',
            url='', poet_name='', poet_description='', poet_url='',
            poet_era='', poet_location='', description='', language_type=''
        )
        assert poem1.get_verse_count() == 3
        
        # Test with empty verses
        poem2 = PoemRecord(
            title='Test', meter='', verses='', qafiya='', theme='',
            url='', poet_name='', poet_description='', poet_url='',
            poet_era='', poet_location='', description='', language_type=''
        )
        assert poem2.get_verse_count() == 0
        
        # Test with mixed empty lines
        poem3 = PoemRecord(
            title='Test', meter='', verses='بيت أول\n\nبيت ثاني\n', qafiya='', theme='',
            url='', poet_name='', poet_description='', poet_url='',
            poet_era='', poet_location='', description='', language_type=''
        )
        assert poem3.get_verse_count() == 2

class TestSearchCriteria:
    """Test SearchCriteria class"""
    
    def test_default_values(self):
        """Test default SearchCriteria values"""
        criteria = SearchCriteria()
        
        assert criteria.meter is None
        assert criteria.theme is None
        assert criteria.qafiya is None
        assert criteria.poet_name is None
        assert criteria.keywords is None
    
    def test_with_values(self):
        """Test SearchCriteria with values"""
        criteria = SearchCriteria(
            meter='الكامل',
            theme='غزل',
            qafiya='ق',
            min_verses=2,
            max_verses=5,
            keywords=['حب', 'شوق'],
            search_mode="OR"
        )
        
        assert criteria.meter == 'الكامل'
        assert criteria.theme == 'غزل'
        assert criteria.qafiya == 'ق'
        assert criteria.min_verses == 2
        assert criteria.max_verses == 5
        assert criteria.keywords == ['حب', 'شوق']
        assert criteria.search_mode == "OR"
    
    def test_default_search_mode(self):
        """Test default search mode is AND"""
        criteria = SearchCriteria()
        assert criteria.search_mode == "AND"

class TestCorpusManager:
    """Test CorpusManager class"""
    
    def test_init(self, temp_local_knowledge_dir):
        """Test CorpusManager initialization"""
        with patch('poet.data.corpus_manager.DATASETS_AVAILABLE', True):
            manager = CorpusManager(temp_local_knowledge_dir)
            
            # Use pathlib for cross-platform path handling
            expected_name = Path(temp_local_knowledge_dir).name
            assert manager.local_knowledge_path.name == expected_name
            assert not manager.is_loaded()
    
    def test_load_corpus_success(self, corpus_manager):
        """Test successful corpus loading"""
        corpus_manager.load_corpus()
        
        assert corpus_manager.is_loaded()
        assert len(corpus_manager._poems) == 4  # From sample_corpus_data
        assert corpus_manager._poems[0].title == 'قصيدة الكامل الأولى'
        assert corpus_manager._poems[1].meter == 'بحر الطويل'
    
    def test_load_corpus_no_datasets_library(self, temp_local_knowledge_dir):
        """Test error when datasets library is not available"""
        with patch('poet.data.corpus_manager.DATASETS_AVAILABLE', False):
            with pytest.raises(ImportError, match="datasets library is required"):
                CorpusManager(temp_local_knowledge_dir)
    
    def test_build_indices(self, corpus_manager):
        """Test index building"""
        corpus_manager.load_corpus()
        
        # Check meter index
        assert 'بحر الكامل' in corpus_manager._meter_index
        assert 'بحر الطويل' in corpus_manager._meter_index
        assert len(corpus_manager._meter_index['بحر الكامل']) == 2  # Two poems with this meter
        
        # Check theme index
        assert 'غزل' in corpus_manager._theme_index
        assert 'هجاء' in corpus_manager._theme_index
        assert 'مدح' in corpus_manager._theme_index
        
        # Check poet index
        assert 'ابن المعتز' in corpus_manager._poet_index
        assert 'أحمد شوقي' in corpus_manager._poet_index
        
        # Check qafiya index
        assert 'ق' in corpus_manager._qafiya_index
        assert 'ع' in corpus_manager._qafiya_index
        assert len(corpus_manager._qafiya_index['ق']) == 2  # Two poems with ق qafiya
    
    def test_compute_statistics(self, corpus_manager):
        """Test statistics computation"""
        stats = corpus_manager.get_statistics()
        
        assert stats['total_poems'] == 4
        assert stats['unique_meters'] == 3  # الكامل، الطويل، الوافر
        assert stats['unique_themes'] == 3  # غزل، هجاء، مدح
        assert stats['unique_poets'] == 3  # ابن المعتز، أحمد شوقي، المتنبي
        assert 'بحر الكامل' in stats['meters']
        assert 'غزل' in stats['themes']
        assert 'ق' in stats['qafiya']
        assert 'ع' in stats['qafiya']
    
    def test_search_by_meter(self, corpus_manager):
        """Test searching by meter"""
        criteria = SearchCriteria(meter='الكامل')
        results = corpus_manager.search(criteria)
        
        assert len(results) == 2
        assert all('الكامل' in poem.meter for poem in results)
    
    def test_search_by_theme(self, corpus_manager):
        """Test searching by theme"""
        criteria = SearchCriteria(theme='غزل')
        results = corpus_manager.search(criteria)
        
        assert len(results) == 2
        assert all('غزل' in poem.theme for poem in results)
    
    def test_search_by_poet(self, corpus_manager):
        """Test searching by poet"""
        criteria = SearchCriteria(poet_name='ابن المعتز')
        results = corpus_manager.search(criteria)
        
        assert len(results) == 2
        assert all('ابن المعتز' in poem.poet_name for poem in results)
    
    def test_search_by_qafiya(self, corpus_manager):
        """Test searching by qafiya"""
        criteria = SearchCriteria(qafiya='ق')
        results = corpus_manager.search(criteria)
        
        assert len(results) == 2  # Two poems with ق qafiya
        assert all('ق' in poem.qafiya for poem in results)
    
    def test_search_by_verse_count(self, corpus_manager):
        """Test searching by verse count"""
        criteria = SearchCriteria(min_verses=2, max_verses=2)
        results = corpus_manager.search(criteria)
        
        assert len(results) >= 1
        for poem in results:
            assert poem.get_verse_count() == 2
    
    def test_search_with_keywords(self, corpus_manager):
        """Test searching with keywords"""
        criteria = SearchCriteria(keywords=['الكامل'])
        results = corpus_manager.search(criteria)
        
        assert len(results) >= 1
        for poem in results:
            text_content = f"{poem.title} {poem.verses} {poem.description}".lower()
            assert 'الكامل' in text_content
    
    def test_search_combined_criteria(self, corpus_manager):
        """Test searching with multiple criteria"""
        criteria = SearchCriteria(
            meter='الكامل',
            theme='غزل',
            qafiya='ق'
        )
        results = corpus_manager.search(criteria)
        
        assert len(results) == 2
        for poem in results:
            assert 'الكامل' in poem.meter
            assert 'غزل' in poem.theme
            assert 'ق' in poem.qafiya
    
    def test_search_or_mode(self, corpus_manager):
        """Test OR search mode"""
        criteria = SearchCriteria(
            meter='الطويل',
            theme='غزل',
            search_mode='OR'
        )
        results = corpus_manager.search(criteria)
        
        # Should find poems with either الطويل meter OR غزل theme
        assert len(results) == 3  # 1 طويل + 2 غزل
        
        has_طويل = any('الطويل' in poem.meter for poem in results)
        has_غزل = any('غزل' in poem.theme for poem in results)
        assert has_طويل and has_غزل
    
    def test_search_and_vs_or_mode(self, corpus_manager):
        """Test difference between AND and OR modes"""
        # AND mode - must match both
        and_criteria = SearchCriteria(
            meter='الكامل',
            theme='غزل',
            search_mode='AND'
        )
        and_results = corpus_manager.search(and_criteria)
        
        # OR mode - can match either
        or_criteria = SearchCriteria(
            meter='الكامل',
            theme='هجاء',
            search_mode='OR'
        )
        or_results = corpus_manager.search(or_criteria)
        
        # AND should be more restrictive than OR
        assert len(and_results) <= len(or_results)
    
    def test_search_with_limit(self, corpus_manager):
        """Test search with result limit"""
        criteria = SearchCriteria(meter='الكامل')
        results = corpus_manager.search(criteria, limit=1)
        
        assert len(results) == 1
    
    def test_find_by_meter(self, corpus_manager):
        """Test convenience method find_by_meter"""
        results = corpus_manager.find_by_meter('الكامل')
        
        assert len(results) == 2
        assert all('الكامل' in poem.meter for poem in results)
    
    def test_find_by_theme(self, corpus_manager):
        """Test convenience method find_by_theme"""
        results = corpus_manager.find_by_theme('مدح')
        
        assert len(results) == 1
        assert results[0].theme == 'مدح'
    
    def test_find_by_poet(self, corpus_manager):
        """Test convenience method find_by_poet"""
        results = corpus_manager.find_by_poet('المتنبي')
        
        assert len(results) == 1
        assert results[0].poet_name == 'المتنبي'
    
    def test_find_by_qafiya(self, corpus_manager):
        """Test convenience method find_by_qafiya"""
        results = corpus_manager.find_by_qafiya('ق')
        
        assert len(results) == 2
        assert all('ق' in poem.qafiya for poem in results)
    
    def test_get_examples_for_constraints(self, corpus_manager):
        """Test getting examples for specific constraints"""
        results = corpus_manager.get_examples_for_constraints(
            meter='الكامل',
            theme='غزل',
            qafiya='ق',
            verse_count=2
        )
        
        assert len(results) >= 1
        for poem in results:
            assert 'الكامل' in poem.meter
            assert 'غزل' in poem.theme
            assert 'ق' in poem.qafiya
    
    def test_get_available_meters(self, corpus_manager):
        """Test getting available meters"""
        meters = corpus_manager.get_available_meters()
        
        assert 'بحر الكامل' in meters
        assert 'بحر الطويل' in meters
        assert 'بحر الوافر' in meters
        assert len(meters) == 3
    
    def test_get_available_themes(self, corpus_manager):
        """Test getting available themes"""
        themes = corpus_manager.get_available_themes()
        
        assert 'غزل' in themes
        assert 'هجاء' in themes
        assert 'مدح' in themes
        assert len(themes) == 3
    
    def test_get_available_poets(self, corpus_manager):
        """Test getting available poets"""
        poets = corpus_manager.get_available_poets()
        
        assert 'ابن المعتز' in poets
        assert 'أحمد شوقي' in poets
        assert 'المتنبي' in poets
        assert len(poets) == 3
    
    def test_get_available_qafiya(self, corpus_manager):
        """Test getting available qafiya"""
        qafiyas = corpus_manager.get_available_qafiya()
        
        assert 'ق' in qafiyas
        assert 'ع' in qafiyas
        assert 'د' in qafiyas
        assert len(qafiyas) == 3
    
    def test_sample_random(self, corpus_manager):
        """Test random sampling"""
        sample = corpus_manager.sample_random(2)
        
        assert len(sample) == 2
        assert all(isinstance(poem, PoemRecord) for poem in sample)
        
        # Test sampling more than available
        large_sample = corpus_manager.sample_random(10)
        assert len(large_sample) == 4  # All available poems
    
    def test_validate_meter_exists(self, corpus_manager):
        """Test meter validation"""
        assert corpus_manager.validate_meter_exists('الكامل')
        assert corpus_manager.validate_meter_exists('الطويل')
        assert not corpus_manager.validate_meter_exists('الرجز')  # Not in test data
    
    def test_validate_theme_exists(self, corpus_manager):
        """Test theme validation"""
        assert corpus_manager.validate_theme_exists('غزل')
        assert corpus_manager.validate_theme_exists('مدح')
        assert not corpus_manager.validate_theme_exists('رثاء')  # Not in test data
    
    def test_validate_qafiya_exists(self, corpus_manager):
        """Test qafiya validation"""
        assert corpus_manager.validate_qafiya_exists('ق')
        assert corpus_manager.validate_qafiya_exists('ع')
        assert not corpus_manager.validate_qafiya_exists('ز')  # Not in test data
    
    def test_get_meter_variations(self, corpus_manager):
        """Test getting meter variations"""
        variations = corpus_manager.get_meter_variations('الكامل')
        
        assert 'بحر الكامل' in variations
        assert len(variations) == 1
    
    def test_get_theme_variations(self, corpus_manager):
        """Test getting theme variations"""
        variations = corpus_manager.get_theme_variations('غزل')
        
        assert 'غزل' in variations
        assert len(variations) == 1
    
    def test_force_reload(self, corpus_manager):
        """Test force reloading corpus"""
        # Load once
        corpus_manager.load_corpus()
        initial_count = len(corpus_manager._poems)
        
        # Force reload
        corpus_manager.load_corpus(force_reload=True)
        assert len(corpus_manager._poems) == initial_count
        assert corpus_manager.is_loaded()
    
    def test_get_total_poems(self, corpus_manager):
        """Test getting total poem count"""
        total = corpus_manager.get_total_poems()
        assert total == 4
    
    def test_get_raw_dataset(self, corpus_manager):
        """Test getting raw dataset"""
        dataset = corpus_manager.get_raw_dataset()
        assert dataset is not None 
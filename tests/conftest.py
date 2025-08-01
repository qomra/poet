# tests/conftest.py
import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from poet.prompts.prompt_manager import PromptManager
from poet.llm.base_llm import MockLLM, LLMConfig

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path"""
    return Path(__file__).parent.parent

@pytest.fixture
def prompt_manager():
    """Create a PromptManager instance using the actual templates directory"""
    return PromptManager()

@pytest.fixture
def mock_llm():
    """Mock LLM provider using MockLLM"""
    config = LLMConfig(model_name="test-model")
    return MockLLM(config)

@pytest.fixture
def test_data():
    """Load test data from fixtures"""
    test_file = Path(__file__).parent / "fixtures" / "test_data.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        return json.load(f)

# Corpus Manager Fixtures
@pytest.fixture(scope="session")
def sample_corpus_data():
    """Sample corpus data for testing across all test files"""
    return [
        {
            'poem title': 'قصيدة الكامل الأولى',
            'poem meter': 'بحر الكامل',
            'poem verses': 'وَمُتَيَّمٍ جَرَحَ الفُراقُ فُؤادَهُ\nفَالدَمعُ مِن أَجفانِهِ يَتَدَفَّقُ',
            'poem theme': 'غزل',
            'poem url': 'http://example.com/1',
            'poet name': 'ابن المعتز',
            'poet description': 'شاعر عباسي',
            'poet url': 'http://poet1.com',
            'poet era': 'العصر العباسي',
            'poet location': 'بغداد',
            'poem description': 'قصيدة غزلية في الفراق',
            'poem language type': 'فصحى'
        },
        {
            'poem title': 'قصيدة الطويل في الهجاء',
            'poem meter': 'بحر الطويل',
            'poem verses': 'تَمَكَّنَ هَذا الدَهرُ مِمّا يَسوءُني\nوَلَجَّ فَما يَخلي صَفاتِيَ مِن قَرعِ\nوَأَبلَيتُ آمالي بِوَصلٍ يَكُدُّها',
            'poem theme': 'هجاء',
            'poem url': 'http://example.com/2',
            'poet name': 'ابن المعتز',
            'poet description': 'شاعر عباسي',
            'poet url': 'http://poet1.com',
            'poet era': 'العصر العباسي',
            'poet location': 'بغداد',
            'poem description': 'قصيدة هجاء الزمن',
            'poem language type': 'فصحى'
        },
        {
            'poem title': 'قصيدة الكامل الثانية',
            'poem meter': 'بحر الكامل',
            'poem verses': 'بَهَرَتهُ ساعَةُ فِرقَةٍ فَكَأَنَّما\nفي كُلِّ عُضوٍ مِنهُ قَلبٌ يَخفِقُ',
            'poem theme': 'غزل',
            'poem url': 'http://example.com/3',
            'poet name': 'أحمد شوقي',
            'poet description': 'أمير الشعراء',
            'poet url': 'http://poet2.com',
            'poet era': 'عصر النهضة',
            'poet location': 'مصر',
            'poem description': 'قصيدة غزلية',
            'poem language type': 'فصحى'
        },
        {
            'poem title': 'قصيدة مدح',
            'poem meter': 'بحر الوافر',
            'poem verses': 'يا أيها الملك المعظم شأنه\nفي كل أرض ذكره يتردد',
            'poem theme': 'مدح',
            'poem url': 'http://example.com/4',
            'poet name': 'المتنبي',
            'poet description': 'شاعر عظيم',
            'poet url': 'http://poet3.com',
            'poet era': 'العصر العباسي',
            'poet location': 'الكوفة',
            'poem description': 'قصيدة مدح',
            'poem language type': 'فصحى'
        }
    ]

@pytest.fixture
def mock_dataset(sample_corpus_data):
    """Create a mock HuggingFace dataset for testing"""
    # Create a mock dataset that behaves like HuggingFace Dataset
    mock_dataset = Mock()
    mock_dataset.__iter__ = lambda self: iter(sample_corpus_data)
    mock_dataset.__len__ = lambda self: len(sample_corpus_data)
    return mock_dataset

@pytest.fixture
def temp_local_knowledge_dir(sample_corpus_data, mock_dataset):
    """Create temporary local knowledge directory with mock ashaar dataset"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create ashaar directory structure
        ashaar_dir = temp_path / "ashaar"
        ashaar_dir.mkdir()
        
        # Mock the load_dataset function to return our test data
        with patch('poet.data.corpus_manager.load_dataset') as mock_load:
            mock_dataset_dict = {'train': mock_dataset}
            mock_load.return_value = mock_dataset_dict
            
            yield str(temp_path)

@pytest.fixture
def corpus_manager(temp_local_knowledge_dir):
    """Create CorpusManager instance with mocked dataset"""
    with patch('poet.data.corpus_manager.DATASETS_AVAILABLE', True):
        from poet.data.corpus_manager import CorpusManager
        return CorpusManager(temp_local_knowledge_dir)

@pytest.fixture(scope="session")
def session_mock_dataset(sample_corpus_data):
    """Create a session-scoped mock dataset"""
    mock_dataset = Mock()
    mock_dataset.__iter__ = lambda self: iter(sample_corpus_data)
    mock_dataset.__len__ = lambda self: len(sample_corpus_data)
    return mock_dataset

@pytest.fixture(scope="session")
def session_temp_local_knowledge_dir(sample_corpus_data, session_mock_dataset):
    """Create temporary local knowledge directory (session scope)"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create ashaar directory structure
        ashaar_dir = temp_path / "ashaar"
        ashaar_dir.mkdir()
        
        # Mock the load_dataset function to return our test data
        with patch('poet.data.corpus_manager.load_dataset') as mock_load:
            mock_dataset_dict = {'train': session_mock_dataset}
            mock_load.return_value = mock_dataset_dict
            
            yield str(temp_path)

@pytest.fixture(scope="session")
def session_corpus_manager(session_temp_local_knowledge_dir):
    """Create CorpusManager instance (session scope for performance)"""
    with patch('poet.data.corpus_manager.DATASETS_AVAILABLE', True):
        from poet.data.corpus_manager import CorpusManager
        return CorpusManager(session_temp_local_knowledge_dir)

# Real data fixtures for integration tests
@pytest.fixture(scope="session")
def real_corpus_manager():
    """Create CorpusManager with real dataset"""
    from pathlib import Path
    from poet.data.corpus_manager import CorpusManager
    
    kb_path = Path("kb")
    if not kb_path.exists():
        pytest.skip("Real kb/ directory not found")
        
    return CorpusManager(local_knowledge_path=kb_path)

@pytest.fixture(scope="session")
def real_knowledge_retriever(real_corpus_manager):
    """Create KnowledgeRetriever with real corpus"""
    from poet.analysis.knowledge_retriever import KnowledgeRetriever
    
    return KnowledgeRetriever(real_corpus_manager)

# LLM parametrization fixtures
@pytest.fixture(params=["mock", "real"])
def llm_type(request):
    """Parametrize tests to run with both mock and real LLMs"""
    return request.param

@pytest.fixture
def real_llm():
    """Real LLM instance if available, otherwise None"""
    from poet.llm.llm_factory import get_real_llm_from_env
    return get_real_llm_from_env()

@pytest.fixture
def constraint_parser_parametrized(llm_type, mock_llm, real_llm, prompt_manager):
    """ConstraintParser instance - mock or real based on parameter"""
    from poet.analysis.constraint_parser import ConstraintParser
    
    if llm_type == "mock":
        return ConstraintParser(mock_llm, prompt_manager)
    elif llm_type == "real":
        if real_llm is None:
            pytest.skip("Real LLM not available")
        return ConstraintParser(real_llm, prompt_manager)
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}")
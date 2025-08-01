# tests/conftest.py
import pytest
import sys
import json
from pathlib import Path
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
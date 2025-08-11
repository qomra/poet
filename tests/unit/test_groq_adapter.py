# tests/unit/test_groq_adapter.py

import os
import json
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from poet.llm.groq_adapter import GroqAdapter
from poet.llm.base_llm import LLMConfig, LLMError

# Skip all tests in this file if TEST_REAL_LLMS is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_REAL_LLMS"),
    reason="Real LLM tests require TEST_REAL_LLMS environment variable"
)

def handle_rate_limit_error(e: Exception) -> None:
    """Handle rate limit errors by skipping the test."""
    if (hasattr(e, 'status_code') and e.status_code == 429) or (
        "rate limit" in str(e).lower() or
        "429" in str(e) or
        "insufficient_quota" in str(e).lower() or
        "quota" in str(e).lower()):
        pytest.skip(f"Rate limit or quota exceeded: {e}")
    raise e

@pytest.fixture(scope="module")
def llm_configs():
    """Load LLM configurations from fixtures."""
    # First try llms.json, then fall back to llms_example.json
    config_files = ["llms.json", "llms_example.json"]
    
    for config_file in config_files:
        config_path = Path(__file__).parent.parent / "fixtures" / config_file
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    
    pytest.skip("No LLM configuration file found (llms.json or llms_example.json)")

@pytest.fixture(scope="module")
def groq_llm(llm_configs):
    """Create Groq LLM instance if configured."""
    if "groq" not in llm_configs:
        pytest.skip("Groq configuration not found")
    
    config_data = llm_configs["groq"]
    
    # Skip if API key is not set (placeholder)
    api_key = config_data.get("api_key", "")
    
    # Override with environment variable if available
    if os.getenv("GROQ_API_KEY"):
        api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key or api_key == "sk-" or api_key == "your-groq-api-key-here":
        pytest.skip("Groq API key not configured")
    
    config = LLMConfig(
        model_name=config_data["model"],
        api_key=api_key,
        base_url=None,  # Don't use custom base_url for Groq
        timeout=config_data.get("timeout", 30)
    )
    
    return GroqAdapter(config)

class TestGroqAdapter:
    """Test cases for GroqAdapter."""
    
    def test_init_without_groq_package(self):
        """Test initialization fails when groq package is not available."""
        with patch('poet.llm.groq_adapter.GROQ_AVAILABLE', False):
            with pytest.raises(LLMError, match="Groq package not installed"):
                GroqAdapter(LLMConfig(model_name="test-model"))
    
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with patch('poet.llm.groq_adapter.GROQ_AVAILABLE', True):
            with pytest.raises(LLMError, match="Groq API key is required"):
                GroqAdapter(LLMConfig(model_name="test-model"))
    
    @patch('poet.llm.groq_adapter.groq')
    def test_init_success(self, mock_groq):
        """Test successful initialization."""
        mock_groq.Groq.return_value = Mock()
        
        config = LLMConfig(
            model_name="llama3-8b-8192",
            api_key="test-key",
            timeout=30
        )
        
        adapter = GroqAdapter(config)
        assert adapter.config == config
        mock_groq.Groq.assert_called_once_with(
            api_key="test-key",
            base_url=None,
            timeout=30
        )
    
    @patch('poet.llm.groq_adapter.groq')
    def test_generate(self, mock_groq):
        """Test generate method."""
        mock_client = Mock()
        mock_groq.Groq.return_value = mock_client
        
        config = LLMConfig(
            model_name="llama3-8b-8192",
            api_key="test-key"
        )
        
        adapter = GroqAdapter(config)
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "llama3-8b-8192"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.id = "test-id"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        result = adapter.generate("Test prompt")
        assert result == "Test response"
    
    @patch('poet.llm.groq_adapter.groq')
    def test_generate_with_metadata(self, mock_groq):
        """Test generate_with_metadata method."""
        mock_client = Mock()
        mock_groq.Groq.return_value = mock_client
        
        config = LLMConfig(
            model_name="llama3-8b-8192",
            api_key="test-key"
        )
        
        adapter = GroqAdapter(config)
        
        # Mock response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "llama3-8b-8192"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.id = "test-id"
        
        mock_client.chat.completions.create.return_value = mock_response
        
        result = adapter.generate_with_metadata("Test prompt")
        assert result.content == "Test response"
        assert result.model == "llama3-8b-8192"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5
        assert result.finish_reason == "stop"
    
    @patch('poet.llm.groq_adapter.groq')
    def test_is_available(self, mock_groq):
        """Test is_available method."""
        mock_client = Mock()
        mock_groq.Groq.return_value = mock_client
        
        config = LLMConfig(
            model_name="llama3-8b-8192",
            api_key="test-key"
        )
        
        adapter = GroqAdapter(config)
        
        # Mock successful response
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        
        assert adapter.is_available() is True
        
        # Mock failed response
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        assert adapter.is_available() is False
    
    @patch('poet.llm.groq_adapter.groq')
    def test_get_model_info(self, mock_groq):
        """Test get_model_info method."""
        mock_client = Mock()
        mock_groq.Groq.return_value = mock_client
        
        config = LLMConfig(
            model_name="llama3-8b-8192",
            api_key="test-key"
        )
        
        adapter = GroqAdapter(config)
        
        info = adapter.get_model_info()
        assert info["provider"] == "groq"
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is False
        assert info["context_length"] == 8192
    
    def test_get_context_length(self):
        """Test _get_context_length method."""
        with patch('poet.llm.groq_adapter.GROQ_AVAILABLE', True):
            config = LLMConfig(
                model_name="llama3-8b-8192",
                api_key="test-key"
            )
            
            adapter = GroqAdapter(config)
            
            # Test known models
            assert adapter._get_context_length("llama3-8b-8192") == 8192
            assert adapter._get_context_length("mixtral-8x7b-32768") == 32768
            
            # Test unknown model (should return default)
            assert adapter._get_context_length("unknown-model") == 8192


@pytest.mark.real_llm
class TestGroqAdapterRealAPI:
    """Test Groq LLM adapter with real API calls."""
    
    def test_basic_generation(self, groq_llm):
        """Test basic text generation."""
        prompt = "What is poetry? Answer in one sentence."
        
        try:
            response = groq_llm.generate(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "poetry" in response.lower() or "poem" in response.lower()
        except Exception as e:
            handle_rate_limit_error(e)
    
    def test_generation_with_metadata(self, groq_llm):
        """Test generation with full metadata."""
        prompt = "Define Arabic poetry in Arabic. Answer in one sentence."
        
        try:
            response = groq_llm.generate_with_metadata(prompt)
            
            # Check response structure
            assert hasattr(response, 'content')
            assert hasattr(response, 'model')
            assert hasattr(response, 'usage')
            assert hasattr(response, 'metadata')
            
            # Check content
            assert isinstance(response.content, str)
            assert len(response.content) > 0
            
            # Check usage info
            if response.usage:
                assert 'prompt_tokens' in response.usage
                assert 'completion_tokens' in response.usage
                assert 'total_tokens' in response.usage
                assert response.usage['total_tokens'] > 0
            
            # Check metadata
            assert 'response_time' in response.metadata
            assert response.metadata['response_time'] > 0
            assert 'model' in response.metadata
        except Exception as e:
            handle_rate_limit_error(e)
    
    def test_is_available(self, groq_llm):
        """Test availability check."""
        # This makes a real API call, so it should work if credentials are valid
        result = groq_llm.is_available()
        
        if result is False:
            pytest.skip("Groq service is not available")
        
        assert result is True
    
    def test_get_model_info(self, groq_llm):
        """Test model info retrieval."""
        info = groq_llm.get_model_info()
        
        assert info["provider"] == "groq"
        assert info["supports_streaming"] is True
        assert info["supports_functions"] is False
        assert "context_length" in info
        assert info["context_length"] > 0
    
    def test_error_handling(self, groq_llm):
        """Test error handling with invalid requests."""
        # Test with empty prompt
        try:
            response = groq_llm.generate("")
            # Some models might accept empty prompts, so we don't assert failure
        except Exception as e:
            # If it fails, it should be a proper LLM error
            assert "LLM" in str(type(e).__name__)
    
    @pytest.mark.slow
    def test_response_time(self, groq_llm):
        """Test response time performance."""
        prompt = "Write a short haiku about technology."
        
        try:
            start_time = time.time()
            response = groq_llm.generate(prompt)
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Groq is known for fast responses, should be under 10 seconds
            assert response_time < 10.0
            assert len(response) > 0
            
        except Exception as e:
            handle_rate_limit_error(e)
    
    @pytest.mark.slow
    def test_multiple_requests(self, groq_llm):
        """Test multiple sequential requests."""
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "What is the meaning of life?"
        ]
        
        responses = []
        
        try:
            for prompt in prompts:
                response = groq_llm.generate(prompt)
                responses.append(response)
                assert isinstance(response, str)
                assert len(response) > 0
            
            # All responses should be different
            assert len(set(responses)) == len(prompts)
            
        except Exception as e:
            handle_rate_limit_error(e)

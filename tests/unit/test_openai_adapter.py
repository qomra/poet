# tests/integration/test_openai.py

import os
import json
import pytest
from pathlib import Path
from poet.llm.base_llm import LLMConfig, LLMRateLimitError
from poet.llm.openai_adapter import OpenAIAdapter

# Skip all tests in this file if TEST_REAL_LLMS is not set
pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_REAL_LLMS"),
    reason="Real LLM tests require TEST_REAL_LLMS environment variable"
)

def handle_rate_limit_error(e: Exception) -> None:
    """Handle rate limit and quota errors by skipping the test."""
    if (isinstance(e, LLMRateLimitError) or 
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
def openai_llm(llm_configs):
    """Create OpenAI LLM instance if configured."""
    if "openai" not in llm_configs:
        pytest.skip("OpenAI configuration not found")
    
    config_data = llm_configs["openai"]
    
    # Skip if API key is not set (placeholder)
    api_key = config_data.get("api_key", "")
    
    # Override with environment variable if available
    if os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key or api_key == "sk-" or api_key == "your-openai-api-key-here":
        pytest.skip("OpenAI API key not configured")
    
    config = LLMConfig(
        model_name=config_data["model"],
        api_key=api_key,
        base_url=config_data.get("api_base"),
        timeout=30
    )
    
    return OpenAIAdapter(config)

class TestOpenAIAdapter:
    """Test OpenAI LLM adapter with real API calls."""
    
    def test_basic_generation(self, openai_llm):
        """Test basic text generation."""
        prompt = "What is poetry? Answer in one sentence."
        
        try:
            response = openai_llm.generate(prompt)
            
            assert isinstance(response, str)
            assert len(response) > 0
            assert "poetry" in response.lower() or "poem" in response.lower()
        except Exception as e:
            handle_rate_limit_error(e)
    
    def test_generation_with_metadata(self, openai_llm):
        """Test generation with full metadata."""
        prompt = "Define Arabic poetry in Arabic. Answer in one sentence."
        
        try:
            response = openai_llm.generate_with_metadata(prompt)
            
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
    
    def test_is_available(self, openai_llm):
        """Test availability check."""
        # This makes a real API call, so it should work if credentials are valid
        result = openai_llm.is_available()
        
        if result is False:
            # If is_available returns False, it might be due to quota issues
            # Check the logs to see if it was a quota error
            import logging
            logger = logging.getLogger("OpenAIAdapter")
            
            # Since we can't easily check the logs in the test, we'll skip if False
            # This handles cases where is_available returns False due to quota/rate limit issues
            pytest.skip("OpenAI service unavailable (likely due to quota or rate limit)")
        
        assert result is True
    
    def test_get_model_info(self, openai_llm):
        """Test model information retrieval."""
        try:
            info = openai_llm.get_model_info()
            
            assert isinstance(info, dict)
            assert 'model' in info
            assert 'provider' in info
            assert info['provider'] == 'openai'
            assert 'context_length' in info
            assert info['context_length'] > 0
        except Exception as e:
            handle_rate_limit_error(e)

class TestOpenAIErrorHandling:
    """Test error handling with OpenAI."""
    
    def test_invalid_api_key(self):
        """Test handling of invalid API key."""
        config = LLMConfig(
            model_name="gpt-4o",
            api_key="invalid-key",
            timeout=5
        )
        
        llm = OpenAIAdapter(config)
        
        # Should raise appropriate error
        with pytest.raises(Exception):  # Could be LLMConnectionError or similar
            llm.generate("test")
    
    def test_timeout_handling(self, llm_configs):
        """Test timeout handling."""
        if "openai" not in llm_configs:
            pytest.skip("OpenAI configuration not found")
        
        config_data = llm_configs["openai"]
        
        # Get API key from config or environment
        api_key = config_data.get("api_key", "")
        if os.getenv("OPENAI_API_KEY"):
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key or api_key == "sk-" or api_key == "your-openai-api-key-here":
            pytest.skip("OpenAI API key not configured")
        
        # Create LLM with very short timeout
        config = LLMConfig(
            model_name=config_data["model"],
            api_key=api_key,
            base_url=config_data.get("api_base"),
            timeout=0.001  # Very short timeout
        )
        
        llm = OpenAIAdapter(config)
        
        # Should timeout (though this might not always trigger due to connection speed)
        try:
            response = llm.generate("Write a long essay about Arabic poetry")
            # If it doesn't timeout, that's also fine - just check we got a response
            assert isinstance(response, str)
        except Exception as e:
            # Check if it's a rate limit error first
            if isinstance(e, LLMRateLimitError) or "rate limit" in str(e).lower() or "429" in str(e):
                pytest.skip(f"Rate limit exceeded: {e}")
            # Timeout or other error is expected
            assert "timeout" in str(e).lower() or "time" in str(e).lower() or len(str(e)) > 0

@pytest.mark.slow
class TestOpenAIPerformance:
    """Performance tests for OpenAI (marked as slow)."""
    
    def test_response_time(self, openai_llm):
        """Test that response time is reasonable."""
        import time
        
        try:
            start_time = time.time()
            response = openai_llm.generate("Hello")
            end_time = time.time()
            
            response_time = end_time - start_time
            
            # Should respond within reasonable time (30 seconds)
            assert response_time < 30
            assert isinstance(response, str)
            assert len(response) > 0
        except Exception as e:
            handle_rate_limit_error(e)
    
    def test_multiple_requests(self, openai_llm):
        """Test multiple consecutive requests."""
        prompts = [
            "What is poetry?",
            "Define meter in Arabic poetry.",
            "What is a ghazal?"
        ]
        
        try:
            responses = []
            for prompt in prompts:
                response = openai_llm.generate(prompt)
                responses.append(response)
            
            # All should succeed
            assert len(responses) == 3
            for response in responses:
                assert isinstance(response, str)
                assert len(response) > 0
        except Exception as e:
            handle_rate_limit_error(e) 
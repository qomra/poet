# tests/unit/test_anthropic_adapter.py

import pytest
from unittest.mock import Mock, patch
from poet.llm.anthropic_adapter import AnthropicAdapter, ANTHROPIC_AVAILABLE
from poet.llm.base_llm import LLMConfig, LLMError


class TestAnthropicAdapter:
    """Test cases for AnthropicAdapter"""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock LLM config"""
        return LLMConfig(
            model_name="claude-3-5-sonnet-20241022",
            api_key="test-api-key",
            base_url="https://api.anthropic.com",
            timeout=30
        )
    
    def test_initialization_without_anthropic_package(self):
        """Test initialization when anthropic package is not available"""
        with patch('poet.llm.anthropic_adapter.ANTHROPIC_AVAILABLE', False):
            config = LLMConfig(
                model_name="claude-3-5-sonnet-20241022",
                api_key="test-key"
            )
            with pytest.raises(LLMError, match="Anthropic package not installed"):
                AnthropicAdapter(config)
    
    def test_initialization_without_api_key(self):
        """Test initialization without API key"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        config = LLMConfig(
            model_name="claude-3-5-sonnet-20241022",
            api_key=None
        )
        with pytest.raises(LLMError, match="Anthropic API key is required"):
            AnthropicAdapter(config)
    
    @patch('poet.llm.anthropic_adapter.anthropic')
    def test_initialization_success(self, mock_anthropic, mock_config):
        """Test successful initialization"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        adapter = AnthropicAdapter(mock_config)
        
        assert adapter.config == mock_config
        mock_anthropic.Anthropic.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://api.anthropic.com",
            timeout=30
        )
    
    @patch('poet.llm.anthropic_adapter.anthropic')
    def test_generate_success(self, mock_anthropic, mock_config):
        """Test successful generation"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_response.id = "test-id"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.stop_sequence = None
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        adapter = AnthropicAdapter(mock_config)
        result = adapter.generate("Test prompt")
        
        assert result == "Test response"
        mock_client.messages.create.assert_called_once()
    
    @patch('poet.llm.anthropic_adapter.anthropic')
    def test_generate_with_metadata(self, mock_anthropic, mock_config):
        """Test generation with metadata"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        # Mock response
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response")]
        mock_response.model = "claude-3-5-sonnet-20241022"
        mock_response.usage = Mock(input_tokens=10, output_tokens=5)
        mock_response.stop_reason = "end_turn"
        mock_response.id = "test-id"
        mock_response.type = "message"
        mock_response.role = "assistant"
        mock_response.stop_sequence = None
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        adapter = AnthropicAdapter(mock_config)
        result = adapter.generate_with_metadata("Test prompt")
        
        assert result.content == "Test response"
        assert result.model == "claude-3-5-sonnet-20241022"
        assert result.finish_reason == "end_turn"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5
        assert result.usage["total_tokens"] == 15
    
    @patch('poet.llm.anthropic_adapter.anthropic')
    def test_is_available_success(self, mock_anthropic, mock_config):
        """Test availability check success"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        mock_response = Mock()
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client
        
        adapter = AnthropicAdapter(mock_config)
        assert adapter.is_available() is True
    
    @patch('poet.llm.anthropic_adapter.anthropic')
    def test_is_available_failure(self, mock_anthropic, mock_config):
        """Test availability check failure"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.Anthropic.return_value = mock_client
        
        adapter = AnthropicAdapter(mock_config)
        assert adapter.is_available() is False
    
    def test_get_model_info(self, mock_config):
        """Test model info retrieval"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        with patch('poet.llm.anthropic_adapter.anthropic'):
            adapter = AnthropicAdapter(mock_config)
            info = adapter.get_model_info()
            
            assert info["provider"] == "anthropic"
            assert info["supports_streaming"] is True
            assert info["supports_functions"] is True
            assert info["context_length"] == 200000  # For claude-3-5-sonnet
    
    def test_get_context_length(self, mock_config):
        """Test context length retrieval for different models"""
        if not ANTHROPIC_AVAILABLE:
            pytest.skip("Anthropic package not available")
        
        with patch('poet.llm.anthropic_adapter.anthropic'):
            adapter = AnthropicAdapter(mock_config)
            
            # Test exact matches
            assert adapter._get_context_length("claude-3-opus-20240229") == 200000
            assert adapter._get_context_length("claude-2.0") == 100000
            
            # Test partial matches
            assert adapter._get_context_length("claude-3-opus-20240229-custom") == 200000
            
            # Test fallback
            assert adapter._get_context_length("unknown-model") == 100000 
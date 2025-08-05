# tests/unit/test_search_provider.py

import pytest
import os
import json
from unittest.mock import Mock, patch, MagicMock
from poet.data.search_provider import (
    SearchResult, 
    SearchResponse, 
    BaseSearchProvider, 
    SerpSearchProvider, 
    SearchProviderFactory
)


class TestSearchResult:
    """Test SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test creating SearchResult instance"""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="example.com",
            metadata={"key": "value"}
        )
        
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.source == "example.com"
        assert result.metadata["key"] == "value"
    
    def test_search_result_default_metadata(self):
        """Test SearchResult with default metadata"""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            snippet="Test",
            source="example.com"
        )
        
        assert result.metadata == {}


class TestSearchResponse:
    """Test SearchResponse dataclass"""
    
    def test_search_response_creation(self):
        """Test creating SearchResponse instance"""
        results = [
            SearchResult("Title 1", "https://1.com", "Snippet 1", "1.com", {}),
            SearchResult("Title 2", "https://2.com", "Snippet 2", "2.com", {})
        ]
        
        response = SearchResponse(
            results=results,
            total_results=2,
            search_time=1.5,
            query="test query",
            provider="test_provider",
            metadata={"key": "value"}
        )
        
        assert len(response.results) == 2
        assert response.total_results == 2
        assert response.search_time == 1.5
        assert response.query == "test query"
        assert response.provider == "test_provider"
        assert response.metadata["key"] == "value"


class TestBaseSearchProvider:
    """Test BaseSearchProvider abstract class"""
    
    def test_base_provider_initialization(self):
        """Test BaseSearchProvider initialization"""
        config = {"test": "config"}
        
        # Create a concrete implementation for testing
        class TestProvider(BaseSearchProvider):
            def _make_request(self, query, max_results=10, **kwargs):
                return {"test": "response"}
            
            def is_available(self):
                return True
        
        provider = TestProvider(config)
        
        assert provider.config == config
        assert provider.logger is not None
    
    def test_validate_config_default(self):
        """Test default validate_config implementation"""
        class TestProvider(BaseSearchProvider):
            def _make_request(self, query, max_results=10, **kwargs):
                return {"test": "response"}
            
            def is_available(self):
                return True
        
        provider = TestProvider({})
        assert provider.validate_config() is True
    
    def test_search_success(self):
        """Test successful search operation"""
        class TestProvider(BaseSearchProvider):
            def _make_request(self, query, max_results=10, **kwargs):
                return {
                    "organic_results": [
                        {
                            "title": "Test Result",
                            "link": "https://test.com",
                            "snippet": "Test snippet",
                            "source": "test.com"
                        }
                    ]
                }
            
            def is_available(self):
                return True
        
        provider = TestProvider({})
        response = provider.search("test query")
        
        assert isinstance(response, SearchResponse)
        assert len(response.results) == 1
        assert response.results[0].title == "Test Result"
        assert response.query == "test query"
        assert response.provider == "TestProvider"
        assert response.search_time > 0
    
    def test_search_no_response(self):
        """Test search when provider returns no response"""
        class TestProvider(BaseSearchProvider):
            def _make_request(self, query, max_results=10, **kwargs):
                return None
            
            def is_available(self):
                return True
        
        provider = TestProvider({})
        response = provider.search("test query")
        
        assert len(response.results) == 0
        assert response.total_results == 0
        assert "error" in response.metadata
    
    def test_search_exception_handling(self):
        """Test search exception handling"""
        class TestProvider(BaseSearchProvider):
            def _make_request(self, query, max_results=10, **kwargs):
                raise Exception("Test error")
            
            def is_available(self):
                return True
        
        provider = TestProvider({})
        response = provider.search("test query")
        
        assert len(response.results) == 0
        assert "error" in response.metadata
        assert "Test error" in response.metadata["error"]
    
    def test_parse_results_default(self):
        """Test default _parse_results implementation"""
        class TestProvider(BaseSearchProvider):
            def _make_request(self, query, max_results=10, **kwargs):
                return {"test": "response"}
            
            def is_available(self):
                return True
        
        provider = TestProvider({})
        
        # Test with valid organic results
        response_data = {
            "organic_results": [
                {
                    "title": "Test Title",
                    "link": "https://test.com",
                    "snippet": "Test snippet",
                    "source": "test.com",
                    "position": 1
                }
            ]
        }
        
        results = provider._parse_results(response_data)
        assert len(results) == 1
        assert results[0].title == "Test Title"
        assert results[0].url == "https://test.com"
        
        # Test with missing organic results
        empty_results = provider._parse_results({})
        assert len(empty_results) == 0
        
        # Test with malformed result
        malformed_data = {
            "organic_results": [
                {"title": "Test"}  # Missing required fields
            ]
        }
        
        malformed_results = provider._parse_results(malformed_data)
        assert len(malformed_results) == 0


class TestSerpSearchProvider:
    """Test SerpSearchProvider class"""
    
    def test_serper_initialization(self):
        """Test SerpSearchProvider initialization"""
        config = {
            "api_key": "sk-test-key",
            "engine": "google",
            "location": "United States",
            "language": "en"
        }
        
        provider = SerpSearchProvider(config)
        
        assert provider.api_key == "sk-test-key"
        assert provider.engine == "google"
        assert provider.location == "United States"
        assert provider.language == "en"
        assert provider.base_url == "https://serpapi.com/search"
    
    def test_serper_initialization_missing_api_key(self):
        """Test SerpSearchProvider initialization without API key"""
        config = {"engine": "google"}
        
        with pytest.raises(ValueError, match="SerpAPI API key is required"):
            SerpSearchProvider(config)
    
    def test_serper_validate_config_success(self):
        """Test successful config validation"""
        config = {"api_key": "sk-valid-key"}
        provider = SerpSearchProvider(config)
        
        assert provider.validate_config() is True
    
    def test_serper_validate_config_missing_key(self):
        """Test config validation with missing API key"""
        # This should raise ValueError during initialization, not during validate_config
        with pytest.raises(ValueError, match="SerpAPI API key is required"):
            SerpSearchProvider({})
    
    def test_serper_validate_config_invalid_format(self):
        """Test config validation with invalid API key format"""
        config = {"api_key": "invalid-key"}
        provider = SerpSearchProvider(config)
        
        # Should still return True but log warning
        assert provider.validate_config() is True
    
    @patch('poet.data.search_provider.requests.get')
    def test_serper_make_request_success(self, mock_get):
        """Test successful SerpAPI request"""
        # Mock successful response
        mock_response = Mock()
        mock_response.json.return_value = {
            "organic_results": [
                {
                    "title": "Test Result",
                    "link": "https://test.com",
                    "snippet": "Test snippet"
                }
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        config = {"api_key": "sk-test-key"}
        provider = SerpSearchProvider(config)
        
        result = provider._make_request("test query", max_results=5)
        
        assert result is not None
        assert "organic_results" in result
        assert len(result["organic_results"]) == 1
        
        # Verify request parameters
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[0][0] == "https://serpapi.com/search"
        
        params = call_args[1]["params"]
        assert params["api_key"] == "sk-test-key"
        assert params["q"] == "test query"
        assert params["num"] == 5
        assert params["engine"] == "google"
    
    @patch('poet.data.search_provider.requests.get')
    def test_serper_make_request_with_kwargs(self, mock_get):
        """Test SerpAPI request with additional parameters"""
        mock_response = Mock()
        mock_response.json.return_value = {"organic_results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        config = {
            "api_key": "sk-test-key",
            "engine": "bing",
            "location": "Saudi Arabia",
            "language": "ar"
        }
        provider = SerpSearchProvider(config)
        
        provider._make_request(
            "test query",
            max_results=10,
            engine="yahoo",
            location="Egypt",
            language="en",
            safe="active",
            time_period="m"
        )
        
        call_args = mock_get.call_args
        params = call_args[1]["params"]
        
        assert params["engine"] == "yahoo"  # Override from kwargs
        assert params["location"] == "Egypt"  # Override from kwargs
        assert params["hl"] == "en"  # Override from kwargs
        assert params["safe"] == "active"
        assert params["time_period"] == "m"
    
    @patch('poet.data.search_provider.requests.get')
    def test_serper_make_request_http_error(self, mock_get):
        """Test SerpAPI request with HTTP error"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        mock_get.return_value = mock_response
        
        config = {"api_key": "sk-test-key"}
        provider = SerpSearchProvider(config)
        
        # Should handle HTTP error gracefully and return None
        result = provider._make_request("test query")
        assert result is None
    
    @patch('poet.data.search_provider.requests.get')
    def test_serper_make_request_json_error(self, mock_get):
        """Test SerpAPI request with JSON decode error"""
        mock_response = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        config = {"api_key": "sk-test-key"}
        provider = SerpSearchProvider(config)
        
        result = provider._make_request("test query")
        assert result is None
    
    @patch('poet.data.search_provider.requests.get')
    def test_serper_is_available_success(self, mock_get):
        """Test successful availability check"""
        mock_response = Mock()
        mock_response.json.return_value = {"organic_results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        config = {"api_key": "sk-test-key"}
        provider = SerpSearchProvider(config)
        
        assert provider.is_available() is True
    
    @patch('poet.data.search_provider.requests.get')
    def test_serper_is_available_failure(self, mock_get):
        """Test availability check failure"""
        mock_get.side_effect = Exception("Connection error")
        
        config = {"api_key": "sk-test-key"}
        provider = SerpSearchProvider(config)
        
        assert provider.is_available() is False
    
    def test_serper_is_available_invalid_config(self):
        """Test availability check with invalid config"""
        # This should raise ValueError during initialization, not during is_available
        with pytest.raises(ValueError, match="SerpAPI API key is required"):
            SerpSearchProvider({})


class TestSearchProviderFactory:
    """Test SearchProviderFactory class"""
    
    def test_create_serper_provider(self):
        """Test creating Serp provider"""
        provider = SearchProviderFactory.create_provider(
            "serp",
            {"api_key": "sk-test-key"}
        )
        
        assert isinstance(provider, SerpSearchProvider)
        assert provider.api_key == "sk-test-key"
    
    def test_create_unsupported_provider(self):
        """Test creating unsupported provider type"""
        with pytest.raises(ValueError, match="Unsupported search provider type"):
            SearchProviderFactory.create_provider("unsupported", {})
    
    def test_create_serper_provider_convenience(self):
        """Test convenience method for creating Serp provider"""
        provider = SearchProviderFactory.create_serp_provider(
            "sk-test-key",
            engine="bing",
            location="Saudi Arabia"
        )
        
        assert isinstance(provider, SerpSearchProvider)
        assert provider.api_key == "sk-test-key"
        assert provider.engine == "bing"
        assert provider.location == "Saudi Arabia"


# Real integration tests (only run if TEST_REAL_SEARCH=1)
@pytest.mark.skipif(
    not os.getenv("TEST_REAL_SEARCH"),
    reason="Real search tests require TEST_REAL_SEARCH=1"
)
class TestSearchProviderIntegration:
    """Integration tests for search providers with real API calls"""
    
    @pytest.fixture
    def serper_config(self):
        """Get Serp configuration from environment"""
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            pytest.skip("SERPAPI_API_KEY environment variable not set")
        return {"api_key": api_key}
    
    def test_serper_real_search(self, serper_config):
        """Test real Serp search"""
        provider = SerpSearchProvider(serper_config)
        
        # Test availability
        assert provider.is_available() is True
        
        # Test basic search
        response = provider.search("Python programming", max_results=3)
        
        assert isinstance(response, SearchResponse)
        assert len(response.results) > 0
        assert response.search_time > 0
        assert response.query == "Python programming"
        assert response.provider == "SerpSearchProvider"
        
        # Check result structure
        for result in response.results:
            assert isinstance(result, SearchResult)
            assert result.title
            assert result.url
            assert result.snippet
    
    def test_serper_arabic_search(self, serper_config):
        """Test Serp search with Arabic query"""
        provider = SerpSearchProvider(serper_config)
        
        response = provider.search(
            "الشعر العربي",
            max_results=3,
            language="ar",
            location="Saudi Arabia"
        )
        
        assert isinstance(response, SearchResponse)
        assert len(response.results) > 0
        assert response.search_time > 0
    
    def test_serper_search_with_parameters(self, serper_config):
        """Test Serp search with various parameters"""
        provider = SerpSearchProvider(serper_config)
        
        response = provider.search(
            "machine learning",
            max_results=5,
            engine="google",
            location="United States",
            language="en",
            safe="active",
            time_period="m"
        )
        
        assert isinstance(response, SearchResponse)
        assert len(response.results) > 0
        assert response.search_time > 0
    
    def test_serper_error_handling(self, serper_config):
        """Test Serp error handling with invalid query"""
        provider = SerpSearchProvider(serper_config)
        
        # Test with very long query that might cause issues
        long_query = "x" * 1000
        response = provider.search(long_query, max_results=1)
        
        # Should handle gracefully (might return error or empty results)
        assert isinstance(response, SearchResponse)
        assert response.search_time > 0 
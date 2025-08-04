# poet/data/search_provider.py

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
import json
import requests
import os
from pathlib import Path

@dataclass
class SearchResult:
    """Represents a single search result"""
    title: str
    url: str
    snippet: str
    source: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class SearchResponse:
    """Represents a complete search response"""
    results: List[SearchResult]
    total_results: int
    search_time: float
    query: str
    provider: str
    metadata: Dict[str, Any]

class BaseSearchProvider(ABC):
    """
    Abstract base class for search providers.
    
    Provides only basic search functionality - specialized search logic
    should be implemented in the knowledge retriever layer.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the search provider.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def _make_request(self, query: str, max_results: int = 10, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make provider-specific HTTP request.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            Raw response data dictionary or None if failed
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the search provider is available.
        
        Returns:
            True if the provider can be used
        """
        pass
    
    def validate_config(self) -> bool:
        """
        Validate the provider configuration.
        
        Returns:
            True if configuration is valid
        """
        return True
    
    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        """
        Perform a search query.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters
            
        Returns:
            SearchResponse with results and metadata
        """
        start_time = time.time()
        
        try:
            # Make provider-specific request
            response_data = self._make_request(query, max_results, **kwargs)
            
            if not response_data:
                return SearchResponse(
                    results=[],
                    total_results=0,
                    search_time=time.time() - start_time,
                    query=query,
                    provider=self.__class__.__name__,
                    metadata={"error": "No response from provider"}
                )
            
            # Parse results using provider-specific parser
            results = self._parse_results(response_data)
            
            # Extract metadata
            metadata = {
                "provider": self.__class__.__name__,
                "raw_response": response_data
            }
            
            return SearchResponse(
                results=results[:max_results],
                total_results=len(results),
                search_time=time.time() - start_time,
                query=query,
                provider=self.__class__.__name__,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return SearchResponse(
                results=[],
                total_results=0,
                search_time=time.time() - start_time,
                query=query,
                provider=self.__class__.__name__,
                metadata={"error": str(e)}
            )
    
    def _parse_results(self, response_data: Dict[str, Any]) -> List[SearchResult]:
        """
        Parse provider response into SearchResult objects.
        
        Args:
            response_data: Raw response from provider
            
        Returns:
            List of SearchResult objects
        """
        # Default implementation - should be overridden by providers
        results = []
        
        # Try to extract organic results (common format)
        organic_results = response_data.get('organic_results', [])
        
        for result in organic_results:
            try:
                # Check for required fields
                title = result.get('title', '')
                link = result.get('link', '')
                snippet = result.get('snippet', '')
                source = result.get('source', '')
                
                # Skip results with missing required fields
                if not title or not link:
                    continue
                
                search_result = SearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    source=source,
                    metadata={
                        'position': result.get('position'),
                        'displayed_link': result.get('displayed_link'),
                        'date': result.get('date'),
                        'rich_snippet': result.get('rich_snippet', {}),
                        'sitelinks': result.get('sitelinks', [])
                    }
                )
                results.append(search_result)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse search result: {e}")
                continue
        
        return results

class MockSearchProvider(BaseSearchProvider):
    """Mock search provider for testing"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.responses = []
        self.call_count = 0
        self.logger = logging.getLogger(__name__)
    
    def add_response(self, results: List[SearchResult]):
        """Add a response to be returned on next search call"""
        self.responses.append(results)
    
    def reset(self):
        """Reset the mock provider state"""
        self.responses = []
        self.call_count = 0
    
    def _make_request(self, query: str, max_results: int = 10, **kwargs) -> Optional[Dict[str, Any]]:
        """Mock request - returns None to trigger default response"""
        return None
    
    def is_available(self) -> bool:
        """Mock provider is always available"""
        return True
    
    def search(self, query: str, max_results: int = 10, **kwargs) -> SearchResponse:
        """Return mock search results"""
        self.call_count += 1
        self.logger.info(f"Mock search called with query: {query}, max_results: {max_results}")
        
        start_time = time.time()
        
        if self.responses:
            mock_results = self.responses.pop(0)
        else:
            # Default mock response if no responses are set
            mock_results = [
                SearchResult(
                    title=f"Mock Result for: {query}",
                    url="https://example.com/mock",
                    snippet=f"This is a mock search result for the query: {query}",
                    source="mock"
                )
            ]
        
        return SearchResponse(
            results=mock_results[:max_results],
            total_results=len(mock_results),
            search_time=time.time() - start_time,
            query=query,
            provider="MockSearchProvider",
            metadata={"mock": True, "call_count": self.call_count}
        )

class SerperSearchProvider(BaseSearchProvider):
    """
    Serper (SerpAPI) search provider implementation.
    
    Uses SerpAPI to perform web searches with various search engines.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Serper search provider.
        
        Args:
            config: Configuration dictionary with:
                - api_key: SerpAPI API key
                - engine: Search engine (google, bing, etc.)
                - location: Geographic location for results
                - language: Language for results
        """
        super().__init__(config)
        self.api_key = config.get('api_key')
        self.engine = config.get('engine', 'google')
        self.location = config.get('location', 'United States')
        self.language = config.get('language', 'en')
        self.base_url = "https://serpapi.com/search"
        
        if not self.api_key:
            raise ValueError("SerpAPI API key is required")
    
    def validate_config(self) -> bool:
        """Validate Serper configuration"""
        if not self.api_key:
            self.logger.error("SerpAPI API key is missing")
            return False
        
        if not self.api_key.startswith('sk-'):
            self.logger.warning("SerpAPI API key format may be incorrect")
        
        return True
    
    def is_available(self) -> bool:
        """Check if Serper is available"""
        if not self.validate_config():
            return False
        
        # Test with a simple query
        try:
            test_response = self._make_request("test", max_results=1)
            return test_response is not None
        except Exception as e:
            self.logger.error(f"Serper availability check failed: {e}")
            return False
    
    def _make_request(self, query: str, max_results: int = 10, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make HTTP request to SerpAPI.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            **kwargs: Additional parameters
            
        Returns:
            Response data dictionary or None if failed
        """
        
        params = {
            'api_key': self.api_key,
            'q': query,
            'num': min(max_results, 100),  # SerpAPI limit
            'engine': kwargs.get('engine', self.engine),
            'location': kwargs.get('location', self.location),
            'hl': kwargs.get('language', self.language),
        }
        
        # Add optional parameters
        if 'safe' in kwargs:
            params['safe'] = kwargs['safe']
        if 'time_period' in kwargs:
            params['time_period'] = kwargs['time_period']
        if 'gl' in kwargs:
            params['gl'] = kwargs['gl']  # Country code
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except (requests.exceptions.RequestException, Exception) as e:
            self.logger.error(f"SerpAPI request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse SerpAPI response: {e}")
            return None


def _load_search_provider_config() -> Optional[Dict[str, Any]]:
    """Load LLM configuration from fixtures."""
    config_files = ["search_providers.json"]
    
    # Look in tests/fixtures directory
    base_path = Path(__file__).parent.parent.parent / "tests" / "fixtures"
    
    for config_file in config_files:
        config_path = base_path / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    
    return None


class SearchProviderFactory:
    """
    Factory class for creating search provider instances.
    """
    
    @staticmethod
    def create_provider(provider_type: str, config: Dict[str, Any]) -> BaseSearchProvider:
        """
        Create a search provider instance.
        
        Args:
            provider_type: Type of provider ('serper', 'mock', etc.)
            config: Provider-specific configuration
            
        Returns:
            Configured search provider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        if provider_type.lower() == 'serper':
            return SerperSearchProvider(config)
        elif provider_type.lower() == 'mock':
            return MockSearchProvider(config)
        else:
            raise ValueError(f"Unsupported search provider type: {provider_type}")
    
    @staticmethod
    def create_serper_provider(api_key: str, **kwargs) -> SerperSearchProvider:
        """
        Convenience method to create Serper provider.
        
        Args:
            api_key: SerpAPI API key
            **kwargs: Additional configuration options
            
        Returns:
            Configured SerperSearchProvider instance
        """
        config = {
            'api_key': api_key,
            **kwargs
        }
        return SerperSearchProvider(config)
    
    @staticmethod
    def create_provider_from_env():
        """
        Get real Search provider instance from environment variable and configuration.
        
        Returns:
            BaseSearchProvider instance if configured, None otherwise
            
        Environment Variables:
            TEST_REAL_SEARCH: Must be set to enable real LLM loading
            REAL_SEARCH_PROVIDER: Specify which provider to use (default: openai)
        """
        if not os.getenv("TEST_REAL_SEARCH"):
            return None
        
        provider = os.getenv("REAL_SEARCH_PROVIDER", "serper").lower()
        
        config_data = _load_search_provider_config()
        if not config_data or provider not in config_data:
            print(f"DEBUG: No config data or provider not found, returning None")
            return None
        
        provider_config = config_data[provider]
        
        provider = SearchProviderFactory.create_provider(provider,provider_config)
        return provider

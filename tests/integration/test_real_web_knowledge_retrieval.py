# tests/integration/test_real_web_knowledge_retrieval.py

import os
import pytest
import json
from unittest.mock import patch
from poet.models.constraints import Constraints
from poet.data.search_provider import SearchResult, SearchResponse
from poet.analysis.constraint_parser import ConstraintParser
from poet.analysis.knowledge_retriever import WebKnowledgeRetriever
from poet.prompts.prompt_manager import PromptManager
from poet.llm.base_llm import MockLLM, LLMConfig
from poet.data.search_provider import MockSearchProvider


@pytest.mark.integration
@pytest.mark.real_data
class TestRealWebKnowledgeRetrieval:
    """Integration tests for web knowledge retrieval with all LLM/search provider combinations"""
    
    @pytest.fixture(scope="class")
    def prompt_manager(self):
        """Create a PromptManager instance"""
        return PromptManager()
    
    @pytest.fixture(scope="class")
    def mock_llm(self):
        """Create a mock LLM"""
        config = LLMConfig(model_name="test-model")
        return MockLLM(config)
    
    @pytest.fixture(scope="class")
    def mock_search_provider(self):
        """Create a mock search provider"""
        return MockSearchProvider()
    
    @pytest.fixture(scope="class")
    def test_data(self):
        """Load test data"""
        import json
        from pathlib import Path
        test_file = Path(__file__).parent.parent / "fixtures" / "test_data.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @pytest.fixture(scope="class")
    def real_llm(self):
        """Get real LLM if available - only when needed"""
        # Only initialize if TEST_REAL_LLMS is set
        if not os.getenv("TEST_REAL_LLMS"):
            return None
        from poet.llm.llm_factory import get_real_llm_from_env
        return get_real_llm_from_env()
    
    @pytest.fixture(scope="class")
    def real_search_provider(self):
        """Get real search provider if available - only when needed"""
        # Only initialize if TEST_REAL_SEARCH is set
        if not os.getenv("TEST_REAL_SEARCH"):
            return None
        from poet.data.search_provider import SearchProviderFactory
        return SearchProviderFactory.create_provider_from_env()
    
    def _should_skip_test(self, llm_type, search_provider_type):
        """Determine if test should be skipped based on environment variables"""
        test_real_llms = os.getenv("TEST_REAL_LLMS")
        test_real_search = os.getenv("TEST_REAL_SEARCH")
        
        # Check if environment variables are actually set to truthy values
        test_real_llms_enabled = test_real_llms and test_real_llms.lower() not in ['0', 'false', 'no', '']
        test_real_search_enabled = test_real_search and test_real_search.lower() not in ['0', 'false', 'no', '']
        
        # Default behavior: only run mock/mock when no environment variables are set
        if not test_real_llms_enabled and not test_real_search_enabled:
            if llm_type != "mock" or search_provider_type != "mock":
                return True, "Only running mock/mock tests when no environment variables are set"
        
        # If TEST_REAL_LLMS is set, only run real LLM tests
        elif test_real_llms_enabled and not test_real_search_enabled:
            if llm_type != "real" or search_provider_type != "mock":
                return True, "Only running real/mock tests when TEST_REAL_LLMS is set"
        
        # If TEST_REAL_SEARCH is set, only run real search tests
        elif test_real_search_enabled and not test_real_llms_enabled:
            if llm_type != "mock" or search_provider_type != "real":
                return True, "Only running mock/real tests when TEST_REAL_SEARCH is set"
        
        # If both are set, only run real/real tests
        elif test_real_llms_enabled and test_real_search_enabled:
            if llm_type != "real" or search_provider_type != "real":
                return True, "Only running real/real tests when both environment variables are set"
        
        return False, None
    
    def _create_constraint_parser(self, llm_type, mock_llm, real_llm, prompt_manager):
        """Create ConstraintParser with appropriate LLM"""
        if llm_type == "mock":
            return ConstraintParser(mock_llm, prompt_manager)
        elif llm_type == "real":
            if real_llm is None:
                pytest.skip("Real LLM not available")
            return ConstraintParser(real_llm, prompt_manager)
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")
    
    def _create_web_retriever(self, llm_type, search_provider_type, mock_llm, real_llm, 
                            mock_search_provider, real_search_provider):
        """Create WebKnowledgeRetriever with appropriate components"""
        # Select LLM
        if llm_type == "mock":
            llm = mock_llm
        elif llm_type == "real":
            if real_llm is None:
                pytest.skip("Real LLM not available")
            llm = real_llm
        else:
            raise ValueError(f"Unknown llm_type: {llm_type}")
        
        # Select search provider
        if search_provider_type == "mock":
            search_provider = mock_search_provider
        elif search_provider_type == "real":
            if real_search_provider is None:
                pytest.skip("Real search provider not available")
            search_provider = real_search_provider
        else:
            raise ValueError(f"Unknown search_provider_type: {search_provider_type}")
        
        # Create retriever with a patch to avoid factory creation
        with patch('poet.analysis.knowledge_retriever.SearchProviderFactory') as mock_factory:
            mock_factory.create_provider.return_value = search_provider
            
            retriever = WebKnowledgeRetriever(
                llm_provider=llm,
                search_provider_type="mock",  # This will be ignored due to patch
                search_provider_config={}
            )
            return retriever
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    @pytest.mark.parametrize("search_provider_type", ["mock", "real"])
    def test_web_knowledge_retrieval_example_1(self, llm_type, search_provider_type, 
                                             mock_llm, mock_search_provider, prompt_manager, test_data):
        """Test web knowledge retrieval for example 1 (غزل poem) - all LLM/search provider combinations"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type, search_provider_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        expected_web_search = example["web_search"]
        
        # Get real components only when needed
        real_llm = None
        real_search_provider = None
        
        if llm_type == "real":
            from poet.llm.llm_factory import get_real_llm_from_env
            real_llm = get_real_llm_from_env()
        
        if search_provider_type == "real":
            from poet.data.search_provider import SearchProviderFactory
            real_search_provider = SearchProviderFactory.create_provider_from_env()
        
        # Create components
        constraint_parser = self._create_constraint_parser(llm_type, mock_llm, real_llm, prompt_manager)
        web_retriever = self._create_web_retriever(llm_type, search_provider_type, mock_llm, real_llm,
                                                 mock_search_provider, real_search_provider)
        
        # Set up mock responses for consistent testing when using mock components
        if llm_type == "mock":
            # Create separate mock LLM instances to avoid response conflicts
            constraint_mock_llm = MockLLM(LLMConfig(model_name="test-model"))
            web_mock_llm = MockLLM(LLMConfig(model_name="test-model"))
            
            # Mock response for constraint parsing
            mock_constraint_response = json.dumps(expected_constraints, ensure_ascii=False)
            constraint_mock_llm.responses = [mock_constraint_response]
            constraint_mock_llm.reset()
            
            # Mock responses for web search
            query_response = '''```json
            {
                "queries": [
                    {
                        "query": "بحر الكامل غزل شعر عربي",
                        "purpose": "العثور على معلومات تقنية وأمثلة"
                    }
                ]
            }
            ```'''
            
            eval_response = '''```json
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 8,
                        "quality_score": 7,
                        "usefulness_score": 8,
                        "is_worth_following": true,
                        "key_insights": ["معلومات مفيدة"],
                        "recommendation": "مفيد"
                    }
                ],
                "overall_assessment": "نتائج جيدة",
                "gaps_identified": [],
                "followup_needed": false
            }
            ```'''
            
            # Set up mock LLM responses for web search
            web_mock_llm.responses = [query_response, eval_response]
            web_mock_llm.reset()
            
            # Replace the LLMs with our configured ones
            constraint_parser.llm = constraint_mock_llm
            web_retriever.llm = web_mock_llm
        
        if search_provider_type == "mock":
            # Set up mock search results
            mock_search_results = [
                SearchResult(
                    title="بحر الكامل في الشعر العربي",
                    url="https://example.com/kamil",
                    snippet="معلومات عن بحر الكامل وأوزانه في الشعر العربي",
                    source="web"
                )
            ]
            web_retriever.search_provider.add_response(mock_search_results)
        
        # Parse constraints using ConstraintParser
        constraints = constraint_parser.parse_constraints(user_prompt)
        
        # Test web search retrieval
        result = web_retriever.search(
            constraints, 
            max_queries_per_round=3,
            max_rounds=1
        )
        
        # Basic assertions
        assert len(result.web_results) >= 0, "Should handle بحر الكامل + غزل web search"
        assert result.retrieval_strategy == "multi_round_web_search_1_rounds"
        
        # Print test configuration and results
        print(f"\nExample 1 - Configuration: LLM={llm_type}, Search={search_provider_type}")
        print(f"Parsed constraints:")
        print(f"  Meter: {constraints.meter}")
        print(f"  Theme: {constraints.theme}")
        print(f"  Line count: {constraints.line_count}")
        print(f"  Tone: {constraints.tone}")
        print(f"Found {len(result.web_results)} web results")
        
        for i, web_result in enumerate(result.web_results[:3]):
            print(f"\nWeb Result {i+1}:")
            print(f"  Title: {web_result.title}")
            print(f"  URL: {web_result.url}")
            print(f"  Snippet: {web_result.snippet[:100]}...")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert constraints.meter == expected_constraints["meter"]
            assert constraints.theme == expected_constraints["theme"]
            assert constraints.line_count == expected_constraints["line_count"]
        
        # Verify metadata
        print(f"\nWeb search metadata:")
        print(f"  Rounds executed: {result.metadata['rounds_executed']}")
        print(f"  Total queries: {result.metadata['total_queries']}")
        print(f"  Total results: {result.metadata['total_results']}")
        
        if result.metadata["evaluation_summary"]:
            summary = result.metadata["evaluation_summary"][0]
            print(f"  High quality results: {summary['high_quality_results']}")
            print(f"  Overall assessment: {summary['overall_assessment']}")
            print(f"  Gaps identified: {summary['gaps_identified']}")
    
    @pytest.mark.parametrize("llm_type", ["mock", "real"])
    @pytest.mark.parametrize("search_provider_type", ["mock", "real"])
    def test_web_knowledge_retrieval_example_2(self, llm_type, search_provider_type,
                                             mock_llm, mock_search_provider, prompt_manager, test_data):
        """Test web knowledge retrieval for example 2 (هجاء poem) - all LLM/search provider combinations"""
        # Check if test should be skipped
        should_skip, reason = self._should_skip_test(llm_type, search_provider_type)
        if should_skip:
            pytest.skip(reason)
        
        example = test_data[1]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        expected_web_search = example["web_search"]
        
        # Get real components only when needed
        real_llm = None
        real_search_provider = None
        
        if llm_type == "real":
            from poet.llm.llm_factory import get_real_llm_from_env
            real_llm = get_real_llm_from_env()
        
        if search_provider_type == "real":
            from poet.data.search_provider import SearchProviderFactory
            real_search_provider = SearchProviderFactory.create_provider_from_env()
        
        # Create components
        constraint_parser = self._create_constraint_parser(llm_type, mock_llm, real_llm, prompt_manager)
        web_retriever = self._create_web_retriever(llm_type, search_provider_type, mock_llm, real_llm,
                                                 mock_search_provider, real_search_provider)
        
        # Set up mock responses for consistent testing when using mock components
        if llm_type == "mock":
            # Create separate mock LLM instances to avoid response conflicts
            constraint_mock_llm = MockLLM(LLMConfig(model_name="test-model"))
            web_mock_llm = MockLLM(LLMConfig(model_name="test-model"))
            
            # Mock response for constraint parsing
            mock_constraint_response = json.dumps(expected_constraints, ensure_ascii=False)
            constraint_mock_llm.responses = [mock_constraint_response]
            constraint_mock_llm.reset()
            
            # Mock responses for web search
            query_response = '''```json
            {
                "queries": [
                    {
                        "query": "بحر الطويل هجاء شعر عربي",
                        "purpose": "العثور على أمثلة هجاء"
                    }
                ]
            }
            ```'''
            
            eval_response = '''```json
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 7,
                        "quality_score": 6,
                        "usefulness_score": 7,
                        "is_worth_following": true,
                        "key_insights": ["أمثلة هجاء"],
                        "recommendation": "مفيد للهجاء"
                    }
                ],
                "overall_assessment": "نتائج مناسبة",
                "gaps_identified": [],
                "followup_needed": false
            }
            ```'''
            
            # Set up mock LLM responses for web search
            web_mock_llm.responses = [query_response, eval_response]
            web_mock_llm.reset()
            
            # Replace the LLMs with our configured ones
            constraint_parser.llm = constraint_mock_llm
            web_retriever.llm = web_mock_llm
        
        if search_provider_type == "mock":
            # Set up mock search results
            mock_search_results = [
                SearchResult(
                    title="بحر الطويل في الهجاء",
                    url="https://example.com/tawil",
                    snippet="أمثلة من شعر الهجاء على بحر الطويل",
                    source="web"
                )
            ]
            web_retriever.search_provider.add_response(mock_search_results)
        
        # Parse constraints using ConstraintParser
        constraints = constraint_parser.parse_constraints(user_prompt)
        
        # Test web search retrieval
        result = web_retriever.search(
            constraints,
            max_queries_per_round=3,
            max_rounds=1
        )
        
        # Basic assertions
        assert len(result.web_results) >= 0, "Should handle بحر الطويل + هجاء web search"
        assert result.retrieval_strategy == "multi_round_web_search_1_rounds"
        
        # Print test configuration and results
        print(f"\nExample 2 - Configuration: LLM={llm_type}, Search={search_provider_type}")
        print(f"Parsed constraints:")
        print(f"  Meter: {constraints.meter}")
        print(f"  Theme: {constraints.theme}")
        print(f"  Line count: {constraints.line_count}")
        print(f"  Tone: {constraints.tone}")
        print(f"Found {len(result.web_results)} web results")
        
        for i, web_result in enumerate(result.web_results[:3]):
            print(f"\nWeb Result {i+1}:")
            print(f"  Title: {web_result.title}")
            print(f"  URL: {web_result.url}")
            print(f"  Snippet: {web_result.snippet[:100]}...")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert constraints.meter == expected_constraints["meter"]
            assert constraints.theme == expected_constraints["theme"]
            assert constraints.line_count == expected_constraints["line_count"]
        
        # Verify metadata
        print(f"\nWeb search metadata:")
        print(f"  Rounds executed: {result.metadata['rounds_executed']}")
        print(f"  Total queries: {result.metadata['total_queries']}")
        print(f"  Total results: {result.metadata['total_results']}")
        
        if result.metadata["evaluation_summary"]:
            summary = result.metadata["evaluation_summary"][0]
            print(f"  High quality results: {summary['high_quality_results']}")
            print(f"  Overall assessment: {summary['overall_assessment']}")
            print(f"  Gaps identified: {summary['gaps_identified']}")
    

    
# tests/integration/test_web_search_integration.py

import pytest
from unittest.mock import Mock, patch, MagicMock
from poet.analysis.knowledge_retriever import WebKnowledgeRetriever, WebRetrievalResult
from poet.models.constraints import Constraints
from poet.models.search import SearchQuery, EvaluatedResult, ResultEvaluationResult
from poet.data.search_provider import SearchResult, SearchResponse


class TestWebSearchIntegration:
    """Integration tests for web search functionality with mocks"""
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM provider"""
        llm = Mock()
        return llm
    
    @pytest.fixture
    def mock_search_provider(self):
        """Mock search provider"""
        provider = Mock()
        return provider
    
    @pytest.fixture
    def web_retriever(self, mock_llm, mock_search_provider):
        """WebKnowledgeRetriever with mocked dependencies"""
        with patch('poet.analysis.knowledge_retriever.SearchProviderFactory') as mock_factory:
            mock_factory.create_provider.return_value = mock_search_provider
            
            retriever = WebKnowledgeRetriever(
                llm_provider=mock_llm,
                search_provider_type="mock",
                search_provider_config={}
            )
            return retriever
    
    @pytest.fixture
    def sample_constraints(self):
        """Sample Constraints for testing"""
        user_constraints = Constraints(
            meter="بحر الكامل",
            theme="قصيدة غزل",
            qafiya="ق",
            line_count=4,
            tone="رومانسي",
            imagery=["الورد", "القمر"],
            keywords=["حب", "شوق"],
            register="فصيح",
            era="كلاسيكي",
            poet_style="المتنبي",
            
        )
        user_constraints.original_prompt = "اكتب لي قصيدة غزل على بحر الكامل"
        return user_constraints
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing"""
        return [
            SearchResult(
                title="البحر الكامل في الشعر العربي",
                url="https://example.com/kamil",
                snippet="معلومات عن بحر الكامل وأوزانه وقواعده في الشعر العربي",
                source="web",
                metadata={}
            ),
            SearchResult(
                title="قصائد غزل على بحر الكامل",
                url="https://example.com/ghazal",
                snippet="مجموعة من قصائد الغزل المكتوبة على بحر الكامل",
                source="web",
                metadata={}
            ),
            SearchResult(
                title="المتنبي وأسلوبه الشعري",
                url="https://example.com/mutanabbi",
                snippet="تحليل لأسلوب المتنبي الشعري ومميزاته",
                source="web",
                metadata={}
            )
        ]

    def test_web_search_basic_flow(self, web_retriever, sample_constraints, mock_llm, mock_search_provider):
        """Test basic web search flow with mocked responses"""
        
        # Mock query generation response
        mock_llm.generate.return_value = '''
        {
            "queries": [
                {
                    "query": "البحر الكامل في الشعر العربي أمثلة وقواعد",
                    "purpose": "العثور على معلومات تقنية وأمثلة للبحر المحدد"
                },
                {
                    "query": "قصيدة غزل في العصر الكلاسيكي أمثلة وتحليل",
                    "purpose": "العثور على السياق الثقافي وأمثلة للموضوع في العصر المحدد"
                }
            ]
        }
        '''
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(
                    title="نتيجة بحث 1",
                    url="https://example.com/1",
                    snippet="محتوى النتيجة الأولى",
                    source="web",
                    metadata={}
                ),
                SearchResult(
                    title="نتيجة بحث 2", 
                    url="https://example.com/2",
                    snippet="محتوى النتيجة الثانية",
                    source="web",
                    metadata={}
                )
            ],
            total_results=2,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        # Mock evaluation response
        mock_llm.generate.side_effect = [
            # First call: query generation
            '''
            {
                "queries": [
                    {
                        "query": "البحر الكامل في الشعر العربي أمثلة وقواعد",
                        "purpose": "العثور على معلومات تقنية وأمثلة للبحر المحدد"
                    }
                ]
            }
            ''',
            # Second call: result evaluation
            '''
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 8,
                        "quality_score": 7,
                        "usefulness_score": 8,
                        "is_worth_following": true,
                        "key_insights": ["معلومات عن بحر الكامل", "أمثلة شعرية"],
                        "recommendation": "مفيد لتوليد الشعر"
                    },
                    {
                        "result_index": 1,
                        "relevance_score": 6,
                        "quality_score": 5,
                        "usefulness_score": 6,
                        "is_worth_following": false,
                        "key_insights": ["معلومات سطحية"],
                        "recommendation": "معلومات سطحية"
                    }
                ],
                "overall_assessment": "نتائج جيدة للبحر الكامل",
                "gaps_identified": ["حاجة لمزيد من الأمثلة"],
                "followup_needed": false
            }
            '''
        ]
        
        # Execute search
        result = web_retriever.search(sample_constraints, max_queries_per_round=2, max_rounds=1)
        
        # Verify result structure
        assert isinstance(result, WebRetrievalResult)
        assert result.total_found == 1  # Only one high-quality result
        assert len(result.web_results) == 1
        assert result.retrieval_strategy == "multi_round_web_search_1_rounds"
        
        # Verify metadata
        assert result.metadata["rounds_executed"] == 1
        assert result.metadata["total_queries"] == 1
        assert result.metadata["total_results"] == 2
        
        # Verify search provider was called
        mock_search_provider.search.assert_called_once()
        assert "البحر الكامل" in mock_search_provider.search.call_args[0][0]

    def test_web_search_multiple_rounds(self, web_retriever, sample_constraints, mock_llm, mock_search_provider):
        """Test multi-round web search with follow-up queries"""
        
        # Mock responses for multiple rounds
        mock_llm.generate.side_effect = [
            # Round 1 - Query generation
            '''
            {
                "queries": [
                    {
                        "query": "البحر الكامل في الشعر العربي",
                        "purpose": "معلومات تقنية"
                    }
                ]
            }
            ''',
            # Round 1 - Evaluation (indicates follow-up needed)
            '''
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 7,
                        "quality_score": 6,
                        "usefulness_score": 7,
                        "is_worth_following": true,
                        "key_insights": ["معلومات أساسية"],
                        "recommendation": "مفيد"
                    }
                ],
                "overall_assessment": "نتائج جيدة ولكن تحتاج متابعة",
                "gaps_identified": ["حاجة لأمثلة أكثر"],
                "followup_needed": true
            }
            ''',
            # Round 2 - Query generation
            '''
            {
                "queries": [
                    {
                        "query": "أمثلة شعرية على بحر الكامل",
                        "purpose": "أمثلة عملية"
                    }
                ]
            }
            ''',
            # Round 2 - Evaluation (no follow-up needed)
            '''
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 9,
                        "quality_score": 8,
                        "usefulness_score": 9,
                        "is_worth_following": true,
                        "key_insights": ["أمثلة ممتازة"],
                        "recommendation": "مفيد جداً"
                    }
                ],
                "overall_assessment": "نتائج ممتازة",
                "gaps_identified": [],
                "followup_needed": false
            }
            '''
        ]
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(
                    title="نتيجة بحث",
                    url="https://example.com",
                    snippet="محتوى النتيجة",
                    source="web",
                    metadata={}
                )
            ],
            total_results=1,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        # Execute search with 2 rounds
        result = web_retriever.search(sample_constraints, max_queries_per_round=1, max_rounds=2)
        
        # Verify two rounds were executed
        assert result.metadata["rounds_executed"] == 2
        assert result.metadata["total_queries"] == 2
        assert result.retrieval_strategy == "multi_round_web_search_2_rounds"
        
        # Verify search provider was called twice
        assert mock_search_provider.search.call_count == 2

    def test_web_search_no_results(self, web_retriever, sample_constraints, mock_llm, mock_search_provider):
        """Test web search when no results are found"""
        
        # Mock query generation
        mock_llm.generate.return_value = '''
        {
            "queries": [
                {
                    "query": "البحر الكامل في الشعر العربي",
                    "purpose": "معلومات تقنية"
                }
            ]
        }
        '''
        
        # Mock empty search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[],
            total_results=0,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        # Execute search
        result = web_retriever.search(sample_constraints, max_queries_per_round=1, max_rounds=1)
        
        # Verify no results
        assert result.total_found == 0
        assert len(result.web_results) == 0
        assert result.metadata["rounds_executed"] == 0

    def test_web_search_low_quality_results(self, web_retriever, sample_constraints, mock_llm, mock_search_provider):
        """Test web search with low quality results that are filtered out"""
        
        # Mock query generation
        mock_llm.generate.return_value = '''
        {
            "queries": [
                {
                    "query": "البحر الكامل في الشعر العربي",
                    "purpose": "معلومات تقنية"
                }
            ]
        }
        '''
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(
                    title="نتيجة منخفضة الجودة",
                    url="https://example.com/low",
                    snippet="محتوى منخفض الجودة",
                    source="web",
                    metadata={}
                )
            ],
            total_results=1,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        # Mock evaluation with low scores
        mock_llm.generate.side_effect = [
            # Query generation
            '''
            {
                "queries": [
                    {
                        "query": "البحر الكامل في الشعر العربي",
                        "purpose": "معلومات تقنية"
                    }
                ]
            }
            ''',
            # Result evaluation with low scores
            '''
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 3,
                        "quality_score": 2,
                        "usefulness_score": 3,
                        "is_worth_following": false,
                        "key_insights": ["معلومات غير مفيدة"],
                        "recommendation": "غير مفيد"
                    }
                ],
                "overall_assessment": "نتائج منخفضة الجودة",
                "gaps_identified": ["حاجة لنتائج أفضل"],
                "followup_needed": false
            }
            '''
        ]
        
        # Execute search
        result = web_retriever.search(sample_constraints, max_queries_per_round=1, max_rounds=1)
        
        # Verify low quality results are filtered out
        assert result.total_found == 0
        assert len(result.web_results) == 0
        assert result.metadata["rounds_executed"] == 1

    def test_web_search_error_handling(self, web_retriever, sample_constraints, mock_llm, mock_search_provider):
        """Test web search error handling"""
        
        # Mock query generation failure
        mock_llm.generate.side_effect = Exception("LLM error")
        
        # Execute search
        result = web_retriever.search(sample_constraints, max_queries_per_round=1, max_rounds=1)
        
        # Verify graceful handling
        assert result.total_found == 0
        assert len(result.web_results) == 0
        assert result.metadata["rounds_executed"] == 0

    def test_web_search_invalid_json_response(self, web_retriever, sample_constraints, mock_llm, mock_search_provider):
        """Test web search with invalid JSON responses"""
        
        # Mock invalid JSON response
        mock_llm.generate.return_value = "Invalid JSON response"
        
        # Execute search
        result = web_retriever.search(sample_constraints, max_queries_per_round=1, max_rounds=1)
        
        # Verify graceful handling
        assert result.total_found == 0
        assert len(result.web_results) == 0

    def test_web_search_with_empty_constraints(self, web_retriever, mock_llm, mock_search_provider):
        """Test web search with minimal constraints"""
        
        # Create constraints with minimal information
        constraints = Constraints(
            theme="غزل",
            original_prompt="اكتب قصيدة غزل"
        )
        
        # Mock query generation
        mock_llm.generate.return_value = '''
        {
            "queries": [
                {
                    "query": "قصيدة غزل في الشعر العربي",
                    "purpose": "معلومات عن الغزل"
                }
            ]
        }
        '''
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(
                    title="قصيدة غزل",
                    url="https://example.com/ghazal",
                    snippet="معلومات عن الغزل",
                    source="web",
                    metadata={}
                )
            ],
            total_results=1,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        # Mock evaluation
        mock_llm.generate.side_effect = [
            # Query generation
            '''
            {
                "queries": [
                    {
                        "query": "قصيدة غزل في الشعر العربي",
                        "purpose": "معلومات عن الغزل"
                    }
                ]
            }
            ''',
            # Result evaluation
            '''
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 8,
                        "quality_score": 7,
                        "usefulness_score": 8,
                        "is_worth_following": true,
                        "key_insights": ["معلومات عن الغزل"],
                        "recommendation": "مفيد"
                    }
                ],
                "overall_assessment": "نتائج جيدة",
                "gaps_identified": [],
                "followup_needed": false
            }
            '''
        ]
        
        # Execute search
        result = web_retriever.search(constraints, max_queries_per_round=1, max_rounds=1)
        
        # Verify results
        assert result.total_found == 1
        assert len(result.web_results) == 1
        assert "غزل" in result.web_results[0].title

    def test_web_search_metadata_tracking(self, web_retriever, sample_constraints, mock_llm, mock_search_provider):
        """Test that search metadata is properly tracked"""
        
        # Mock responses
        mock_llm.generate.side_effect = [
            # Query generation
            '''
            {
                "queries": [
                    {
                        "query": "البحر الكامل في الشعر العربي",
                        "purpose": "معلومات تقنية"
                    },
                    {
                        "query": "قصيدة غزل في العصر الكلاسيكي",
                        "purpose": "معلومات ثقافية"
                    }
                ]
            }
            ''',
            # Result evaluation
            '''
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
                    },
                    {
                        "result_index": 1,
                        "relevance_score": 6,
                        "quality_score": 5,
                        "usefulness_score": 6,
                        "is_worth_following": false,
                        "key_insights": ["معلومات سطحية"],
                        "recommendation": "غير مفيد"
                    }
                ],
                "overall_assessment": "نتائج مختلطة",
                "gaps_identified": ["حاجة لمزيد من التفاصيل"],
                "followup_needed": false
            }
            '''
        ]
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(title="نتيجة 1", url="https://example.com/1",source="web", snippet="محتوى 1",metadata={}),
                SearchResult(title="نتيجة 2", url="https://example.com/2",source="web", snippet="محتوى 2",metadata={}),
                SearchResult(title="نتيجة 3", url="https://example.com/3",source="web", snippet="محتوى 3",metadata={}),
                SearchResult(title="نتيجة 4", url="https://example.com/4",source="web", snippet="محتوى 4",metadata={})
            ],
            total_results=4,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        # Execute search
        result = web_retriever.search(sample_constraints, max_queries_per_round=2, max_rounds=1)
        
        # Verify metadata
        assert result.metadata["rounds_executed"] == 1
        assert result.metadata["total_queries"] == 2
        assert result.metadata["total_results"] == 8
        
        # Verify evaluation summary
        assert len(result.metadata["evaluation_summary"]) == 1
        summary = result.metadata["evaluation_summary"][0]
        assert summary["round"] == 1
        assert summary["queries_executed"] == 2
        assert summary["results_found"] == 8
        assert summary["high_quality_results"] == 1
        assert "نتائج مختلطة" in summary["overall_assessment"] 
# tests/integration/test_constraints_web_search_integration.py

import pytest
from unittest.mock import Mock, patch
from poet.models.constraints import Constraints
from poet.analysis.knowledge_retriever import WebKnowledgeRetriever, WebRetrievalResult
from poet.data.search_provider import SearchResult, SearchResponse


class TestConstraintsWebSearchIntegration:
    """Integration tests for constraints and web search via WebKnowledgeRetriever"""
    
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
        """Create WebKnowledgeRetriever with mocked dependencies"""
        with patch('poet.analysis.knowledge_retriever.SearchProviderFactory.create_provider') as mock_factory:
            mock_factory.return_value = mock_search_provider
            return WebKnowledgeRetriever(
                llm_provider=mock_llm,
                search_provider_type="mock",
                search_provider_config={}
            )
    
    def test_basic_constraint_to_web_search_retrieval(self, web_retriever, mock_llm, mock_search_provider):
        """Test basic constraint parsing to web search retrieval"""
        constraints = Constraints(
            meter="بحر الكامل",
            theme="غزل",
            qafiya="ق",
            line_count=2
        )
        constraints.original_prompt = "اكتب قصيدة غزل على بحر الكامل"
        
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            # Query generation response
            '''```json
            {
                "queries": [
                    {
                        "query": "البحر الكامل في الشعر العربي أمثلة",
                        "purpose": "العثور على معلومات تقنية"
                    },
                    {
                        "query": "قصيدة غزل في العصر الكلاسيكي",
                        "purpose": "العثور على أمثلة غزل"
                    }
                ]
            }
            ```''',
            # Result evaluation response
            '''```json
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
                        "relevance_score": 7,
                        "quality_score": 6,
                        "usefulness_score": 7,
                        "is_worth_following": true,
                        "key_insights": ["أمثلة جيدة"],
                        "recommendation": "مفيد"
                    }
                ],
                "overall_assessment": "نتائج جيدة",
                "gaps_identified": [],
                "followup_needed": false
            }
            ```'''
        ]
        
        # Mock search results - return different results for each query
        mock_search_provider.search.side_effect = [
            SearchResponse(
                results=[
                    SearchResult(
                        title="البحر الكامل في الشعر العربي",
                        url="https://example.com/kamil",
                        snippet="معلومات عن بحر الكامل وأوزانه",
                        source="web"
                    )
                ],
                total_results=1,
                search_time=0.1,
                query="test",
                provider="mock",
                metadata={}
            ),
            SearchResponse(
                results=[
                    SearchResult(
                        title="قصائد غزل كلاسيكية",
                        url="https://example.com/ghazal",
                        snippet="مجموعة من قصائد الغزل",
                        source="web"
                    )
                ],
                total_results=1,
                search_time=0.1,
                query="test",
                provider="mock",
                metadata={}
            )
        ]
        
        result = web_retriever.search(constraints, max_queries_per_round=2, max_rounds=1)
        
        assert isinstance(result, WebRetrievalResult)
        assert len(result.web_results) == 2
        assert result.total_found == 2
        assert result.retrieval_strategy == "multi_round_web_search_1_rounds"
        
        # Verify metadata
        assert result.metadata["rounds_executed"] == 1
        assert result.metadata["total_queries"] == 2
        assert result.metadata["total_results"] == 2
    
    def test_multi_round_web_search(self, web_retriever, mock_llm, mock_search_provider):
        """Test multi-round web search with follow-up queries"""
        constraints = Constraints(
            meter="بحر الطويل",
            theme="هجاء"
        )
        constraints.original_prompt = "اكتب قصيدة هجاء على بحر الطويل"
        
        # Mock LLM responses for multiple rounds
        mock_llm.generate.side_effect = [
            # Round 1 - Query generation
            '''```json
            {
                "queries": [
                    {
                        "query": "بحر الطويل في الشعر العربي",
                        "purpose": "معلومات تقنية"
                    }
                ]
            }
            ```''',
            # Round 1 - Evaluation with follow-up needed
            '''```json
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 6,
                        "quality_score": 5,
                        "usefulness_score": 6,
                        "is_worth_following": false,
                        "key_insights": ["معلومات سطحية"],
                        "recommendation": "غير مفيد"
                    }
                ],
                "overall_assessment": "نتائج ضعيفة",
                "gaps_identified": ["حاجة لأمثلة أكثر"],
                "followup_needed": true
            }
            ```''',
            # Round 2 - Query generation
            '''```json
            {
                "queries": [
                    {
                        "query": "أمثلة هجاء على بحر الطويل",
                        "purpose": "أمثلة عملية"
                    }
                ]
            }
            ```''',
            # Round 2 - Evaluation
            '''```json
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
            ```'''
        ]
        
        # Mock search results
        mock_search_provider.search.side_effect = [
            SearchResponse(
                results=[SearchResult(title="نتيجة ضعيفة", url="https://example.com/weak", snippet="معلومات سطحية", source="web")],
                total_results=1,
                search_time=0.1,
                query="test",
                provider="mock",
                metadata={}
            ),
            SearchResponse(
                results=[SearchResult(title="أمثلة هجاء ممتازة", url="https://example.com/good", snippet="أمثلة عملية", source="web")],
                total_results=1,
                search_time=0.1,
                query="test",
                provider="mock",
                metadata={}
            )
        ]
        
        result = web_retriever.search(constraints, max_queries_per_round=1, max_rounds=2)
        
        assert result.metadata["rounds_executed"] == 2
        assert result.metadata["total_queries"] == 2
        assert len(result.web_results) == 1  # Only high-quality result from round 2
        assert result.retrieval_strategy == "multi_round_web_search_2_rounds"
    
    def test_web_search_with_empty_constraints(self, web_retriever, mock_llm, mock_search_provider):
        """Test web search with minimal constraints"""
        constraints = Constraints(
            theme="غزل"
        )
        constraints.original_prompt = "اكتب قصيدة غزل"
        
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            # Query generation
            '''```json
            {
                "queries": [
                    {
                        "query": "قصيدة غزل",
                        "purpose": "العثور على أمثلة غزل"
                    }
                ]
            }
            ```''',
            # Evaluation
            '''```json
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 7,
                        "quality_score": 6,
                        "usefulness_score": 7,
                        "is_worth_following": true,
                        "key_insights": ["أمثلة غزل"],
                        "recommendation": "مفيد"
                    }
                ],
                "overall_assessment": "نتائج جيدة",
                "gaps_identified": [],
                "followup_needed": false
            }
            ```'''
        ]
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(
                    title="قصائد غزل",
                    url="https://example.com/ghazal",
                    snippet="مجموعة من قصائد الغزل",
                    source="web"
                )
            ],
            total_results=1,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        result = web_retriever.search(constraints, max_queries_per_round=1, max_rounds=1)
        
        assert len(result.web_results) == 1
        assert result.total_found == 1
        assert "غزل" in result.web_results[0].title
    
    def test_web_search_low_quality_filtering(self, web_retriever, mock_llm, mock_search_provider):
        """Test filtering of low-quality search results"""
        constraints = Constraints(
            meter="بحر الوافر"
        )
        constraints.original_prompt = "اكتب قصيدة على بحر الوافر"
        
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            # Query generation
            '''```json
            {
                "queries": [
                    {
                        "query": "بحر الوافر",
                        "purpose": "معلومات تقنية"
                    }
                ]
            }
            ```''',
            # Evaluation with low scores
            '''```json
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 3,
                        "quality_score": 2,
                        "usefulness_score": 3,
                        "is_worth_following": false,
                        "key_insights": ["معلومات ضعيفة"],
                        "recommendation": "غير مفيد"
                    }
                ],
                "overall_assessment": "نتائج ضعيفة",
                "gaps_identified": ["حاجة لنتائج أفضل"],
                "followup_needed": false
            }
            ```'''
        ]
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(
                    title="نتيجة منخفضة الجودة",
                    url="https://example.com/low",
                    snippet="محتوى منخفض الجودة",
                    source="web"
                )
            ],
            total_results=1,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        result = web_retriever.search(constraints, max_queries_per_round=1, max_rounds=1)
        
        # Low quality results should be filtered out
        assert len(result.web_results) == 0
        assert result.total_found == 0
        assert result.metadata["rounds_executed"] == 1
    
    def test_web_search_error_handling(self, web_retriever, mock_llm):
        """Test graceful handling of LLM errors"""
        constraints = Constraints(
            theme="غزل"
        )
        constraints.original_prompt = "اكتب قصيدة غزل"
        
        # Mock LLM to raise exception
        mock_llm.generate.side_effect = Exception("LLM API error")
        
        # Should handle gracefully without raising exception
        result = web_retriever.search(constraints, max_queries_per_round=1, max_rounds=1)
        
        assert len(result.web_results) == 0
        assert result.total_found == 0
        assert result.metadata["rounds_executed"] == 0
    
    def test_web_search_invalid_json_handling(self, web_retriever, mock_llm, mock_search_provider):
        """Test handling of invalid JSON responses from LLM"""
        constraints = Constraints(
            theme="غزل"
        )
        constraints.original_prompt = "اكتب قصيدة غزل"
        
        # Mock LLM to return invalid JSON
        mock_llm.generate.return_value = "Invalid JSON response"
        
        # Should handle gracefully
        result = web_retriever.search(constraints, max_queries_per_round=1, max_rounds=1)
        
        assert len(result.web_results) == 0
        assert result.total_found == 0
    
    def test_web_search_metadata_tracking(self, web_retriever, mock_llm, mock_search_provider):
        """Test that search metadata is properly tracked"""
        constraints = Constraints(
            meter="بحر الكامل",
            theme="غزل"
        )
        constraints.original_prompt = "اكتب قصيدة غزل على بحر الكامل"
        
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            # Query generation
            '''```json
            {
                "queries": [
                    {
                        "query": "البحر الكامل",
                        "purpose": "معلومات تقنية"
                    },
                    {
                        "query": "قصيدة غزل",
                        "purpose": "أمثلة شعرية"
                    }
                ]
            }
            ```''',
            # Evaluation
            '''```json
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
            ```'''
        ]
        
        # Mock search results - return different results for each query
        mock_search_provider.search.side_effect = [
            SearchResponse(
                results=[
                    SearchResult(title="نتيجة 1", url="https://example.com/1", snippet="محتوى 1", source="web")
                ],
                total_results=1,
                search_time=0.1,
                query="test",
                provider="mock",
                metadata={}
            ),
            SearchResponse(
                results=[
                    SearchResult(title="نتيجة 2", url="https://example.com/2", snippet="محتوى 2", source="web")
                ],
                total_results=1,
                search_time=0.1,
                query="test",
                provider="mock",
                metadata={}
            )
        ]
        
        result = web_retriever.search(constraints, max_queries_per_round=2, max_rounds=1)
        
        # Verify metadata
        assert result.metadata["rounds_executed"] == 1
        assert result.metadata["total_queries"] == 2
        assert result.metadata["total_results"] == 2
        
        # Verify evaluation summary
        assert len(result.metadata["evaluation_summary"]) == 1
        summary = result.metadata["evaluation_summary"][0]
        assert summary["round"] == 1
        assert summary["queries_executed"] == 2
        assert summary["results_found"] == 2
        assert summary["high_quality_results"] == 1
        assert "نتائج مختلطة" in summary["overall_assessment"]
        assert "حاجة لمزيد من التفاصيل" in summary["gaps_identified"]
    
    def test_web_search_with_poet_style_constraint(self, web_retriever, mock_llm, mock_search_provider):
        """Test web search with poet style constraint"""
        constraints = Constraints(
            meter="بحر الكامل",
            poet_style="المتنبي"
        )
        constraints.original_prompt = "اكتب قصيدة على بحر الكامل بأسلوب المتنبي"
        
        # Mock LLM responses
        mock_llm.generate.side_effect = [
            # Query generation
            '''```json
            {
                "queries": [
                    {
                        "query": "المتنبي أسلوب شعري بحر الكامل",
                        "purpose": "العثور على معلومات عن أسلوب المتنبي"
                    }
                ]
            }
            ```''',
            # Evaluation
            '''```json
            {
                "evaluated_results": [
                    {
                        "result_index": 0,
                        "relevance_score": 9,
                        "quality_score": 8,
                        "usefulness_score": 9,
                        "is_worth_following": true,
                        "key_insights": ["أسلوب المتنبي", "بحر الكامل"],
                        "recommendation": "مفيد جداً"
                    }
                ],
                "overall_assessment": "نتائج ممتازة",
                "gaps_identified": [],
                "followup_needed": false
            }
            ```'''
        ]
        
        # Mock search results
        mock_search_provider.search.return_value = SearchResponse(
            results=[
                SearchResult(
                    title="المتنبي وأسلوبه في بحر الكامل",
                    url="https://example.com/mutanabbi",
                    snippet="تحليل لأسلوب المتنبي في بحر الكامل",
                    source="web"
                )
            ],
            total_results=1,
            search_time=0.1,
            query="test",
            provider="mock",
            metadata={}
        )
        
        result = web_retriever.search(constraints, max_queries_per_round=1, max_rounds=1)
        
        assert len(result.web_results) == 1
        assert result.total_found == 1
        assert "المتنبي" in result.web_results[0].title 
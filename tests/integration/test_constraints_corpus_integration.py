# tests/integration/test_constraints_corpus_integration.py

import pytest
from poet.models.constraints import UserConstraints
from poet.analysis.knowledge_retriever import KnowledgeRetriever, RetrievalResult
from poet.data.corpus_manager import PoemRecord


class TestConstraintsCorpusIntegration:
    """Integration tests for constraints and corpus search via KnowledgeRetriever"""
    
    @pytest.fixture
    def knowledge_retriever(self, corpus_manager):
        """Create KnowledgeRetriever with test corpus"""
        return KnowledgeRetriever(corpus_manager)
    
    def test_basic_constraint_to_corpus_retrieval(self, knowledge_retriever):
        """Test basic constraint parsing to corpus retrieval"""
        constraints = UserConstraints(
            meter="بحر الكامل",
            theme="غزل",
            line_count=2
        )
        
        result = knowledge_retriever.retrieve_examples(constraints, max_examples=3)
        
        assert isinstance(result, RetrievalResult)
        assert len(result.examples) <= 3
        assert all(isinstance(ex, PoemRecord) for ex in result.examples)
        assert result.retrieval_strategy in ["best_match_exact", "best_match_mixed"]
        
        # Check that examples match constraints
        for example in result.examples:
            assert "الكامل" in example.meter or "غزل" in example.theme
    
    def test_exact_match_strategy(self, knowledge_retriever):
        """Test exact match retrieval strategy"""
        constraints = UserConstraints(
            meter="بحر الكامل",
            theme="غزل"
        )
        
        result = knowledge_retriever.retrieve_examples(
            constraints, 
            max_examples=5, 
            strategy="exact_match"
        )
        
        assert result.retrieval_strategy == "exact_match"
        assert result.search_criteria.search_mode == "AND"
        
        # All examples should match both constraints
        for example in result.examples:
            assert "الكامل" in example.meter
            assert "غزل" in example.theme
    
    def test_diverse_strategy(self, knowledge_retriever):
        """Test diverse retrieval strategy"""
        constraints = UserConstraints(
            meter="بحر الكامل",
            theme="غزل",
            poet_style="ابن المعتز"
        )
        
        result = knowledge_retriever.retrieve_examples(
            constraints,
            max_examples=4,
            strategy="diverse"
        )
        
        assert result.retrieval_strategy == "diverse"
        assert "meter_examples" in result.metadata
        assert "theme_examples" in result.metadata
        assert len(result.examples) <= 4
    
    def test_constraint_feasibility_validation(self, knowledge_retriever):
        """Test constraint feasibility validation"""
        # Valid constraints
        valid_constraints = UserConstraints(
            meter="بحر الكامل",
            theme="غزل"
        )
        
        validation = knowledge_retriever.validate_constraints_feasibility(valid_constraints)
        assert validation["feasible"] is True
        assert len(validation["issues"]) == 0
        
        # Invalid constraints
        invalid_constraints = UserConstraints(
            meter="بحر غير موجود",
            theme="موضوع غير موجود"
        )
        
        validation = knowledge_retriever.validate_constraints_feasibility(invalid_constraints)
        assert validation["feasible"] is False
        assert len(validation["issues"]) > 0
        assert any("not found in corpus" in issue for issue in validation["issues"])
    
    def test_constraint_statistics(self, knowledge_retriever):
        """Test constraint statistics generation"""
        constraints = UserConstraints(
            meter="بحر الكامل",
            theme="غزل",
            poet_style="ابن المعتز"
        )
        
        stats = knowledge_retriever.get_constraint_statistics(constraints)
        
        assert "meter_matches" in stats
        assert "theme_matches" in stats
        assert "poet_matches" in stats
        assert "exact_matches" in stats
        assert "partial_matches" in stats
        assert "coverage" in stats
        
        assert stats["meter_matches"] >= 0
        assert stats["exact_matches"] <= stats["partial_matches"]
        assert 0 <= stats["coverage"]["exact_percentage"] <= 100
    
    def test_alternative_suggestions(self, knowledge_retriever):
        """Test alternative constraint suggestions"""
        constraints = UserConstraints(
            meter="بحر نادر",
            theme="موضوع نادر"
        )
        
        suggestions = knowledge_retriever.suggest_alternatives(constraints)
        
        assert "meters" in suggestions
        assert "themes" in suggestions
        assert "poets" in suggestions
        
        assert isinstance(suggestions["meters"], list)
        assert isinstance(suggestions["themes"], list)
        assert isinstance(suggestions["poets"], list)
    
    def test_constraints_to_search_criteria_conversion(self, knowledge_retriever):
        """Test conversion from UserConstraints to SearchCriteria"""
        constraints = UserConstraints(
            meter="بحر الكامل",
            theme="غزل",
            line_count=3,
            keywords=["حب", "شوق"],
            register="فصحى",
            era="كلاسيكي",
            poet_style="المتنبي"
        )
        
        # Test AND mode
        and_criteria = knowledge_retriever._constraints_to_search_criteria(constraints, mode="AND")
        
        assert and_criteria.meter == "بحر الكامل"
        assert and_criteria.theme == "غزل"
        assert and_criteria.poet_name == "المتنبي"
        assert and_criteria.poet_era == "كلاسيكي"
        assert and_criteria.language_type == "فصحى"
        assert and_criteria.min_verses == 2
        assert and_criteria.max_verses == 4
        assert and_criteria.keywords == ["حب", "شوق"]
        assert and_criteria.search_mode == "AND"
        
        # Test OR mode
        or_criteria = knowledge_retriever._constraints_to_search_criteria(constraints, mode="OR")
        assert or_criteria.search_mode == "OR"
    
    def test_best_match_fallback_logic(self, knowledge_retriever):
        """Test best match strategy fallback from AND to OR"""
        # Create constraints that likely won't have many exact matches
        constraints = UserConstraints(
            meter="بحر الطويل",
            theme="مدح",
            poet_style="المتنبي"
        )
        
        result = knowledge_retriever.retrieve_examples(
            constraints,
            max_examples=5,
            strategy="best_match"
        )
        
        # Should have some results due to fallback logic
        assert len(result.examples) > 0
        assert result.retrieval_strategy in ["best_match_exact", "best_match_mixed"]
        
        if result.retrieval_strategy == "best_match_mixed":
            assert "exact_matches" in result.metadata
            assert "total_candidates" in result.metadata
    
    def test_empty_constraints_handling(self, knowledge_retriever):
        """Test handling of empty/minimal constraints"""
        empty_constraints = UserConstraints()
        
        result = knowledge_retriever.retrieve_examples(empty_constraints, max_examples=3)
        
        # Should still return some examples (random sampling)
        assert len(result.examples) > 0
        assert isinstance(result, RetrievalResult)
    
    def test_line_count_constraint_mapping(self, knowledge_retriever):
        """Test line count constraint mapping to verse count search"""
        constraints = UserConstraints(
            line_count=2
        )
        
        search_criteria = knowledge_retriever._constraints_to_search_criteria(constraints)
        
        assert search_criteria.min_verses == 1
        assert search_criteria.max_verses == 3
        
        result = knowledge_retriever.retrieve_examples(constraints, max_examples=3)
        
        # Should find poems with approximately 2 verses
        for example in result.examples:
            verse_count = example.get_verse_count()
            assert 1 <= verse_count <= 3
    
    def test_multiple_retrieval_strategies_consistency(self, knowledge_retriever):
        """Test that different strategies return reasonable results for same constraints"""
        constraints = UserConstraints(
            meter="بحر الكامل",
            theme="غزل"
        )
        
        exact_result = knowledge_retriever.retrieve_examples(
            constraints, max_examples=3, strategy="exact_match"
        )
        
        diverse_result = knowledge_retriever.retrieve_examples(
            constraints, max_examples=3, strategy="diverse"
        )
        
        best_result = knowledge_retriever.retrieve_examples(
            constraints, max_examples=3, strategy="best_match"
        )
        
        # All should return some results
        assert len(exact_result.examples) >= 0
        assert len(diverse_result.examples) >= 0
        assert len(best_result.examples) >= 0
        
        # Strategies should be recorded correctly
        assert exact_result.retrieval_strategy == "exact_match"
        assert diverse_result.retrieval_strategy == "diverse"
        assert best_result.retrieval_strategy in ["best_match_exact", "best_match_mixed"] 
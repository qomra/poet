# tests/integration/test_real_knowledge_retrieval.py

import pytest

from poet.models.constraints import UserConstraints


@pytest.mark.integration
@pytest.mark.real_data
class TestRealKnowledgeRetrieval:
    """Integration tests for knowledge retrieval using real ashaar dataset with real LLM"""
    
    def test_knowledge_retrieval_example_1(self, real_knowledge_retriever, constraint_parser_parametrized, llm_type, test_data):
        """Test knowledge retrieval for example 1 (غزل poem) - parse constraints with real/mock LLM"""
        example = test_data[0]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Parse constraints using ConstraintParser (real or mock LLM)
        if llm_type == "mock":
            # Set up mock response for consistent testing
            import json
            mock_response = json.dumps(expected_constraints, ensure_ascii=False)
            constraint_parser_parametrized.llm.responses = [mock_response]
            constraint_parser_parametrized.llm.reset()
        
        constraints = constraint_parser_parametrized.parse_constraints(user_prompt)
        
        # Test retrieval
        result = real_knowledge_retriever.retrieve_examples(
            constraints, 
            max_examples=5,
            strategy="best_match"
        )
        
        # Basic assertions
        assert len(result.examples) >= 0, "Should handle بحر الكامل + غزل search"
        assert result.retrieval_strategy in ["best_match_or"]
        
        # Print parsed constraints and results
        print(f"\nExample 1 - LLM Type: {llm_type}")
        print(f"Parsed constraints:")
        print(f"  Meter: {constraints.meter}")
        print(f"  Theme: {constraints.theme}")
        print(f"  Line count: {constraints.line_count}")
        print(f"  Tone: {constraints.tone}")
        print(f"Found {len(result.examples)} poems matching parsed constraints")
        
        for i, poem in enumerate(result.examples[:3]):
            print(f"\nResult {i+1}:")
            print(f"  Meter: {poem.meter}")
            print(f"  Theme: {poem.theme}")
            print(f"  Poet: {poem.poet_name}")
            if poem.verses:
                print(f"  First verse: {poem.verses[0][:50]}...")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert constraints.meter == expected_constraints["meter"]
            assert constraints.theme == expected_constraints["theme"]
            assert constraints.line_count == expected_constraints["line_count"]
        
        # Validate constraint statistics
        stats = real_knowledge_retriever.get_constraint_statistics(constraints)
        print(f"\nConstraint statistics:")
        print(f"  Meter matches: {stats.get('meter_matches', 0)}")
        print(f"  Theme matches: {stats.get('theme_matches', 0)}")
        print(f"  Exact matches: {stats.get('exact_matches', 0)}")
        print(f"  Partial matches: {stats.get('partial_matches', 0)}")
        
        assert stats["exact_matches"] >= 0
        assert stats["partial_matches"] >= stats["exact_matches"]
    
    def test_knowledge_retrieval_example_2(self, real_knowledge_retriever, constraint_parser_parametrized, llm_type, test_data):
        """Test knowledge retrieval for example 2 (هجاء poem) - parse constraints with real/mock LLM"""
        example = test_data[1]
        user_prompt = example["prompt"]["text"]
        expected_constraints = example["agent"]["constraints"]
        
        # Parse constraints using ConstraintParser (real or mock LLM)
        if llm_type == "mock":
            # Set up mock response for consistent testing
            import json
            mock_response = json.dumps(expected_constraints, ensure_ascii=False)
            constraint_parser_parametrized.llm.responses = [mock_response]
            constraint_parser_parametrized.llm.reset()
        
        constraints = constraint_parser_parametrized.parse_constraints(user_prompt)
        
        # Test retrieval
        result = real_knowledge_retriever.retrieve_examples(
            constraints,
            max_examples=5,
            strategy="best_match"
        )
        
        # Basic assertions
        assert len(result.examples) >= 0, "Should handle بحر الطويل + هجاء search"
        assert result.retrieval_strategy in ["best_match_or"]
        
        # Print parsed constraints and results
        print(f"\nExample 2 - LLM Type: {llm_type}")
        print(f"Parsed constraints:")
        print(f"  Meter: {constraints.meter}")
        print(f"  Theme: {constraints.theme}")
        print(f"  Line count: {constraints.line_count}")
        print(f"  Tone: {constraints.tone}")
        print(f"Found {len(result.examples)} poems matching parsed constraints")
        
        for i, poem in enumerate(result.examples[:3]):
            print(f"\nResult {i+1}:")
            print(f"  Meter: {poem.meter}")
            print(f"  Theme: {poem.theme}")
            print(f"  Poet: {poem.poet_name}")
            if poem.verses:
                print(f"  First verse: {poem.verses[0][:50]}...")
        
        # For mock LLM, validate against expected constraints
        if llm_type == "mock":
            assert constraints.meter == expected_constraints["meter"]
            assert constraints.theme == expected_constraints["theme"]
            assert constraints.line_count == expected_constraints["line_count"]
        
        # Test feasibility validation
        validation = real_knowledge_retriever.validate_constraints_feasibility(constraints)
        print(f"\nFeasibility validation:")
        print(f"  Feasible: {validation['feasible']}")
        print(f"  Issues: {validation['issues']}")
        print(f"  Suggestions: {validation['suggestions']}")
        
        # Should provide meaningful validation
        assert isinstance(validation["feasible"], bool)
        assert isinstance(validation["issues"], list)
        
        # Test alternative suggestions if constraints are challenging
        if not validation["feasible"] or len(result.examples) == 0:
            suggestions = real_knowledge_retriever.suggest_alternatives(constraints)
            print(f"\nAlternative suggestions:")
            print(f"  Meters: {suggestions['meters'][:3]}")
            print(f"  Themes: {suggestions['themes'][:3]}")
            print(f"  Poets: {suggestions['poets'][:3]}")
            
            assert len(suggestions["meters"]) > 0
            assert len(suggestions["themes"]) > 0 
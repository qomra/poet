# tests/integration/test_real_enricher_integration.py

import pytest
from pathlib import Path

from poet.models.constraints import Constraints, ExampleData
from poet.models.search import CorpusExample, WebExample
from poet.data.enricher import DataEnricher


@pytest.fixture
def enricher_with_corpus_config():
    """DataEnricher instance with corpus configuration"""
    enricher = DataEnricher()
    enricher.config = {
        'sources': [
            {
                'type': 'corpus',
                'local_knowledge_path': 'dataset',  # Will be mocked in test
                'top_k': 5,
                'search_criteria': ['meter', 'qafiya']
            }
        ]
    }
    return enricher


@pytest.fixture
def local_corpus_manager():
    """Create CorpusManager with local dataset directory"""
    from poet.data.corpus_manager import CorpusManager
    
    dataset_path = Path("dataset")
    if not dataset_path.exists():
        pytest.skip("Local dataset/ directory not found")
    
    print(f"Creating corpus manager with path: {dataset_path}")
    print(f"Ashaar directory exists: {(dataset_path / 'ashaar').exists()}")
    
    try:
        corpus_manager = CorpusManager(local_knowledge_path=str(dataset_path))
        print(f"Corpus manager created successfully")
        
        corpus_manager.load_corpus()
        print(f"Loaded corpus manager with {corpus_manager.get_total_poems()} poems")
        
        # Check if dataset was loaded
        if hasattr(corpus_manager, '_dataset') and corpus_manager._dataset:
            print(f"Dataset object exists: {type(corpus_manager._dataset)}")
            print(f"Dataset length: {len(corpus_manager._dataset)}")
        else:
            print("No dataset object found")
            
        return corpus_manager
        
    except Exception as e:
        print(f"Error creating corpus manager: {e}")
        import traceback
        traceback.print_exc()
        raise


@pytest.mark.integration
class TestRealEnricherIntegration:
    """Integration tests for DataEnricher using local corpus data"""
    
    def test_enricher_integration_example_1(self, enricher_with_corpus_config, local_corpus_manager, test_data):
        """Test enricher integration for example 1 (غزل poem) - corpus enrichment test"""
        example = test_data[0]
        expected_constraints = example["agent"]["constraints"]
        
        # Create constraints directly from fixture data
        constraints = Constraints(
            meter=expected_constraints["meter"],
            qafiya=expected_constraints["qafiya"],
            line_count=expected_constraints["line_count"],
            theme=expected_constraints["theme"],
            tone=expected_constraints["tone"],
            imagery=expected_constraints.get("imagery", []),
            keywords=expected_constraints.get("keywords", []),
            register=expected_constraints.get("register"),
            era=expected_constraints.get("era"),
            poet_style=expected_constraints.get("poet_style"),
            sections=expected_constraints.get("sections", []),
            ambiguities=expected_constraints.get("ambiguities", [])
        )
        
        # Create context (no LLM needed for corpus-only enrichment)
        context = {}
        
        # Mock the corpus manager initialization to use local one
        from unittest.mock import patch
        with patch.object(enricher_with_corpus_config, '_initialize_retrievers') as mock_init:
            def setup_local_retriever(sources, ctx):
                from poet.analysis.knowledge_retriever import CorpusKnowledgeRetriever
                enricher_with_corpus_config._corpus_retriever = CorpusKnowledgeRetriever(local_corpus_manager)
                
            mock_init.side_effect = setup_local_retriever
            
            # Test enrichment
            input_data = {'constraints': constraints}
            result = enricher_with_corpus_config.run(input_data, context)
        
        # Basic assertions
        assert result['enrichment_performed'] is True
        assert result['enriched_data_available'] is True
        enriched_constraints = result['constraints']
        
        # Validate example_data structure
        assert enriched_constraints.example_data is not None
        assert isinstance(enriched_constraints.example_data, ExampleData)
        
        # Validate corpus examples
        corpus_examples = enriched_constraints.example_data.corpus_examples
        assert isinstance(corpus_examples, list)
        
        if len(corpus_examples) > 0:
            # Validate first corpus example
            first_example = corpus_examples[0]
            assert isinstance(first_example, CorpusExample)
            
            # Validate DataExample methods work
            formatted_content = first_example.get_formatted_content()
            assert isinstance(formatted_content, str)
            assert len(formatted_content) > 0
            
            source_desc = first_example.get_source_description()
            assert isinstance(source_desc, str)
            assert "corpus:" in source_desc
            
            # Validate search criteria
            assert isinstance(first_example.search_criteria, list)
            assert len(first_example.search_criteria) > 0
            
            # Validate poem fields
            assert isinstance(first_example.title, str)
            assert isinstance(first_example.verses, str)
            assert isinstance(first_example.meter, str)
            assert isinstance(first_example.qafiya, str)
        
        # Validate metadata
        retrieval_metadata = enriched_constraints.example_data.retrieval_metadata
        assert isinstance(retrieval_metadata, dict)
        
        if 'corpus' in retrieval_metadata:
            corpus_meta = retrieval_metadata['corpus']
            assert 'total_found' in corpus_meta
            assert 'strategy' in corpus_meta
            assert 'search_criteria_used' in corpus_meta
            assert 'examples_count' in corpus_meta
            
            assert corpus_meta['search_criteria_used'] == ['meter', 'qafiya']
            assert corpus_meta['examples_count'] == len(corpus_examples)
        
        # Print results for debugging
        print(f"\nExample 1 Enricher Test")
        print(f"Input constraints:")
        print(f"  Meter: {constraints.meter}")
        print(f"  Theme: {constraints.theme}")
        print(f"  Qafiya: {constraints.qafiya}")
        print(f"  Line count: {constraints.line_count}")
        
        print(f"\nEnrichment results:")
        print(f"  Corpus examples found: {len(corpus_examples)}")
        print(f"  Web examples found: {len(enriched_constraints.example_data.web_examples)}")
        
        for i, example in enumerate(corpus_examples[:3]):
            print(f"\nCorpus Example {i+1}:")
            print(f"  Title: {example.title}")
            print(f"  Meter: {example.meter}")
            print(f"  Qafiya: {example.qafiya}")
            print(f"  Poet: {example.poet_name}")
            print(f"  Search criteria: {example.search_criteria}")
            if example.verses:
                verses_lines = example.verses.split('\n')
                print(f"  First verse: {verses_lines[0][:50]}...")
        
        # Test serialization roundtrip
        constraints_dict = enriched_constraints.to_dict()
        reconstructed_constraints = Constraints.from_dict(constraints_dict)
        
        # Validate reconstructed constraints maintain DataExample objects
        assert reconstructed_constraints.example_data is not None
        assert isinstance(reconstructed_constraints.example_data, ExampleData)
        
        if len(corpus_examples) > 0:
            reconstructed_examples = reconstructed_constraints.example_data.corpus_examples
            assert len(reconstructed_examples) == len(corpus_examples)
            assert isinstance(reconstructed_examples[0], CorpusExample)
            
            # Test that methods still work after serialization/deserialization
            formatted_content = reconstructed_examples[0].get_formatted_content()
            assert isinstance(formatted_content, str)
            assert len(formatted_content) > 0
        
        print(f"\nSerialization test passed: DataExample objects maintained through roundtrip") 
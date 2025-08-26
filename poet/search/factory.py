# poet/search/factory.py

from typing import Dict, Any, Type
from poet.core.node import Node
from poet.search.best_of_n_node import BestOfNNode


def create_best_of_n_node(node_type: str, node_config: Dict[str, Any], 
                         search_config: Dict[str, Any], llm, prompt_manager) -> BestOfNNode:
    """
    Factory function to create BestOfN nodes with different underlying node types.
    
    Args:
        node_type: Type of underlying node ('generation', 'evaluation', 'refiner_chain')
        node_config: Configuration for the underlying node
        search_config: Configuration for the search strategy
        llm: LLM instance for the underlying nodes
        prompt_manager: Prompt manager for the underlying nodes
        
    Returns:
        Configured BestOfNNode
    """
    
    # Create the underlying node based on type
    underlying_node = _create_underlying_node(node_type, node_config, llm, prompt_manager)
    
    # Extract search parameters
    n_candidates = search_config.get('n_candidates', 5)
    selection_prompt = search_config.get('selection_prompt', f'{node_type}_selection')
    selection_metric = search_config.get('selection_metric', 'overall_score')
    temperature_range = search_config.get('temperature_range', [0.5, 0.7, 0.9, 1.1, 1.3])
    
    # Create BestOfN node
    return BestOfNNode(
        underlying_node=underlying_node,
        n_candidates=n_candidates,
        selection_prompt=selection_prompt,
        selection_metric=selection_metric,
        temperature_range=temperature_range
    )


def _create_underlying_node(node_type: str, node_config: Dict[str, Any], llm, prompt_manager) -> Node:
    """Create the underlying node based on type."""
    
    if node_type == 'generation':
        from poet.generation.poem_generator import SimplePoemGenerator
        return SimplePoemGenerator(llm=llm, prompt_manager=prompt_manager, **node_config)
        
    elif node_type == 'evaluation':
        from poet.evaluation.poem import PoemEvaluator
        return PoemEvaluator(llm=llm, prompt_manager=prompt_manager, **node_config)
        
    elif node_type == 'refiner_chain':
        from poet.refinement.refiner_chain import RefinerChain
        return RefinerChain(llm=llm, **node_config)
        
    else:
        raise ValueError(f"Unknown node type: {node_type}")


def get_node_mapping() -> Dict[str, Type[Node]]:
    """Get mapping of node names to their classes for pipeline registration."""
    return {
        'best_of_n_generation': BestOfNNode,
        'best_of_n_evaluation': BestOfNNode,
        'best_of_n_refiner_chain': BestOfNNode,
    }

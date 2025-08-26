import logging
from typing import List, Dict, Any, Optional
from poet.core.node import Node
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment


class RefinerChain(Node):
    """
    Configuration parser that builds a sequence of refinement nodes.
    
    This node doesn't execute refinement logic itself, but instead builds
    a flat sequence of individual nodes that can be executed by the pipeline.
    
    Example configuration:
    refiner_chain:
      refiners: [prosody_refiner, qafiya_refiner]
      max_iterations: 3
      target_quality: 0.85
    
    This translates to 9 nodes:
    - poem_evaluation_1, prosody_refiner_1, qafiya_refiner_1
    - poem_evaluation_2, prosody_refiner_2, qafiya_refiner_2  
    - poem_evaluation_3, prosody_refiner_3, qafiya_refiner_3
    """
    
    def __init__(self, llm, refiners=None, max_iterations: int = 3, target_quality: float = 0.8, **kwargs):
        kwargs.pop('max_iterations', None)  # Avoid duplicate parameter
        super().__init__(**kwargs)
        
        self.llm = llm
        self.max_iterations = max_iterations
        self.target_quality = target_quality
        
        # Handle refiners parameter - can be list of strings (names) or actual refiner objects
        if isinstance(refiners, list) and all(isinstance(r, str) for r in refiners):
            self.refiner_names = refiners
        else:
            self.refiner_names = ['prosody_refiner', 'qafiya_refiner']
    
    def build_refinement_sequence(self, context: Dict[str, Any]) -> List[Node]:
        """
        Build a flat sequence of refinement nodes.
        
        Args:
            context: Pipeline context
            
        Returns:
            List of nodes representing the refinement sequence
        """
        refinement_nodes = []
        
        for iteration in range(1, self.max_iterations + 1):
            # Add evaluation node for this iteration
            evaluation_node = self._create_evaluation_node(iteration, context)
            refinement_nodes.append(evaluation_node)
            
            # Add each refiner for this iteration
            for refiner_name in self.refiner_names:
                refiner_node = self._create_refiner_node(refiner_name, iteration, context)
                refinement_nodes.append(refiner_node)
        
        self.logger.info(f"🔧 Built refinement sequence with {len(refinement_nodes)} nodes across {self.max_iterations} iterations")
        return refinement_nodes
    
    def _create_evaluation_node(self, iteration: int, context: Dict[str, Any]) -> Node:
        """Create evaluation node for specific iteration."""
        from poet.evaluation.poem import PoemEvaluator
        
        return PoemEvaluator(
            llm=self.llm,
            name=f"poem_evaluation_{iteration}",
            iteration=iteration,
            target_quality=self.target_quality,
            metrics=['prosody', 'qafiya']
        )
    
    def _create_refiner_node(self, refiner_name: str, iteration: int, context: Dict[str, Any]) -> Node:
        """Create refiner node for specific iteration."""
        try:
            # Check if this is a BestOfNNode wrapper
            if refiner_name.startswith('best_of_n_'):
                # Extract the underlying refiner name
                underlying_refiner = refiner_name.replace('best_of_n_', '')
                underlying_node = self._create_underlying_refiner(underlying_refiner, iteration, context)
                
                if underlying_node:
                    from poet.search.best_of_n_node import BestOfNNode
                    return BestOfNNode(
                        underlying_node=underlying_node,
                        name=refiner_name,
                        node_type='refiner_chain',
                        **context.get('refiner_config', {})
                    )
                else:
                    return None
            
            # Handle regular refiners
            return self._create_underlying_refiner(refiner_name, iteration, context)
                
        except ImportError as e:
            self.logger.error(f"📦 Failed to import {refiner_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"🔧 Failed to create refiner {refiner_name}: {e}")
            return None
    
    def _create_underlying_refiner(self, refiner_name: str, iteration: int, context: Dict[str, Any]) -> Node:
        """Create the underlying refiner node."""
        try:
            if refiner_name == 'prosody_refiner':
                from poet.refinement.prosody import ProsodyRefiner
                return ProsodyRefiner(
                    llm=self.llm,
                    prompt_manager=context.get('prompt_manager'),
                    name=f"prosody_refiner_{iteration}",
                    iteration=iteration
                )
            elif refiner_name == 'qafiya_refiner':
                from poet.refinement.qafiya import QafiyaRefiner
                return QafiyaRefiner(
                    llm=self.llm,
                    prompt_manager=context.get('prompt_manager'),
                    name=f"qafiya_refiner_{iteration}",
                    iteration=iteration
                )
            elif refiner_name == 'line_count_refiner':
                from poet.refinement.line_count import LineCountRefiner
                return LineCountRefiner(
                    llm=self.llm,
                    prompt_manager=context.get('prompt_manager'),
                    name=f"line_count_refiner_{iteration}",
                    iteration=iteration
                )
            elif refiner_name == 'tashkeel_refiner':
                from poet.refinement.tashkeel import TashkeelRefiner
                return TashkeelRefiner(
                    llm=self.llm,
                    prompt_manager=context.get('prompt_manager'),
                    name=f"tashkeel_refiner_{iteration}",
                    iteration=iteration
                )
            else:
                self.logger.warning(f"❓ Unknown refiner type: {refiner_name}")
                return None
                
        except ImportError as e:
            self.logger.error(f"📦 Failed to import {refiner_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"🔧 Failed to create refiner {refiner_name}: {e}")
            return None
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        This method should not be called directly.
        RefinerChain is a configuration parser, not an executable node.
        """
        raise NotImplementedError(
            "RefinerChain is a configuration parser and should not be executed directly. "
            "Use build_refinement_sequence() to get the actual nodes to execute."
        )
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations', 'refiner_chain_used', 'refiners_used']


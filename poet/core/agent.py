# poet/core/dynamic_agent.py

from typing import Dict, Any, Optional
import logging
from .pipeline import PipelineBuilder, PipelineEngine
from .node import Node
from pathlib import Path
from poet.prompts import get_global_prompt_manager


class DynamicAgent:
    """
    Dynamic agent that creates and runs pipelines from configuration.
    
    This agent replaces the hardcoded agent.py with a fully dynamic system
    that can create pipelines from configuration without any hardcoding.
    Uses flat compute graphs and node-level harmony.
    """
    
    def __init__(self, config: Dict[str, Any], llm, prompt_manager=None):
        self.config = config
        self.llm = llm
        # Use global prompt manager instead of instance-specific one
        self.prompt_manager = get_global_prompt_manager()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create pipeline builder and register all available nodes
        self.pipeline_builder = PipelineBuilder()
        self._register_nodes()
        
        # Build the pipeline from configuration
        self.pipeline = self._build_pipeline()
    
    def _register_nodes(self):
        """Register all available node types with the pipeline builder."""
        try:
            # Import and register all node classes
            from poet.analysis.constraint_parser import ConstraintParser
            from poet.analysis.qafiya_selector import QafiyaSelector
            from poet.analysis.bahr_selector import BahrSelector
            from poet.generation.poem_generator import SimplePoemGenerator
            from poet.data.enricher import DataEnricher
            from poet.refinement.refiner_chain import RefinerChain
            from poet.evaluation.poem import PoemEvaluator
            from poet.search.best_of_n_node import BestOfNNode
            
            # Register each node type
            self.pipeline_builder.register_node("constraints_parser", ConstraintParser)
            self.pipeline_builder.register_node("qafiya_selector", QafiyaSelector)
            self.pipeline_builder.register_node("bahr_selector", BahrSelector)
            self.pipeline_builder.register_node("data_enrichment", DataEnricher)
            self.pipeline_builder.register_node("generation", SimplePoemGenerator)
            self.pipeline_builder.register_node("refiner_chain", RefinerChain)
            self.pipeline_builder.register_node("evaluation", PoemEvaluator)
            
            # Register BestOfN search nodes
            self.pipeline_builder.register_node("best_of_n_generation", BestOfNNode)
            self.pipeline_builder.register_node("best_of_n_evaluation", BestOfNNode)
            
            self.logger.info("📋 All nodes registered successfully")
            
        except ImportError as e:
            self.logger.error(f"📦 Failed to import node classes: {e}")
            raise
        except Exception as e:
            self.logger.error(f"🔧 Failed to register nodes: {e}")
            raise
    
    def _build_pipeline(self) -> PipelineEngine:
        """Build the pipeline from configuration."""
        try:
            agent_config = self.config.get("agent", {})
            pipeline_config = agent_config.get("pipeline", [])
            
            # Create context for the pipeline
            context = {
                'llm': self.llm,
                'prompt_manager': self.prompt_manager,
                'config': self.config
            }
            
            # Build pipeline using the builder
            pipeline = self.pipeline_builder.build_pipeline(pipeline_config, context)
            
            self.logger.info(f"🏗️ Pipeline built successfully with {len(pipeline.compute_graph)} nodes")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"🏚️ Failed to build pipeline: {e}")
            raise
    
    def run_pipeline(self, user_prompt: [str|Dict[str, Any]], initial_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            user_prompt: User's poetry request
            initial_constraints: Optional initial constraints
            
        Returns:
            Pipeline results
        """
        self.logger.info("🚀 Starting dynamic pipeline execution")
        
        try:
            # Prepare initial input
            initial_input = {
                'user_prompt': user_prompt if isinstance(user_prompt, str) else user_prompt['prompt'],
                'poem_id': -1 if isinstance(user_prompt, str) else user_prompt['poem_id']
            }
            
            if initial_constraints:
                initial_input['initial_constraints'] = initial_constraints
            
            # Run the pipeline
            result = self.pipeline.run_pipeline(initial_input)
            
            # Add metadata
            result['user_prompt'] = user_prompt
            result['success'] = True
            result['pipeline_info'] = self.pipeline.get_pipeline_info()
            
            # Generate harmony using node-level data
            harmony_text = self.pipeline.generate_harmony(self.llm)
            result['harmony_reasoning'] = harmony_text
            
            self.logger.info("🎉 Dynamic pipeline completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"💀 Dynamic pipeline failed: {e}")
            return {
                'user_prompt': user_prompt,
                'success': False,
                'error': str(e)
            }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline."""
        return self.pipeline.get_pipeline_info()
    
    def update_pipeline_config(self, new_pipeline_config: list):
        """
        Update the pipeline configuration and rebuild.
        
        Args:
            new_pipeline_config: New pipeline configuration
        """
        try:
            # Update the config
            if 'agent' not in self.config:
                self.config['agent'] = {}
            self.config['agent']['pipeline'] = new_pipeline_config
            
            # Rebuild the pipeline
            self.pipeline = self._build_pipeline()
            
            self.logger.info("🔄 Pipeline updated and rebuilt successfully")
            
        except Exception as e:
            self.logger.error(f"🔄 Failed to update pipeline: {e}")
            raise

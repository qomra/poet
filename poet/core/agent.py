# poet/core/dynamic_agent.py

from typing import Dict, Any, Optional
import logging
from .pipeline import PipelineBuilder, PipelineEngine
from .node import Node
from pathlib import Path


class DynamicAgent:
    """
    Dynamic agent that creates and runs pipelines from configuration.
    
    This agent replaces the hardcoded agent.py with a fully dynamic system
    that can create pipelines from configuration without any hardcoding.
    """
    
    def __init__(self, config: Dict[str, Any], llm, prompt_manager=None):
        self.config = config
        self.llm = llm
        self.prompt_manager = prompt_manager
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

            from poet.refinement.refiner_chain import RefinerChain
            from poet.evaluation.poem import PoemEvaluator
            
            # Register each node type
            self.pipeline_builder.register_node("constraints_parser", ConstraintParser)
            self.pipeline_builder.register_node("qafiya_selector", QafiyaSelector)
            self.pipeline_builder.register_node("bahr_selector", BahrSelector)
            self.pipeline_builder.register_node("generation", SimplePoemGenerator)

            self.pipeline_builder.register_node("refiner_chain", RefinerChain)
            self.pipeline_builder.register_node("evaluation", PoemEvaluator)
            
            self.logger.info("All nodes registered successfully")
            
        except ImportError as e:
            self.logger.error(f"Failed to import node classes: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to register nodes: {e}")
            raise
    
    def _build_pipeline(self) -> PipelineEngine:
        """Build the pipeline from configuration."""
        try:
            agent_config = self.config.get("agent", {})
            pipeline_config = agent_config.get("pipeline", [])
            
            # Check if harmony capture is enabled
            harmony_enabled = self.config.get("agent", {}).get("compilers", {}).get("harmony", {}).get("enabled", False)
            
            # Create context for the pipeline
            context = {
                'llm': self.llm,
                'prompt_manager': self.prompt_manager,
                'config': self.config,
                'harmony_capture_enabled': harmony_enabled
            }
            
            # Build pipeline using the builder
            pipeline = self.pipeline_builder.build_pipeline(pipeline_config, context)
            
            self.logger.info(f"Pipeline built successfully with {len(pipeline.nodes)} nodes")
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to build pipeline: {e}")
            raise
    
    def run_pipeline(self, user_prompt: str, initial_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            user_prompt: User's poetry request
            initial_constraints: Optional initial constraints
            
        Returns:
            Pipeline results
        """
        self.logger.info("Starting dynamic pipeline execution")
        
        # Start harmony capture if enabled
        harmony_enabled = self.config.get("agent", {}).get("compilers", {}).get("harmony", {}).get("enabled", False)
        if harmony_enabled:
            from poet.logging.integration import HarmonyIntegration
            self.logger.info("Harmony capture enabled - starting execution capture")
            HarmonyIntegration.start_captured_execution(user_prompt, initial_constraints)
        
        try:
            # Prepare initial input
            initial_input = {
                'user_prompt': user_prompt
            }
            
            if initial_constraints:
                initial_input['initial_constraints'] = initial_constraints
            
            # Run the pipeline
            result = self.pipeline.run_pipeline(initial_input)
            
            # Add metadata
            result['user_prompt'] = user_prompt
            result['success'] = True
            result['pipeline_info'] = self.pipeline.get_pipeline_info()
            
            # Complete harmony capture if enabled
            if harmony_enabled:
                self.logger.info("Completing harmony capture and generating reasoning")
                harmony_config = self.config.get("agent", {}).get("compilers", {}).get("harmony", {})
                output_dir = Path(harmony_config.get("output_dir", "outputs/harmony"))
                
                # Get final poem and quality assessment from result
                final_poem = result.get('poem')
                quality_assessment = result.get('evaluation')
                
                harmony_reasoning = HarmonyIntegration.complete_and_reason(
                    llm=self.llm,
                    final_poem=final_poem,
                    quality_assessment=quality_assessment,
                    output_dir=output_dir
                )
                
                if harmony_reasoning:
                    result['harmony_reasoning'] = harmony_reasoning
                    self.logger.info(f"Harmony reasoning generated and saved to {output_dir}")
                else:
                    self.logger.warning("Failed to generate harmony reasoning")
            
            self.logger.info("Dynamic pipeline completed successfully")
            return result
            
        except Exception as e:
            # Complete harmony capture even on error
            if harmony_enabled:
                try:
                    from poet.logging.integration import HarmonyIntegration
                    harmony_config = self.config.get("agent", {}).get("compilers", {}).get("harmony", {})
                    output_dir = Path(harmony_config.get("output_dir", "outputs/harmony"))
                    
                    HarmonyIntegration.complete_and_reason(
                        llm=self.llm,
                        final_poem=None,
                        quality_assessment={'error': str(e)},
                        output_dir=output_dir
                    )
                except Exception as harmony_error:
                    self.logger.error(f"Failed to complete harmony capture on error: {harmony_error}")
            
            raise e
            self.logger.error(f"Dynamic pipeline failed: {e}")
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
            
            self.logger.info("Pipeline updated and rebuilt successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to update pipeline: {e}")
            raise

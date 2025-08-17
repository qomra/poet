# poet/core/pipeline.py

from typing import Dict, Any, List, Type, Optional
import logging
from .node import Node


class PipelineEngine:
    """
    Dynamic pipeline engine that creates and runs pipelines from configuration.
    
    Accepts a list of Node classes and their configurations, dynamically
    instantiates them, and executes the pipeline step by step.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nodes = []
        self.context = {}
        
    def add_node(self, node_class: Type[Node], node_config: Optional[Dict[str, Any]] = None):
        """
        Add a node to the pipeline.
        
        Args:
            node_class: The Node class to instantiate
            node_config: Configuration for the node
        """
        try:
            # Extract required parameters from context
            llm = self.context.get('llm')
            prompt_manager = self.context.get('prompt_manager')
            
            # Check if harmony capture is enabled
            harmony_enabled = self.context.get('harmony_capture_enabled', False)
            self.logger.info(f"Adding node {node_class.__name__}, harmony_capture_enabled: {harmony_enabled}")
            
            # Handle different node types with their specific requirements
            if node_class.__name__ == 'RefinerChain' :
                # RefinerChain only needs llm and config
                if node_config:
                    node = node_class(llm=llm, **node_config)
                else:
                    node = node_class(llm=llm)
            else:
                # Other nodes need llm and prompt_manager
                if node_config:
                    node = node_class(llm=llm, prompt_manager=prompt_manager, **node_config)
                else:
                    node = node_class(llm=llm, prompt_manager=prompt_manager)
            
            # If harmony capture is enabled, wrap the node with capture middleware
            if harmony_enabled and not node_class.__name__.startswith('Captured'):
                from poet.logging.capture_middleware import capture_component
                # Wrap the node immediately after creation, before adding to pipeline
                node = capture_component(node, node_class.__name__)
            
            self.nodes.append(node)
            self.logger.info(f"Added node: {node.name}")
        except Exception as e:
            self.logger.error(f"Failed to add node {node_class.__name__}: {e}")
            raise
    
    def set_context(self, context: Dict[str, Any]):
        """
        Set the pipeline context (LLM, prompt_manager, etc.).
        
        Args:
            context: Context dictionary
        """
        self.context = context
    
    def run_pipeline(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete pipeline.
        
        Args:
            initial_input: Initial input data for the first node
            
        Returns:
            Final output from the pipeline
        """
        self.logger.info(f"Starting pipeline with {len(self.nodes)} nodes")
        
        current_data = initial_input.copy()
        
        try:
            for i, node in enumerate(self.nodes):
                self.logger.info(f"Running node {i+1}/{len(self.nodes)}: {node.name}")
                
                # Validate input
                if not node.validate_input(current_data):
                    raise ValueError(f"Node {node.name} input validation failed")
                
                # Run node
                output_data = node.run(current_data, self.context)
                
                # Validate output
                if not node.validate_output(output_data):
                    raise ValueError(f"Node {node.name} output validation failed")
                
                # Merge output with current data for next node
                current_data.update(output_data)
                
                self.logger.info(f"Node {node.name} completed successfully")
            
            self.logger.info("Pipeline completed successfully")
            return current_data
            
        except Exception as e:
            self.logger.error(f"Pipeline failed at node {node.name if 'node' in locals() else 'unknown'}: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline structure.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "node_count": len(self.nodes),
            "nodes": [
                {
                    "name": node.name,
                    "class": node.__class__.__name__,
                    "config": node.config,
                    "required_inputs": node.get_required_inputs(),
                    "output_keys": node.get_output_keys()
                }
                for node in self.nodes
            ]
        }


class PipelineBuilder:
    """
    Builder class for creating pipelines from configuration.
    """
    
    def __init__(self):
        self.node_registry = {}
    
    def register_node(self, name: str, node_class: Type[Node]):
        """
        Register a node class with a name.
        
        Args:
            name: Name to register the node under
            node_class: The Node class
        """
        self.node_registry[name] = node_class
    
    def build_pipeline(self, pipeline_config: List[Any], context: Dict[str, Any]) -> PipelineEngine:
        """
        Build a pipeline from configuration.
        
        Args:
            pipeline_config: List of node configurations
            context: Pipeline context
            
        Returns:
            Configured PipelineEngine
        """
        pipeline = PipelineEngine({})
        pipeline.set_context(context)
        
        for step in pipeline_config:
            if isinstance(step, str):
                # Simple node name
                if step not in self.node_registry:
                    raise ValueError(f"Unknown node type: {step}")
                pipeline.add_node(self.node_registry[step])
                
            elif isinstance(step, dict):
                # Node with configuration
                for node_name, node_config in step.items():
                    if node_name not in self.node_registry:
                        raise ValueError(f"Unknown node type: {node_name}")
                    
                    if isinstance(node_config, dict):
                        pipeline.add_node(self.node_registry[node_name], node_config)
                    else:
                        pipeline.add_node(self.node_registry[node_name])
            else:
                raise ValueError(f"Invalid pipeline step: {step}")
        
        return pipeline

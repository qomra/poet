# poet/core/pipeline.py

from typing import Dict, Any, List, Type, Optional
import logging
from .node import Node


class PipelineEngine:
    """
    Dynamic pipeline engine that creates and runs pipelines from configuration.
    
    Accepts a list of Node classes and their configurations, dynamically
    instantiates them, and executes the pipeline step by step.
    Supports building flat compute graphs from nested configurations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.nodes = []
        self.compute_graph = []  # Flat compute graph
        self.context = {}
        
    def build_compute_graph(self, pipeline_config: List[Any], context: Dict[str, Any]):
        """
        Build a flat compute graph from nested configurations.
        
        Args:
            pipeline_config: List of node configurations
            context: Pipeline context
        """
        self.context = context
        self.compute_graph = []
        
        for node_config in pipeline_config:
            if isinstance(node_config, dict):
                # Handle nested structures like RefinerChain
                self._flatten_node_config(node_config, context)
            else:
                # Simple node
                node = self._create_node(node_config, context)
                if node:
                    self.compute_graph.append(node)
        
        self.logger.info(f"🏗️ Built compute graph with {len(self.compute_graph)} nodes")
        
    def _flatten_node_config(self, node_config: Dict[str, Any], context: Dict[str, Any]):
        """Flatten nested node configurations into individual nodes."""
        if 'refiner_chain' in node_config:
            # Create RefinerChain configuration parser
            from poet.refinement.refiner_chain import RefinerChain
            
            refiner_chain = RefinerChain(
                llm=context['llm'],
                refiners=node_config['refiner_chain'].get('refiners', ['prosody_refiner', 'qafiya_refiner']),
                max_iterations=node_config['refiner_chain'].get('max_iterations', 3),
                target_quality=node_config['refiner_chain'].get('target_quality', 0.8)
            )
            
            # Build the refinement sequence
            refinement_nodes = refiner_chain.build_refinement_sequence(context)
            
            # Add all nodes to the compute graph
            self.compute_graph.extend(refinement_nodes)
            
        else:
            # Handle other nested structures
            self.logger.info(f"Processing node config: {node_config}")
            node = self._create_node(node_config, context)
            if node:
                self.compute_graph.append(node)
                self.logger.info(f"Successfully added node to compute graph: {node.__class__.__name__}")
            else:
                self.logger.error(f"Failed to create node from config: {node_config}")
    
    def _create_node(self, node_config: Any, context: Dict[str, Any]) -> Optional[Node]:
        """Create a node from configuration."""
        try:
            # Extract required parameters from context
            llm = context.get('llm')
            prompt_manager = context.get('prompt_manager')
            
            # Check if node has its own LLM configuration
            if isinstance(node_config, dict) and 'llm' in node_config:
                llm = self._create_llm_from_config(node_config['llm'])
            
            # Handle BestOfN nodes specially
            if isinstance(node_config, dict) and any(key.startswith('best_of_n_') for key in node_config.keys()):
                return self._create_best_of_n_node(node_config, llm, prompt_manager)
            
            # Handle different node types with their specific requirements
            if isinstance(node_config, dict):
                node_class = self._get_node_class(node_config)
                
                # Handle nested configurations like {'pre_generated_generation': {'dataset_path': '...'}}
                if len(node_config) == 1 and list(node_config.keys())[0] in ['generation', 'evaluation', 'refiner_chain', 'constraints_parser', 
                                                                           'qafiya_selector', 'bahr_selector', 'knowledge_retriever', 'data_enricher', 
                                                                           'pre_generated_generation']:
                    # Extract the nested config
                    filtered_config = list(node_config.values())[0] if isinstance(list(node_config.values())[0], dict) else {}
                else:
                    # Filter out the node name key and llm from the config
                    filtered_config = {k: v for k, v in node_config.items() 
                                     if k not in ['generation', 'evaluation', 'refiner_chain', 'constraints_parser', 
                                                'qafiya_selector', 'bahr_selector', 'knowledge_retriever', 'data_enricher', 
                                                'pre_generated_generation', 'llm']}
                
                self.logger.info(f"Creating {node_class.__name__} with config: {filtered_config}")
                
                if node_class.__name__ == 'RefinerChain':
                    # RefinerChain only needs llm and config
                    return node_class(llm=llm, **filtered_config)
                elif node_class.__name__ == 'DataEnricher':
                    # DataEnricher needs llm and config but not prompt_manager
                    return node_class(llm=llm, **filtered_config)
                else:
                    # Other nodes need llm and prompt_manager
                    return node_class(llm=llm, prompt_manager=prompt_manager, **filtered_config)
            else:
                # Simple string node name
                node_class = self._get_node_class(node_config)
                return node_class(llm=llm, prompt_manager=prompt_manager)
                
        except Exception as e:
            self.logger.error(f"🚨 Failed to create node: {e}")
            return None
    
    def _create_llm_from_config(self, llm_config: Dict[str, Any]):
        """Create LLM instance from node-specific configuration."""
        from poet.llm.base_llm import LLMConfig
        from poet.llm.openai_adapter import OpenAIAdapter
        from poet.llm.anthropic_adapter import AnthropicAdapter
        from poet.llm.groq_adapter import GroqAdapter
        from poet.llm.base_llm import MockLLM
        import os
        
        provider = llm_config.get("provider", "groq")
        
        # Get API key from config or environment
        api_key = llm_config.get("api_key")
        if not api_key:
            if provider == "openai":
                api_key = os.getenv("OPENAI_API_KEY")
            elif provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY")
            elif provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
            elif provider == "vllm":
                api_key = ""  # vLLM doesn't require API key
            elif provider == "mock":
                api_key = None
        
        if not api_key and provider not in ["mock", "vllm"]:
            self.logger.warning(f"No API key found for {provider}, falling back to global LLM")
            return self.context.get('llm')
        
        # Create LLM config
        llm_config_obj = LLMConfig(
            model_name=llm_config.get("model"),
            api_key=api_key,
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens"),
            timeout=llm_config.get("timeout", 320),
            base_url=llm_config.get("api_base")
        )
        
        # Create appropriate LLM adapter
        if provider == "groq":
            return GroqAdapter(llm_config_obj)
        elif provider == "openai":
            return OpenAIAdapter(llm_config_obj)
        elif provider == "anthropic":
            return AnthropicAdapter(llm_config_obj)
        elif provider == "vllm":
            from poet.llm.vllm_adapter import VLLMAdapter
            return VLLMAdapter(llm_config_obj)
        elif provider == "mock":
            return MockLLM(llm_config_obj)
        else:
            self.logger.warning(f"Unknown LLM provider: {provider}, falling back to global LLM")
            return self.context.get('llm')
    
    def _get_node_class(self, node_config: Any) -> Type[Node]:
        """Get node class from configuration."""
        # This would need to be implemented based on your node registry
        # For now, using a simple mapping
        node_mapping = {
            'constraints_parser': 'ConstraintParser',
            'qafiya_selector': 'QafiyaSelector',
            'bahr_selector': 'BahrSelector',
            'data_enrichment': 'DataEnricher',
            'generation': 'SimplePoemGenerator',
            'pre_generated_generation': 'PreGeneratedPoemGenerator',
            'evaluation': 'PoemEvaluator',
            'refiner_chain': 'RefinerChain'
        }
        
        if isinstance(node_config, str):
            node_name = node_config
        elif isinstance(node_config, dict):
            # Find the key that's not a configuration parameter
            for key in node_config.keys():
                if key in node_mapping:
                    node_name = key
                    break
            else:
                raise ValueError(f"Unknown node configuration: {node_config}")
        else:
            raise ValueError(f"Invalid node configuration: {node_config}")
        
        # Import and return the node class
        if node_name == 'constraints_parser':
            from poet.analysis.constraint_parser import ConstraintParser
            return ConstraintParser
        elif node_name == 'qafiya_selector':
            from poet.analysis.qafiya_selector import QafiyaSelector
            return QafiyaSelector
        elif node_name == 'bahr_selector':
            from poet.analysis.bahr_selector import BahrSelector
            return BahrSelector
        elif node_name == 'data_enrichment':
            from poet.data.enricher import DataEnricher
            return DataEnricher
        elif node_name == 'generation':
            from poet.generation.poem_generator import SimplePoemGenerator
            return SimplePoemGenerator
        elif node_name == 'pre_generated_generation':
            from poet.generation.poem_generator import PreGeneratedPoemGenerator
            return PreGeneratedPoemGenerator
        elif node_name == 'evaluation':
            from poet.evaluation.poem import PoemEvaluator
            return PoemEvaluator
        elif node_name == 'refiner_chain':
            from poet.refinement.refiner_chain import RefinerChain
            return RefinerChain
        else:
            raise ValueError(f"Unknown node type: {node_name}")
    
    def _create_best_of_n_node(self, node_config: Dict[str, Any], llm, prompt_manager):
        """Create a BestOfN node with the appropriate underlying node."""
        from poet.search.factory import create_best_of_n_node
        
        # Determine the node type from the configuration key
        node_type = None
        for key in node_config.keys():
            if key.startswith('best_of_n_'):
                if 'evaluation' in key:
                    node_type = 'evaluation'
                elif 'refiner' in key:
                    node_type = 'refiner_chain'
                elif 'generation' in key:
                    node_type = 'generation'
                break
        
        # Fallback: determine from content if key doesn't help
        if node_type is None:
            if 'metrics' in node_config:
                node_type = 'evaluation'
            elif 'refiners' in node_config:
                node_type = 'refiner_chain'
            else:
                node_type = 'generation'
        
        # Extract search configuration from node_config
        search_config = {
            'n_candidates': node_config.get('n_candidates', 5),
            'selection_prompt': node_config.get('selection_prompt', f'{node_type}_selection'),
            'selection_metric': node_config.get('selection_metric', 'overall_score'),
            'temperature_range': node_config.get('temperature_range', [0.5, 0.7, 0.9, 1.1, 1.3])
        }
        
        # Create the BestOfN node
        return create_best_of_n_node(node_type, node_config, search_config, llm, prompt_manager)
    
    def run_pipeline(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete pipeline using the flat compute graph.
        
        Args:
            initial_input: Initial input data for the first node
            
        Returns:
            Final output from the pipeline
        """
        self.logger.info(f"🚀 Starting pipeline with {len(self.compute_graph)} nodes")
        
        current_data = initial_input.copy()
        
        try:
            for i, node in enumerate(self.compute_graph):
                self.logger.info(f"🎬 Running node {i+1}/{len(self.compute_graph)}: {node.name}")
                
                # Validate input
                if not node.validate_input(current_data):
                    raise ValueError(f"Node {node.name} input validation failed")
                
                # Run node
                output_data = node.run(current_data, self.context)
                
                # Store harmony data
                node._store_harmony_data(current_data, output_data)
                
                # Validate output
                if not node.validate_output(output_data):
                    raise ValueError(f"Node {node.name} output validation failed")
                
                # Merge output with current data for next node
                current_data.update(output_data)
                
                # Debug: Log what's in the current data after merge
                if 'poem' in current_data:
                    self.logger.info(f"🔍 After merge - Poem verses: {current_data['poem'].verses}")
                if 'evaluation' in current_data:
                    self.logger.info(f"🔍 After merge - Evaluation quality: {current_data['evaluation'].quality_score if hasattr(current_data['evaluation'], 'quality_score') else 'N/A'}")
                
                # Check if we should stop refinement
                if self._should_stop_refinement(node, output_data):
                    self.logger.info(f"🎯 Stopping refinement at node {node.name}")
                    break
                
                self.logger.info(f"🎭 Node {node.name} completed successfully")
            
            self.logger.info("🎉 Pipeline completed successfully")
            
            
            return current_data
            
        except Exception as e:
            self.logger.error(f"💣 Pipeline failed at node {node.name if 'node' in locals() else 'unknown'}: {e}")
            raise
    
    def _should_stop_refinement(self, node: Node, output_data: Dict[str, Any]) -> bool:
        """Check if refinement should stop (e.g., quality target met)."""
        # Check if this is an evaluation node and quality target is met
        if hasattr(node, 'iteration') and hasattr(node, 'target_quality'):
            quality_score = output_data.get('quality_score', 0)
            target_quality = node.target_quality
            if target_quality is not None:
                return quality_score >= target_quality
        return False
    
    def generate_harmony(self, llm) -> str:
        """
        Generate harmony by visiting the compute graph.
        
        Args:
            llm: LLM instance for generating harmony
            
        Returns:
            Generated harmony text
        """
        self.logger.info("🎹 Generating harmony from compute graph")
        
        harmony_data = []
        
        for node in self.compute_graph:
            node_harmony = node.get_harmony()
            harmony_data.append(node_harmony)
        
        # Generate final harmony using LLM
        return self._generate_final_harmony(harmony_data, llm)
    
    def _generate_final_harmony(self, harmony_data: List[Dict[str, Any]], llm) -> str:
        """Generate final harmony text from node harmony data using the harmony prompt template."""
        from poet.prompts import get_global_prompt_manager
        
        prompt_manager = get_global_prompt_manager()
        
        # Format execution steps from harmony data
        execution_steps = []
        for i, node_data in enumerate(harmony_data):
            step_info = f"Step {i+1}: {node_data['node_name']}\n"
            step_info += f"Input: {node_data['input_summary']}\n"
            step_info += f"Output: {node_data['output_summary']}\n"
            step_info += f"Reasoning: {node_data['reasoning']}\n"
            execution_steps.append(step_info)
        
        execution_steps_text = "\n\n".join(execution_steps)
        
        # Get final poem and quality assessment from the last node
        final_poem = "No poem generated"
        quality_assessment = "No quality assessment"
        
        for node_data in reversed(harmony_data):
            if 'poem' in node_data.get('output_summary', ''):
                final_poem = node_data['output_summary']
                break
        
        for node_data in reversed(harmony_data):
            if 'quality' in node_data.get('output_summary', '').lower():
                quality_assessment = node_data['output_summary']
                break
        
        # Get user prompt from first node
        user_prompt = "No user prompt available"
        initial_constraints = "No constraints available"
        
        if harmony_data:
            first_node = harmony_data[0]
            user_prompt = first_node.get('input_summary', 'No user prompt available')
            initial_constraints = first_node.get('output_summary', 'No constraints available')
        
        # Format the harmony prompt
        harmony_prompt = prompt_manager.format_prompt(
            'harmony_structured',
            user_prompt=user_prompt,
            initial_constraints=initial_constraints,
            execution_steps=execution_steps_text,
            final_poem=final_poem,
            quality_assessment=quality_assessment,
            conversation_start_date="2025-08-26"
        )
        
        # Generate harmony using LLM
        harmony_response = llm.generate(harmony_prompt)
        
        # Parse the harmony response like the old implementation
        return self._parse_harmony_response(harmony_response)
    
    def _parse_harmony_response(self, response: str) -> str:
        """
        Parse the LLM response and extract structured data.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed harmony text
        """
        try:
            import json
            import re
            
            # First try to extract JSON from code blocks
            json_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
            json_blocks = re.findall(json_block_pattern, response, re.DOTALL)
            
            if json_blocks:
                json_str = json_blocks[0]
            else:
                # Fallback: find JSON between { and }
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start == -1 or json_end == 0 or json_end <= json_start:
                    # No JSON found, return raw response
                    return response
                json_str = response[json_start:json_end]
            
            # Parse the JSON
            data = json.loads(json_str)
            
            # Convert to readable text format
            harmony_text = []
            
            if 'analysis' in data:
                for i, step in enumerate(data['analysis'], 1):
                    harmony_text.append(f"Step {i}: {step.get('step', 'Unknown')}")
                    harmony_text.append(f"Explanation: {step.get('explanation', '')}")
                    harmony_text.append("")
            
            if 'final_poem' in data:
                harmony_text.append(f"Final Poem:")
                harmony_text.append(data['final_poem'])
                harmony_text.append("")
            
            if 'conclusion' in data:
                harmony_text.append(f"Conclusion: {data['conclusion']}")
            
            if harmony_text:
                return "\n".join(harmony_text)
            else:
                # Fallback to raw response if parsing fails
                return response
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON in harmony response: {e}")
            # Fallback to raw response if JSON parsing fails
            return response
        except Exception as e:
            self.logger.error(f"Failed to parse harmony response: {e}")
            # Fallback to raw response if any parsing fails
            return response
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline structure.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "node_count": len(self.compute_graph),
            "nodes": [
                {
                    "name": node.name,
                    "class": node.__class__.__name__,
                    "config": node.config,
                    "required_inputs": node.get_required_inputs(),
                    "output_keys": node.get_output_keys()
                }
                for node in self.compute_graph
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
        pipeline.build_compute_graph(pipeline_config, context)
        return pipeline

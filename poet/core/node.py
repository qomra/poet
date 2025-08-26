# poet/core/node.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class Node(ABC):
    """
    Base class for all pipeline nodes.
    
    Each node represents a step in the poetry generation pipeline and must
    implement a run method that processes input data and returns output data.
    Each node is responsible for storing its own harmony data.
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = self.__class__.__name__
        
        # Harmony data storage
        self.harmony_data = {
            'input': None,
            'output': None,
            'reasoning': None,
            'metadata': {}
        }
    
    @abstractmethod
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the node's main logic.
        
        Args:
            input_data: Data passed from previous nodes or initial input
            context: Pipeline context including LLM, prompt_manager, etc.
            
        Returns:
            Output data to be passed to next nodes
        """
        pass
    
    def _store_harmony_data(self, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        """Store input and output data for harmony generation."""
        self.harmony_data['input'] = input_data
        self.harmony_data['output'] = output_data
        self.harmony_data['reasoning'] = self._generate_reasoning(input_data, output_data)
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """
        Generate natural reasoning for this specific node.
        
        Args:
            input_data: Input data to the node
            output_data: Output data from the node
            
        Returns:
            Natural language reasoning about what the node did
        """
        return f"Node {self.name} processed the input and generated output."
    
    def get_harmony(self) -> Dict[str, Any]:
        """
        Return node-specific harmony data for LLM consumption.
        
        Returns:
            Dictionary containing node harmony information
        """
        return {
            'node_type': self.__class__.__name__,
            'node_name': self.name,
            'input_summary': self._summarize_input(),
            'output_summary': self._summarize_output(),
            'reasoning': self.harmony_data['reasoning'],
            'metadata': self.harmony_data['metadata']
        }
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        # Basic summarization - can be overridden by subclasses
        input_keys = list(self.harmony_data['input'].keys())
        return f"Input contains: {', '.join(input_keys)}"
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        # Basic summarization - can be overridden by subclasses
        output_keys = list(self.harmony_data['output'].keys())
        return f"Output contains: {', '.join(output_keys)}"
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data before processing.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        return True
    
    def validate_output(self, output_data: Dict[str, Any]) -> bool:
        """
        Validate output data after processing.
        
        Args:
            output_data: Output data to validate
            
        Returns:
            True if output is valid, False otherwise
        """
        return True
    
    def get_required_inputs(self) -> list:
        """
        Get list of required input keys for this node.
        
        Returns:
            List of required input key names
        """
        return []
    
    def get_output_keys(self) -> list:
        """
        Get list of output keys this node produces.
        
        Returns:
            List of output key names
        """
        return []
    
    def __str__(self):
        return f"{self.name}({self.config})"
    
    def __repr__(self):
        return self.__str__()

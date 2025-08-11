# poet/core/node.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class Node(ABC):
    """
    Base class for all pipeline nodes.
    
    Each node represents a step in the poetry generation pipeline and must
    implement a run method that processes input data and returns output data.
    """
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        self.name = self.__class__.__name__
    
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

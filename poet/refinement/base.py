# poet/refinement/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.core.node import Node


@dataclass
class RefinementStep:
    """Represents a single refinement step in the chain"""
    refiner_name: str
    iteration: int
    before: LLMPoem
    after: LLMPoem
    quality_before: Optional[float] = None
    quality_after: Optional[float] = None
    details: Optional[str] = None


class BaseRefiner(Node):
    """Simple base class for all refiners"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """Refine poem based on evaluation feedback"""
        pass
    
    @abstractmethod
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        """Decide if refinement is needed based on evaluation"""
        pass
    
    @property
    def name(self) -> str:
        """Refiner name for logging and configuration"""
        return getattr(self, '_name', self.__class__.__name__.lower().replace('refiner', '_refiner'))
    
    @name.setter
    def name(self, value: str):
        """Set the refiner name"""
        self._name = value
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the refiner node.
        
        Args:
            input_data: Input data containing poem and constraints
            context: Pipeline context
            
        Returns:
            Output data with refined poem
        """
        # Extract required data
        poem = input_data.get('poem')
        constraints = input_data.get('constraints')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # For now, just return the poem as-is (no actual refinement)
        # In a real implementation, this would apply various refinements
        self.logger.info(f"Refiner node executed (no actual refinement applied)")
        
        return {
            'poem': poem,
            'refined': True,
            'refinement_iterations': 0
        }
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations'] 
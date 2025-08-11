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
        evaluation = input_data.get('evaluation')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Check if refinement is needed
        if evaluation and not self.should_refine(evaluation):
            self.logger.info(f"{self.name}: No refinement needed")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
        
        # Apply refinement
        try:
            # Since we can't use async in sync context, we'll need to handle this differently
            # For now, we'll create a simple refinement that can be done synchronously
            refined_poem = self._apply_sync_refinement(poem, constraints, evaluation)
            
            self.logger.info(f"{self.name}: Refinement applied successfully")
            return {
                'poem': refined_poem,
                'refined': True,
                'refinement_iterations': 1
            }
        except Exception as e:
            self.logger.error(f"{self.name}: Refinement failed: {e}")
            return {
                'poem': poem,
                'refined': False,
                'refinement_iterations': 0
            }
    
    def _apply_sync_refinement(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """
        Apply refinement synchronously. This is a fallback since the main refine method is async.
        Subclasses should override this method to provide actual refinement logic.
        """
        # Default implementation - return poem unchanged
        # Subclasses should override this method
        return poem
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints', 'evaluation']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations'] 
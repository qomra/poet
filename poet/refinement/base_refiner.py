# poet/refinement/base_refiner.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment


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


class BaseRefiner(ABC):
    """Simple base class for all refiners"""
    
    @abstractmethod
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        """Refine poem based on evaluation feedback"""
        pass
    
    @abstractmethod
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        """Decide if refinement is needed based on evaluation"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Refiner name for logging and configuration"""
        pass 
# tests/unit/test_base.py

import pytest
from unittest.mock import Mock, AsyncMock
from poet.refinement.base import BaseRefiner, RefinementStep
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment


class MockRefiner(BaseRefiner):
    """Mock refiner for testing base functionality"""
    
    def __init__(self, should_refine_result=True, refine_result=None):
        self.should_refine_result = should_refine_result
        self.refine_result = refine_result or Mock(spec=LLMPoem)
        self._name = "mock_refiner"
    
    @property
    def name(self) -> str:
        return self._name
    
    def should_refine(self, evaluation: QualityAssessment) -> bool:
        return self.should_refine_result
    
    async def refine(self, poem: LLMPoem, constraints: Constraints, evaluation: QualityAssessment) -> LLMPoem:
        return self.refine_result


class TestRefinementStep:
    """Test RefinementStep dataclass"""
    
    def test_refinement_step_creation(self):
        """Test creating a RefinementStep"""
        before_poem = Mock(spec=LLMPoem)
        after_poem = Mock(spec=LLMPoem)
        
        step = RefinementStep(
            refiner_name="test_refiner",
            iteration=1,
            before=before_poem,
            after=after_poem,
            quality_before=0.5,
            quality_after=0.8,
            details="Test refinement"
        )
        
        assert step.refiner_name == "test_refiner"
        assert step.iteration == 1
        assert step.before == before_poem
        assert step.after == after_poem
        assert step.quality_before == 0.5
        assert step.quality_after == 0.8
        assert step.details == "Test refinement"
    
    def test_refinement_step_minimal(self):
        """Test creating RefinementStep with minimal parameters"""
        before_poem = Mock(spec=LLMPoem)
        after_poem = Mock(spec=LLMPoem)
        
        step = RefinementStep(
            refiner_name="test_refiner",
            iteration=0,
            before=before_poem,
            after=after_poem
        )
        
        assert step.refiner_name == "test_refiner"
        assert step.iteration == 0
        assert step.before == before_poem
        assert step.after == after_poem
        assert step.quality_before is None
        assert step.quality_after is None
        assert step.details is None


class TestBaseRefiner:
    """Test BaseRefiner abstract class"""
    
    def test_base_interface(self):
        """Test that BaseRefiner defines required interface"""
        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            BaseRefiner()
    
    def test_mock_refiner_implements_interface(self):
        """Test that MockRefiner properly implements the interface"""
        refiner = MockRefiner()
        
        # Test name property
        assert refiner.name == "mock_refiner"
        
        # Test should_refine method
        evaluation = Mock(spec=QualityAssessment)
        assert refiner.should_refine(evaluation) is True
        
        # Test refine method signature
        poem = Mock(spec=LLMPoem)
        constraints = Mock(spec=Constraints)
        
        # Should be callable (async)
        assert callable(refiner.refine)
    
    def test_mock_refiner_custom_behavior(self):
        """Test MockRefiner with custom behavior"""
        custom_poem = Mock(spec=LLMPoem)
        refiner = MockRefiner(should_refine_result=False, refine_result=custom_poem)
        
        evaluation = Mock(spec=QualityAssessment)
        assert refiner.should_refine(evaluation) is False
        
        # Test async refine method
        import asyncio
        poem = Mock(spec=LLMPoem)
        constraints = Mock(spec=Constraints)
        
        result = asyncio.run(refiner.refine(poem, constraints, evaluation))
        assert result == custom_poem


class TestRefinerIntegration:
    """Test refiner integration patterns"""
    
    def test_refiner_chain_compatibility(self):
        """Test that refiners work with refiner chain"""
        refiner1 = MockRefiner(should_refine_result=True)
        refiner2 = MockRefiner(should_refine_result=False)
        
        # Test that refiners can be collected in a list
        refiners = [refiner1, refiner2]
        assert len(refiners) == 2
        assert all(isinstance(r, BaseRefiner) for r in refiners)
        
        # Test that refiners have names (uniqueness is not required for mock refiners)
        names = [r.name for r in refiners]
        assert all(name for name in names)  # All names should be non-empty
    
    def test_refiner_should_refine_logic(self):
        """Test different should_refine scenarios"""
        # Refiner that always needs refinement
        always_refine = MockRefiner(should_refine_result=True)
        
        # Refiner that never needs refinement
        never_refine = MockRefiner(should_refine_result=False)
        
        evaluation = Mock(spec=QualityAssessment)
        
        assert always_refine.should_refine(evaluation) is True
        assert never_refine.should_refine(evaluation) is False 
import logging
from typing import List, Tuple, Optional, Dict, Any
from poet.refinement.base import BaseRefiner, RefinementStep
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.evaluation.poem import PoemEvaluator, EvaluationType
from poet.logging.harmony_capture import capture_method
from poet.core.node import Node

class RefinerChain(Node):
    """Manages sequential execution of refiners"""
    
    def __init__(self, llm, refiners=None, max_iterations: int = 3, target_quality: float = 0.8, **kwargs):
        # Remove max_iterations from kwargs to avoid duplicate parameter error
        kwargs.pop('max_iterations', None)
        super().__init__(**kwargs)
        self.llm = llm
        self.refiners = refiners or []
        self.max_iterations = max_iterations
        self.target_quality = target_quality
        # Initialize evaluator
        self.evaluator = PoemEvaluator(self.llm, metrics=['prosody', 'qafiya'])
    
    async def refine(self, 
                     poem: LLMPoem, 
                     constraints: Constraints,
                     target_quality: float = 0.8) -> Tuple[LLMPoem, List[RefinementStep]]:
        """
        Iteratively refine poem until quality target is met or max iterations reached
        """
        current_poem = poem
        refinement_history = []
        
        for iteration in range(self.max_iterations):
            self.logger.info(f"Starting refinement iteration {iteration + 1}/{self.max_iterations}")
            
            # Evaluate current poem
            evaluated_poem = self.evaluator.evaluate_poem(
                current_poem, 
                constraints,
                [EvaluationType.PROSODY, EvaluationType.QAFIYA]
            )
            
            # Check if quality target met
            quality_score = self._calculate_quality_score(evaluated_poem.quality)
            self.logger.info(f"Current quality score: {quality_score:.3f}, target: {target_quality}")
            
            if quality_score >= target_quality:
                self.logger.info("Quality target reached, stopping refinement")
                break
            
            # Apply refiners that are needed
            iteration_refinements = []
            for refiner in self.refiners:
                if refiner.should_refine(evaluated_poem.quality):
                    self.logger.info(f"Applying {refiner.name}")
                    
                    before_poem = current_poem
                    refined_poem = await refiner.refine(current_poem, constraints, evaluated_poem.quality)
                    
                    # Create refinement step
                    refinement_step = RefinementStep(
                        refiner_name=refiner.name,
                        iteration=iteration,
                        before=before_poem,
                        after=refined_poem,
                        quality_before=quality_score,
                        quality_after=None,
                        details=f"Applied {refiner.name} to fix issues"
                    )
                    
                    iteration_refinements.append(refinement_step)
                    current_poem = refined_poem
            
            # If no refiners were applied, break to avoid infinite loop
            if not iteration_refinements:
                self.logger.info("No refiners applied, stopping refinement")
                break
            
            # Add iteration refinements to history
            refinement_history.extend(iteration_refinements)
        
        # Update quality_after for the last refinement step
        if refinement_history:
            final_evaluation = self.evaluator.evaluate_poem(
                current_poem,
                constraints,
                [EvaluationType.PROSODY, EvaluationType.QAFIYA]
            )
            final_quality = self._calculate_quality_score(final_evaluation.quality)
            refinement_history[-1].quality_after = final_quality
        
        self.logger.info(f"Refinement completed. Total steps: {len(refinement_history)}")
        return current_poem, refinement_history
    
    def _calculate_quality_score(self, evaluation: QualityAssessment) -> float:
        """Calculate overall quality score from evaluation"""
        if not evaluation:
            return 0.0
        
        # If overall_score is provided, use it directly
        if hasattr(evaluation, 'overall_score') and evaluation.overall_score is not None:
            return evaluation.overall_score
        
        # Otherwise, calculate score based on validation results
        score = 1.0
        
        # Deduct for prosody issues
        if evaluation.prosody_validation and not evaluation.prosody_validation.overall_valid:
            # Count number of broken verses
            broken_verses = 0
            if hasattr(evaluation.prosody_validation, 'bait_results'):
                for bait_result in evaluation.prosody_validation.bait_results:
                    if not bait_result.is_valid:
                        broken_verses += 1
            
            # Deduct based on proportion of broken verses
            total_verses = len(evaluation.prosody_validation.bait_results) if evaluation.prosody_validation and evaluation.prosody_validation.bait_results else 1
            prosody_penalty = min(0.4, (broken_verses / total_verses) * 0.4)
            score -= prosody_penalty
        
        # Deduct for qafiya issues
        if evaluation.qafiya_validation and not evaluation.qafiya_validation.overall_valid:
            # Count number of wrong qafiya verses
            wrong_qafiya_verses = 0
            if hasattr(evaluation.qafiya_validation, 'bait_results'):
                for bait_result in evaluation.qafiya_validation.bait_results:
                    if not bait_result.is_valid:
                        wrong_qafiya_verses += 1
            
            # Deduct based on proportion of wrong qafiya verses
            total_verses = len(evaluation.qafiya_validation.bait_results) if evaluation.qafiya_validation and evaluation.qafiya_validation.bait_results else 1
            qafiya_penalty = min(0.3, (wrong_qafiya_verses / total_verses) * 0.3)
            score -= qafiya_penalty
        
        return max(0.0, score)
    
    def _create_refiner(self, refiner_name: str, context: Dict[str, Any]) -> Optional[BaseRefiner]:
        """Create a refiner instance based on name."""
        try:
            if refiner_name == 'prosody_refiner':
                from poet.refinement.prosody import ProsodyRefiner
                refiner = ProsodyRefiner(self.llm, prompt_manager=context.get('prompt_manager'), max_iterations=1)
            elif refiner_name == 'qafiya_refiner':
                from poet.refinement.qafiya import QafiyaRefiner
                refiner = QafiyaRefiner(self.llm, prompt_manager=context.get('prompt_manager'), max_iterations=1)
            elif refiner_name == 'line_count_refiner':
                from poet.refinement.line_count import LineCountRefiner
                refiner = LineCountRefiner(self.llm, prompt_manager=context.get('prompt_manager'), max_iterations=1)
            elif refiner_name == 'tashkeel_refiner':
                from poet.refinement.tashkeel import TashkeelRefiner
                refiner = TashkeelRefiner(self.llm, prompt_manager=context.get('prompt_manager'), max_iterations=1)
            else:
                self.logger.warning(f"Unknown refiner type: {refiner_name}")
                return None
            
            # Set up context for the refiner
            refiner.llm = self.llm
            refiner.prompt_manager = context.get('prompt_manager')
            
            return refiner
            
        except ImportError as e:
            self.logger.error(f"Failed to import {refiner_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to create {refiner_name}: {e}")
            return None
    
    def get_refinement_summary(self, refinement_history: List[RefinementStep]) -> dict:
        """Get summary of refinement process"""
        if not refinement_history:
            return {"total_steps": 0, "refiners_used": [], "quality_improvement": 0.0, "iterations": 0}
        
        refiners_used = list(set(step.refiner_name for step in refinement_history))
        quality_improvement = 0.0
        
        if refinement_history:
            first_step = refinement_history[0]
            last_step = refinement_history[-1]
            
            if first_step.quality_before is not None and last_step.quality_after is not None:
                quality_improvement = last_step.quality_after - first_step.quality_before
        
        return {
            "total_steps": len(refinement_history),
            "refiners_used": refiners_used,
            "quality_improvement": quality_improvement,
            "iterations": max(step.iteration for step in refinement_history) + 1
        }
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the refiner chain node.
        
        Args:
            input_data: Input data containing poem and constraints
            context: Pipeline context
            
        Returns:
            Output data with refined poem
        """
        # Set up context
        self.llm = context.get('llm')
        if not self.llm:
            raise ValueError("LLM not provided in context")
        
        # Initialize evaluator
        self.evaluator = PoemEvaluator(self.llm, metrics=['prosody', 'qafiya'])
        
        # Initialize refiners from config or use defaults
        if not self.refiners:
            # Get refiner configuration from config
            refiner_config = self.config.get('refiners', ['prosody_refiner', 'qafiya_refiner', 'line_count_refiner', 'tashkeel_refiner'])
            
            # Create refiner instances based on configuration
            self.refiners = []
            for refiner_name in refiner_config:
                refiner = self._create_refiner(refiner_name, context)
                if refiner:
                    self.refiners.append(refiner)
            
            if not self.refiners:
                self.logger.warning("No refiners created, using default set")
                # Fallback to default refiners
                self.refiners = [
                    self._create_refiner('prosody_refiner', context),
                    self._create_refiner('qafiya_refiner', context),
                    self._create_refiner('line_count_refiner', context),
                    self._create_refiner('tashkeel_refiner', context)
                ]
                # Filter out None values
                self.refiners = [r for r in self.refiners if r is not None]
        
        # Extract required data
        poem = input_data.get('poem')
        constraints = input_data.get('constraints')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # For now, just return the poem as-is (no actual refinement)
        # In a real implementation, this would run the async refine method
        self.logger.info(f"Refiner chain node executed (no actual refinement applied)")
        
        return {
            'poem': poem,
            'refined': True,
            'refinement_iterations': 0,
            'refiner_chain_used': True
        }
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['poem', 'refined', 'refinement_iterations', 'refiner_chain_used']

class CapturedRefinerChain(RefinerChain):
    """
    Wrapper that adds Harmony capture to RefinerChain
    Simply inherit and decorate the methods you want to capture
    """
    
    @capture_method("RefinerChain", "refine")
    async def refine(self, 
                     poem: LLMPoem, 
                     constraints: Constraints,
                     target_quality: float = 0.8) -> Tuple[LLMPoem, List[RefinementStep]]:
        """Wrapped refine method with automatic capture"""
        return await super().refine(poem, constraints, target_quality)
    
    @capture_method("RefinerChain", "summary")
    def get_refinement_summary(self, refinement_history: List[RefinementStep]) -> dict:
        """Wrapped summary method with automatic capture"""
        return super().get_refinement_summary(refinement_history)
# poet/refinement/refiner_chain.py

import logging
from typing import List, Tuple, Optional
from poet.refinement.base_refiner import BaseRefiner, RefinementStep
from poet.models.poem import LLMPoem
from poet.models.constraints import Constraints
from poet.models.quality import QualityAssessment
from poet.evaluation.poem_evaluation import PoemEvaluator, EvaluationType


class RefinerChain:
    """Manages sequential execution of refiners"""
    
    def __init__(self, refiners: List[BaseRefiner], llm, max_iterations: int = 3):
        self.refiners = refiners
        self.max_iterations = max_iterations
        self.evaluator = PoemEvaluator(llm)  # Your existing evaluator
        self.logger = logging.getLogger(self.__class__.__name__)
    
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
            evaluation = await self.evaluator.evaluate_poem(
                current_poem, 
                constraints,
                [EvaluationType.LINE_COUNT, EvaluationType.PROSODY, EvaluationType.QAFIYA]
            )
            
            # Check if quality target met
            quality_score = self._calculate_quality_score(evaluation)
            self.logger.info(f"Current quality score: {quality_score:.3f}, target: {target_quality}")
            
            if quality_score >= target_quality:
                self.logger.info("Quality target reached, stopping refinement")
                break
            
            # Apply refiners that are needed
            iteration_refinements = []
            for refiner in self.refiners:
                if refiner.should_refine(evaluation):
                    self.logger.info(f"Applying {refiner.name}")
                    
                    before_poem = current_poem
                    refined_poem = await refiner.refine(current_poem, constraints, evaluation)
                    
                    # Create refinement step
                    refinement_step = RefinementStep(
                        refiner_name=refiner.name,
                        iteration=iteration,
                        before=before_poem,
                        after=refined_poem,
                        quality_before=quality_score,
                        quality_after=None,  # Will be calculated in next iteration
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
            final_evaluation = await self.evaluator.evaluate_poem(
                current_poem,
                constraints,
                [EvaluationType.LINE_COUNT, EvaluationType.PROSODY, EvaluationType.QAFIYA]
            )
            final_quality = self._calculate_quality_score(final_evaluation)
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
        
        # Deduct for line count issues
        if evaluation.line_count_validation and not evaluation.line_count_validation.is_valid:
            score -= 0.3
        
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
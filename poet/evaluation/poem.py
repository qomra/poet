import logging
from typing import List, Optional, Dict, Any
from enum import Enum

from poet.models.poem import LLMPoem
from poet.models.quality import QualityAssessment
from poet.models.constraints import Constraints
from poet.models.line_count import LineCountValidationResult
from poet.models.prosody import ProsodyValidationResult
from poet.models.qafiya import QafiyaValidationResult
from poet.models.tashkeel import TashkeelValidationResult
from poet.evaluation.line_count import LineCountEvaluator
from poet.evaluation.prosody import ProsodyEvaluator
from poet.evaluation.qafiya import QafiyaEvaluator
from poet.evaluation.tashkeel import TashkeelEvaluator
from poet.llm.base_llm import BaseLLM
from poet.core.node import Node
from poet.prompts import get_global_prompt_manager

logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    """Types of evaluations that can be performed"""
    LINE_COUNT = "line_count"
    PROSODY = "prosody"
    QAFIYA = "qafiya"
    TASHKEEL = "tashkeel"


class PoemEvaluator(Node):
    """
    Evaluates poem quality using multiple metrics.
    
    Supports iteration context for refinement pipelines.
    """
    
    def __init__(self, llm, metrics: List[str] = None, 
                 iteration: int = None, target_quality: float = None, **kwargs):
        super().__init__(**kwargs)
        
        self.llm = llm
        self.prompt_manager = get_global_prompt_manager()
        self.metrics = metrics or ['prosody', 'qafiya']
        self.iteration = iteration
        self.target_quality = target_quality
        
        # Initialize evaluators
        self.prosody_evaluator = ProsodyEvaluator(self.llm)
        self.qafiya_evaluator = QafiyaEvaluator(self.llm)
    
    def run(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate poem quality.
        
        Args:
            input_data: Input data containing poem and constraints
            context: Pipeline context
            
        Returns:
            Output data with evaluation results
        """
        # Validate inputs
        poem = input_data.get('poem')
        constraints = input_data.get('constraints')
        
        if not poem:
            raise ValueError("poem not found in input_data")
        if not constraints:
            raise ValueError("constraints not found in input_data")
        
        # Evaluate poem
        evaluation = self.evaluate_poem(poem, constraints, [EvaluationType.PROSODY, EvaluationType.QAFIYA])
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(evaluation)
        
        # Log quality score for monitoring
        iteration_text = f" (Iteration {self.iteration})" if self.iteration else ""
        self.logger.info(f"📊 Quality Score{iteration_text}: {quality_score:.3f}")
        
        if self.target_quality:
            if quality_score >= self.target_quality:
                self.logger.info(f"🎯 Quality target {self.target_quality:.3f} MET! Stopping refinement.")
            else:
                self.logger.info(f"🎯 Quality target {self.target_quality:.3f} NOT met. Continuing refinement.")
        
        # Store harmony data
        output_data = {
            'evaluation': evaluation,
            'evaluated': True,
            'quality_score': quality_score,
            'target_quality': self.target_quality,
            'iteration': self.iteration
        }
        
        self._store_harmony_data(input_data, output_data)
        
        return output_data
    
    def evaluate_poem(self, poem: LLMPoem, constraints: Constraints, 
                     evaluation_types: List[EvaluationType]) -> QualityAssessment:
        """
        Evaluate poem using specified evaluation types.
        
        Args:
            poem: Poem to evaluate
            constraints: Poem constraints
            evaluation_types: Types of evaluation to perform
            
        Returns:
            Quality assessment results
        """
        self.logger.info(f"📊 Evaluating poem with metrics: {[et.value for et in evaluation_types]}")
        
        prosody_validation = None
        qafiya_validation = None
        
        # Perform prosody evaluation
        if EvaluationType.PROSODY in evaluation_types:
            self.logger.info("🎵 Performing prosody validation")
            updated_poem = self.prosody_evaluator.validate_poem(poem, constraints.meter)
            prosody_validation = updated_poem.prosody_validation
        
        # Perform qafiya evaluation
        if EvaluationType.QAFIYA in evaluation_types:
            self.logger.info("🎯 Performing qafiya validation")
            qafiya_validation = self.qafiya_evaluator.evaluate_qafiya(
                poem, 
                expected_qafiya=constraints.qafiya,
                qafiya_harakah=constraints.qafiya_harakah,
                qafiya_type=constraints.qafiya_type.value if constraints.qafiya_type else None,
                qafiya_type_description_and_examples=constraints.qafiya_type_description_and_examples
            )
        
        # Create quality assessment
        assessment = QualityAssessment(
            prosody_issues=[],
            line_count_issues=[],
            qafiya_issues=[],
            overall_score=1.0,
            is_acceptable=True,
            recommendations=[],
            prosody_validation=prosody_validation,
            qafiya_validation=qafiya_validation
        )
        
        self.logger.info("✅ Poem evaluation completed")
        return assessment
    
    def _calculate_quality_score(self, evaluation: QualityAssessment) -> float:
        """Calculate overall quality score from evaluation."""
        if not evaluation:
            return 0.0
        
        # Start with base score
        score = 1.0
        
        # Debug logging to see what's happening
        self.logger.info(f"🔍 Quality calculation debug:")
        
        # Apply penalties based on validation results
        if evaluation.prosody_validation:
            prosody_valid = evaluation.prosody_validation.overall_valid
            self.logger.info(f"  📊 Prosody validation: {'PASSED' if prosody_valid else 'FAILED'}")
            if not prosody_valid:
                score -= 0.4
                self.logger.info(f"  ⚠️ Applied prosody penalty, score now: {score:.3f}")
        else:
            self.logger.info(f"  📊 Prosody validation: NOT AVAILABLE")
        
        if evaluation.qafiya_validation:
            qafiya_valid = evaluation.qafiya_validation.overall_valid
            self.logger.info(f"  📊 Qafiya validation: {'PASSED' if qafiya_valid else 'FAILED'}")
            if not qafiya_valid:
                score -= 0.3
                self.logger.info(f"  ⚠️ Applied qafiya penalty, score now: {score:.3f}")
        else:
            self.logger.info(f"  📊 Qafiya validation: NOT AVAILABLE")
        
        # If overall_score is provided, use it as a reference
        if hasattr(evaluation, 'overall_score') and evaluation.overall_score is not None:
            score = min(score, evaluation.overall_score)
            self.logger.info(f"  📊 Using overall_score: {evaluation.overall_score:.3f}")
        
        final_score = max(0.0, score)
        self.logger.info(f"  🎯 Final quality score: {final_score:.3f}")
        
        return final_score
    
    def _generate_reasoning(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> str:
        """Generate natural reasoning for this evaluation node."""
        iteration_text = f" (Iteration {self.iteration})" if self.iteration else ""
        quality_score = output_data.get('quality_score', 0)
        target_quality = self.target_quality or 0.8
        
        reasoning = f"I evaluated the poem's quality{iteration_text}."
        reasoning += f" The overall quality score is {quality_score:.2f}."
        
        if self.target_quality:
            if quality_score >= target_quality:
                reasoning += f" This meets the target quality threshold of {target_quality}."
            else:
                reasoning += f" This does not meet the target quality threshold of {target_quality}, so refinement should continue."
        
        # Add specific evaluation details
        evaluation = output_data.get('evaluation')
        if evaluation:
            if evaluation.prosody_validation:
                prosody_valid = evaluation.prosody_validation.overall_valid
                reasoning += f" Prosody validation: {'Passed' if prosody_valid else 'Failed'}."
            
            if evaluation.qafiya_validation:
                qafiya_valid = evaluation.qafiya_validation.overall_valid
                reasoning += f" Qafiya validation: {'Passed' if qafiya_valid else 'Failed'}."
        
        return reasoning
    
    def _summarize_input(self) -> str:
        """Summarize input data for harmony."""
        if not self.harmony_data['input']:
            return "No input data"
        
        poem = self.harmony_data['input'].get('poem')
        if poem:
            return f"Evaluated poem with {len(poem.verses)} verses"
        return "Evaluated poem"
    
    def _summarize_output(self) -> str:
        """Summarize output data for harmony."""
        if not self.harmony_data['output']:
            return "No output data"
        
        evaluation = self.harmony_data['output'].get('evaluation')
        poem = self.harmony_data['output'].get('poem')
        
        if evaluation and poem:
            eval_info = evaluation
            overall = getattr(eval_info, 'overall_score', 'unknown')
            prosody = getattr(eval_info, 'prosody_score', 'unknown')
            qafiya_score = getattr(eval_info, 'qafiya_score', 'unknown')
            
            # Show detailed evaluation feedback
            feedback = f"Scores: overall {overall}, prosody {prosody}, qafiya {qafiya_score}"
            
            # Extract detailed validation information from poem.quality
            if hasattr(poem, 'quality') and poem.quality:
                quality = poem.quality
                
                # Add specific issues found with actual error details
                if hasattr(quality, 'prosody_issues') and quality.prosody_issues:
                    issues = quality.prosody_issues
                    if isinstance(issues, list) and len(issues) > 0:
                        feedback += f"\nProsody issues ({len(issues)} found):"
                        for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                            feedback += f"\n  - Issue {i+1}: {issue}"
                
                if hasattr(quality, 'qafiya_issues') and quality.qafiya_issues:
                    issues = quality.qafiya_issues
                    if isinstance(issues, list) and len(issues) > 0:
                        feedback += f"\nQafiya issues ({len(issues)} found):"
                        for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                            feedback += f"\n  - Issue {i+1}: {issue}"
                
                if hasattr(quality, 'line_count_issues') and quality.line_count_issues:
                    issues = quality.line_count_issues
                    if isinstance(issues, list) and len(issues) > 0:
                        feedback += f"\nLine count issues ({len(issues)} found):"
                        for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                            feedback += f"\n  - Issue {i+1}: {issue}"
                
                if hasattr(quality, 'tashkeel_issues') and quality.tashkeel_issues:
                    issues = quality.tashkeel_issues
                    if isinstance(issues, list) and len(issues) > 0:
                        feedback += f"\nTashkeel issues ({len(issues)} found):"
                        for i, issue in enumerate(issues[:3]):  # Show first 3 issues
                            feedback += f"\n  - Issue {i+1}: {issue}"
            
            return feedback
        
        return "Poem evaluation completed"
    
    def get_required_inputs(self) -> list:
        """Get list of required input keys for this node."""
        return ['poem', 'constraints']
    
    def get_output_keys(self) -> list:
        """Get list of output keys this node produces."""
        return ['evaluation', 'evaluated', 'quality_score', 'target_quality', 'iteration']
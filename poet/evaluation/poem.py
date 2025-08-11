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

logger = logging.getLogger(__name__)


class EvaluationType(Enum):
    """Types of evaluations that can be performed"""
    LINE_COUNT = "line_count"
    PROSODY = "prosody"
    QAFIYA = "qafiya"
    TASHKEEL = "tashkeel"


class PoemEvaluator:
    """
    Orchestrates the poem evaluation workflow by running multiple validators
    and consolidating their results into a unified quality assessment.
    """
    
    def __init__(self, llm: BaseLLM):
        """
        Initialize PoemEvaluator with required components.
        
        Args:
            llm: LLM instance for validators that need it
        """
        self.llm = llm
        self.line_count_validator = LineCountEvaluator()
        self.prosody_validator = ProsodyEvaluator(llm)
        self.qafiya_validator = QafiyaEvaluator(llm)
        self.tashkeel_validator = TashkeelEvaluator(llm)
    
    def evaluate_poem(self, poem: LLMPoem, constraints: Constraints, 
                     evaluations: List[EvaluationType]) -> LLMPoem:
        """
        Evaluate poem using specified validators and update poem quality.
        
        Args:
            poem: The poem to evaluate
            constraints: User constraints for evaluation
            evaluations: List of evaluation types to perform
            
        Returns:
            Updated poem with quality assessment
        """
        logger.info(f"Starting poem evaluation with {len(evaluations)} evaluation types")
        
        # Initialize quality components
        line_count_issues = []
        prosody_issues = []
        qafiya_issues = []
        tashkeel_issues = []
        # Store detailed validation results
        line_count_validation = None
        prosody_validation = None
        qafiya_validation = None
        tashkeel_validation = None
        # Step 1: Line count validation (if requested)
        if EvaluationType.LINE_COUNT in evaluations:
            logger.info("Performing line count validation")
            line_count_result = self.line_count_validator.evaluate_line_count(poem)
            # Store detailed validation result
            line_count_validation = line_count_result
            if not line_count_result.is_valid:
                line_count_issues.append(line_count_result.validation_summary)
        
        # Step 2: Prosody validation (if requested)
        if EvaluationType.PROSODY in evaluations:
            logger.info("Performing prosody validation")
            try:
                prosody_result = self.prosody_validator.validate_poem(poem, constraints.meter)
                # Store detailed validation result
                prosody_validation = prosody_result.prosody_validation
                # Extract issues from prosody validation result
                if prosody_result.prosody_validation:
                    for bait_result in prosody_result.prosody_validation.bait_results:
                        if not bait_result.is_valid and bait_result.error_details:
                            prosody_issues.append(bait_result.error_details)
            except Exception as e:
                logger.error(f"Error in prosody validation: {e}")
                prosody_issues.append(f"خطأ في التحقق العروضي: {str(e)}")
        
        # Step 3: Qafiya validation (if requested)
        if EvaluationType.QAFIYA in evaluations:
            logger.info("Performing qafiya validation")
            try:
                qafiya_result = self.qafiya_validator.evaluate_qafiya(
                    poem, 
                    expected_qafiya=constraints.qafiya,
                    qafiya_harakah=constraints.qafiya_harakah,
                    qafiya_type=constraints.qafiya_type.value if constraints.qafiya_type is not None else None,
                    qafiya_type_description_and_examples=constraints.qafiya_type_description_and_examples
                )
                # Store detailed validation result
                qafiya_validation = qafiya_result
                # Extract issues from qafiya validation result
                if qafiya_result.issues:
                    qafiya_issues.extend(qafiya_result.issues)
            except Exception as e:
                logger.error(f"Error in qafiya validation: {e}")
                qafiya_issues.append(f"خطأ في التحقق من القافية: {str(e)}")
        
        # Step 4: Tashkeel validation (if requested)
        if EvaluationType.TASHKEEL in evaluations:
            logger.info("Performing tashkeel validation")
            try:
                tashkeel_result = self.tashkeel_validator.evaluate_tashkeel(poem)
                # Store detailed validation result
                tashkeel_validation = tashkeel_result
                # Extract issues from tashkeel validation result
                if tashkeel_result.issues:
                    tashkeel_issues.extend(tashkeel_result.issues)
                elif tashkeel_result.bait_results:
                    for bait_result in tashkeel_result.bait_results:
                        if not bait_result.is_valid and bait_result.error_details:
                            tashkeel_issues.append(bait_result.error_details)
            except Exception as e:
                logger.error(f"Error in tashkeel validation: {e}")
                tashkeel_issues.append(f"خطأ في التحقق من التشكيل: {str(e)}")
        
        # Step 4: Consolidate quality assessment
        self._update_poem_quality(
            poem, 
            line_count_issues=line_count_issues,
            prosody_issues=prosody_issues,
            qafiya_issues=qafiya_issues,
            tashkeel_issues=tashkeel_issues,
            line_count_validation=line_count_validation,
            prosody_validation=prosody_validation,
            qafiya_validation=qafiya_validation,
            tashkeel_validation=tashkeel_validation
        )
        
        logger.info("Poem evaluation completed")
        return poem
    
    def _update_poem_quality(self, poem: LLMPoem, 
                           line_count_issues: List[str] = None,
                           prosody_issues: List[str] = None,
                           qafiya_issues: List[str] = None,
                           tashkeel_issues: List[str] = None,
                           line_count_validation: Optional[LineCountValidationResult] = None,
                           prosody_validation: Optional[ProsodyValidationResult] = None,
                           qafiya_validation: Optional[QafiyaValidationResult] = None,
                           tashkeel_validation: Optional[TashkeelValidationResult] = None):
        """
        Update poem quality assessment based on validation results.
        
        Args:
            poem: The poem to update
            line_count_issues: Issues from line count validation
            prosody_issues: Issues from prosody validation
            qafiya_issues: Issues from qafiya validation
        """
        # Collect all issues
        all_line_count_issues = line_count_issues or []
        all_prosody_issues = prosody_issues or []
        all_qafiya_issues = qafiya_issues or []
        all_tashkeel_issues = tashkeel_issues or []
        
        # Calculate overall score
        overall_score = 1.0
        
        # Deduct points for issues
        if all_line_count_issues:
            overall_score -= 0.3
        if all_prosody_issues:
            overall_score -= min(0.4, len(all_prosody_issues) * 0.1)
        if all_qafiya_issues:
            overall_score -= min(0.3, len(all_qafiya_issues) * 0.1)
        if all_tashkeel_issues:
            overall_score -= min(0.2, len(all_tashkeel_issues) * 0.05)
        
        # Determine if acceptable
        is_acceptable = overall_score >= 0.7 and not all_line_count_issues
        
        # Generate recommendations
        recommendations = []
        if all_line_count_issues:
            recommendations.append("تأكد من أن عدد الأبيات زوجي")
        if all_prosody_issues:
            recommendations.append("راجع الأوزان العروضية للأبيات")
        if all_qafiya_issues:
            recommendations.append("راجع القافية في الأبيات")
        if all_tashkeel_issues:
            recommendations.append("راجع التشكيل في الأبيات")
        
        # Create quality assessment
        poem.quality = QualityAssessment(
            prosody_issues=all_prosody_issues,
            line_count_issues=all_line_count_issues,
            qafiya_issues=all_qafiya_issues,
            tashkeel_issues=all_tashkeel_issues,
            overall_score=overall_score,
            is_acceptable=is_acceptable,
            recommendations=recommendations,
            line_count_validation=line_count_validation,
            prosody_validation=prosody_validation,
            qafiya_validation=qafiya_validation,
            tashkeel_validation=tashkeel_validation
        ) 
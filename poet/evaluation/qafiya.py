# poet/evaluation/qafiya.py

import logging
from typing import Optional, Dict, Any, List
from poet.models.poem import LLMPoem
from poet.prompts import get_global_prompt_manager
from poet.llm.base_llm import BaseLLM
from poet.core.node import Node
from poet.models.qafiya import QafiyaBaitResult, QafiyaValidationResult
import json


class QafiyaValidationError(Exception):
    """Raised when qafiya validation fails"""
    pass


class QafiyaEvaluator:
    """
    Validates qafiya (rhyme) consistency in Arabic poetry using LLM analysis.
    
    Analyzes the rhyme patterns across all baits to ensure consistency
    and identifies any misaligned baits according to classical Arabic qafiya rules.
    """
    
    def __init__(self, llm_provider: BaseLLM, **kwargs):
        self.llm = llm_provider
        self.prompt_manager = get_global_prompt_manager()
        self.logger = logging.getLogger(__name__)
    
    def evaluate_qafiya(self, poem: LLMPoem, expected_qafiya: Optional[str] = None, 
                       qafiya_harakah: Optional[str] = None, qafiya_type: Optional[str] = None,
                       qafiya_type_description_and_examples: Optional[str] = None) -> QafiyaValidationResult:
        """
        Validate qafiya consistency across all baits in the poem.
        
        Assumes the poem has already been validated for line count and
        has proper diacritics applied. Now works with complete qafiya specifications.
        
        Args:
            poem: LLMPoem object containing the verses to validate
            expected_qafiya: Expected qafiya letter (e.g., "ق", "ع", "ل")
            qafiya_harakah: Expected qafiya harakah (e.g., "مكسور", "مضموم", "مفتوح", "ساكن")
            qafiya_type: Expected qafiya type (e.g., "متواتر", "متراكب", "متدارك", "متكاوس", "مترادف")
            qafiya_pattern: Expected qafiya pattern (e.g., "قُ", "عِ", "رَ", "لْ")
            
        Returns:
            QafiyaValidationResult with validation details
            
        Raises:
            QafiyaValidationError: If validation fails
        """
        try:
            # Get baits from poem (assumes line count already validated)
            baits = poem.get_baits()
            if not baits:
                return self._create_invalid_result(
                    "لا توجد أبيات للتحقق من القافية",
                    [],
                    expected_qafiya=expected_qafiya,
                    qafiya_harakah=qafiya_harakah,
                    qafiya_type=qafiya_type,
                    )
            
            # Format the qafiya validation prompt with complete specifications
            # Evaluate each bait individually against the qafiya specifications
            bait_results = []
            misaligned_lines = []
            issues = []
            
            for i, bait in enumerate(baits):
                bait_number = i + 1
                
                # Format single bait for evaluation
                bait_formatted = f"{bait_number}. {'#'.join(bait)}"
                
                formatted_prompt = self.prompt_manager.format_prompt(
                    'qafiya_validation',
                    verses=bait_formatted,
                    qafiya=expected_qafiya,
                    qafiya_type=qafiya_type,
                    qafiya_harakah=qafiya_harakah,
                    qafiya_type_description_and_examples=qafiya_type_description_and_examples
                )
                
                # Get LLM response for this single bait
                response = self.llm.generate(formatted_prompt)
                
                # Parse the structured response
                try:
                    validation_data = self._parse_llm_response(response)
                    
                    # Check if this bait is valid
                    is_valid = validation_data.get('is_valid', False)
                    error_details = validation_data.get('issue', None)
                except Exception as e:
                    # If parsing fails, treat as invalid with error details
                    self.logger.warning(f"📝 Failed to parse LLM response for bait {bait_number}: {e}")
                    is_valid = False
                    error_details = f"فشل في تحليل استجابة الذكاء الاصطناعي: {str(e)}"
                
                # Add to overall results
                if not is_valid:
                    misaligned_lines.append(bait_number)
                    if error_details:
                        issues.append(error_details)
                
                bait_result = QafiyaBaitResult(
                    bait_number=bait_number,
                    is_valid=is_valid,
                    error_details=error_details
                )
                bait_results.append(bait_result)
            
            valid_baits = len(baits) - len(misaligned_lines)
            
            # Create overall result
            overall_valid = len(misaligned_lines) == 0
            validation_summary = self._generate_validation_summary(
                valid_baits, len(misaligned_lines), expected_qafiya=expected_qafiya
            )
            
            return QafiyaValidationResult(
                overall_valid=overall_valid,
                total_baits=len(baits),
                valid_baits=valid_baits,
                invalid_baits=len(misaligned_lines),
                bait_results=bait_results,
                validation_summary=validation_summary,
                misaligned_bait_numbers=misaligned_lines,
                expected_qafiya=expected_qafiya,
                qafiya_harakah=qafiya_harakah,
                qafiya_type=qafiya_type,
            )
            
        except Exception as e:
            self.logger.error(f"🎯 Failed to validate qafiya: {e}")
            raise QafiyaValidationError(f"Qafiya validation failed: {e}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the structured JSON response from the LLM.
        
        Args:
            response: Raw LLM response containing JSON
            
        Returns:
            Parsed validation data dictionary
            
        Raises:
            QafiyaValidationError: If JSON parsing fails
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                # Try to extract from markdown code blocks
                if '```json' in response:
                    json_start = response.find('```json') + 7
                    json_end = response.find('```', json_start)
                    if json_end == -1:
                        json_end = len(response)
                else:
                    raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end].strip()
            
            # Parse JSON
            data = json.loads(json_str)
            
            # Validate required structure
            self._validate_response_structure(data)
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(f"📄 Failed to parse JSON response: {e}")
            self.logger.error(f"📝 Response text: {response[:200]}...")
            raise QafiyaValidationError(f"Invalid JSON response: {e}")
        except ValueError as e:
            self.logger.error(f"📋 Invalid response format: {e}")
            self.logger.error(f"📝 Response text: {response[:200]}...")
            raise QafiyaValidationError(f"Invalid response format: {e}")
    
    def _validate_response_structure(self, data: Dict[str, Any]):
        """
        Validate that the response has the expected structure.
        
        Args:
            data: Parsed JSON data
            
        Raises:
            ValueError: If structure is invalid
        """
        required_fields = ['is_valid']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}. Response keys: {list(data.keys())}")
        
        # Validate is_valid is a boolean
        if not isinstance(data['is_valid'], bool):
            raise ValueError(f"'is_valid' must be a boolean, got {type(data['is_valid']).__name__}: {data['is_valid']}")
        
        # Validate issue is a string or null if present
        if 'issue' in data and data['issue'] is not None and not isinstance(data['issue'], str):
            raise ValueError(f"'issue' must be a string or null, got {type(data['issue']).__name__}: {data['issue']}")
    
    def _create_invalid_result(self, error_message: str, baits: List[tuple], 
                              expected_qafiya: Optional[str] = None,
                              qafiya_harakah: Optional[str] = None,
                              qafiya_type: Optional[str] = None) -> QafiyaValidationResult:
        """Create an invalid result for error cases"""
        bait_results = []
        for i, bait in enumerate(baits):
            bait_results.append(QafiyaBaitResult(
                bait_number=i + 1,
                is_valid=False,
                error_details=error_message
            ))
        
        return QafiyaValidationResult(
            overall_valid=False,
            total_baits=len(baits),
            valid_baits=0,
            invalid_baits=len(baits),
            bait_results=bait_results,
            validation_summary=error_message,
            misaligned_bait_numbers=list(range(1, len(baits) + 1)),
            expected_qafiya=expected_qafiya,
            qafiya_harakah=qafiya_harakah,
            qafiya_type=qafiya_type
        )
    
    def _generate_validation_summary(self, valid_baits: int, invalid_baits: int, expected_qafiya: Optional[str] = None) -> str:
        """Generate validation summary"""
        total_baits = valid_baits + invalid_baits
        
        if invalid_baits == 0:
            expected_info = f" على القافية المطلوبة ({expected_qafiya})" if expected_qafiya else ""
            return f"جميع الأبيات ({total_baits}) صحيحة قافياً{expected_info}"
        elif valid_baits == 0:
            expected_info = f" على القافية المطلوبة ({expected_qafiya})" if expected_qafiya else ""
            return f"جميع الأبيات ({total_baits}) خاطئة قافياً{expected_info}"
        else:
            # If less than 5 invalid baits, mention specific bait numbers
            if invalid_baits < 5:
                invalid_bait_numbers = list(range(1, invalid_baits + 1))
                bait_numbers_str = "، ".join(map(str, invalid_bait_numbers))
                expected_info = f" على القافية المطلوبة ({expected_qafiya})" if expected_qafiya else ""
                return f"{valid_baits} من {total_baits} أبيات صحيحة قافياً. الأبيات الخاطئة: {bait_numbers_str}{expected_info}"
            else:
                expected_info = f" على القافية المطلوبة ({expected_qafiya})" if expected_qafiya else ""
                return f"{valid_baits} من {total_baits} أبيات صحيحة قافياً. عدد الأبيات الخاطئة: {invalid_baits}{expected_info}" 
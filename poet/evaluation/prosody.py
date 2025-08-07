import logging
import json
from typing import Dict, List, Optional, Any
from poet.models.poem import LLMPoem
from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
from poet.prompts.prompt_manager import PromptManager
from poet.llm.base_llm import BaseLLM
from poet.utils.bohour.arudi_style import get_arudi_style

# Import bohour classes
from poet.utils.bohour.bahr import (
    Taweel, Madeed, Baseet, Wafer, Kamel, Hazaj, Rajaz, Ramal,
    Saree, Munsareh, Khafeef, Mudhare, Muqtadheb, Mujtath,
    Mutakareb, Mutadarak
)
from poet.utils.bohour.tafeela import Tafeela

logger = logging.getLogger(__name__)


class ProsodyEvaluator:
    """Validates Arabic poetry prosody using bohour library"""
    
    # Bahr name mapping to bohour classes
    BAHR_MAPPING = {
        # Full names
        "بحر الطويل": Taweel,
        "بحر المديد": Madeed,
        "بحر البسيط": Baseet,
        "بحر الوافر": Wafer,
        "بحر الكامل": Kamel,
        "بحر الهزج": Hazaj,
        "بحر الرجز": Rajaz,
        "بحر الرمل": Ramal,
        "بحر السريع": Saree,
        "بحر المنسرح": Munsareh,
        "بحر الخفيف": Khafeef,
        "بحر المضارع": Mudhare,
        "بحر المقتضب": Muqtadheb,
        "بحر المجتث": Mujtath,
        "بحر المتقارب": Mutakareb,
        "بحر المحدث": Mutadarak,
        
        # Short names
        "طويل": Taweel,
        "مديد": Madeed,
        "بسيط": Baseet,
        "وافر": Wafer,
        "كامل": Kamel,
        "هزج": Hazaj,
        "رجز": Rajaz,
        "رمل": Ramal,
        "سريع": Saree,
        "منسرح": Munsareh,
        "خفيف": Khafeef,
        "مضارع": Mudhare,
        "مقتضب": Muqtadheb,
        "مجتث": Mujtath,
        "متقارب": Mutakareb,
        "محدث": Mutadarak,
        
        # English names
        "Taweel": Taweel,
        "Madeed": Madeed,
        "Baseet": Baseet,
        "Wafer": Wafer,
        "Kamel": Kamel,
        "Hazaj": Hazaj,
        "Rajaz": Rajaz,
        "Ramal": Ramal,
        "Saree": Saree,
        "Munsareh": Munsareh,
        "Khafeef": Khafeef,
        "Mudhare": Mudhare,
        "Muqtadheb": Muqtadheb,
        "Mujtath": Mujtath,
        "Mutakareb": Mutakareb,
        "Mutadarak": Mutadarak,
    }
    
    def __init__(self, llm_provider: Optional[BaseLLM] = None):
        """Initialize ProsodyEvaluator"""
        self.prompt_manager = PromptManager()
        self.llm = llm_provider
    
    def validate_poem(self, poem: LLMPoem, bahr: str) -> LLMPoem:
        """
        Validate poem prosody and return updated poem with validation results.
        
        Assumes the poem has already been validated for line count and
        has proper diacritics applied.
        
        Note: Quality assessment is now handled by PoemEvaluator.
        """
        
        # Get bahr class
        bahr_class = self._get_bahr_class(bahr)
        if not bahr_class:
            logger.error(f"Unknown bahr: {bahr}")
            # Create a basic validation result for unknown bahr
            validation_result = ProsodyValidationResult(
                overall_valid=False,
                total_baits=0,
                valid_baits=0,
                invalid_baits=0,
                bait_results=[],
                validation_summary=f"بحر غير معروف: {bahr}",
                bahr_used=bahr
            )
            poem.prosody_validation = validation_result
            return poem
        
        # Validate each bait
        bait_results = []
        valid_baits = 0
        invalid_baits = 0
        
        for bait in poem.get_baits():
            result = self._validate_bait(bait, bahr_class)
            bait_results.append(result)
            
            if result.is_valid:
                valid_baits += 1
            else:
                invalid_baits += 1
        
        # Create validation result
        overall_valid = invalid_baits == 0
        validation_summary = self._generate_validation_summary(valid_baits, invalid_baits, bahr, bait_results)
        
        prosody_result = ProsodyValidationResult(
            overall_valid=overall_valid,
            total_baits=len(bait_results),
            valid_baits=valid_baits,
            invalid_baits=invalid_baits,
            bait_results=bait_results,
            bahr_used=bahr,
            validation_summary=validation_summary,
            diacritics_applied=True
        )
        
        # Update poem with validation results
        poem.prosody_validation = prosody_result
        
        return poem
    
    def _get_bahr_class(self, bahr_string: str):
        """Get bohour bahr class from string"""
        return self.BAHR_MAPPING.get(bahr_string)
    
    def _get_arabic_bahr_name(self, bahr_class):
        """Get Arabic bahr name from bahr class"""
        # Reverse mapping to find Arabic name
        for arabic_name, class_ref in self.BAHR_MAPPING.items():
            if class_ref == bahr_class:
                # Prefer full Arabic names over short ones
                if arabic_name.startswith("بحر "):
                    return arabic_name
                # If no full name found, return the short name
                return arabic_name
        
        # Fallback to class name if no Arabic name found
        return bahr_class.__name__
    
    def _validate_bait(self, bait: tuple, bahr_class) -> BaitValidationResult:
        """Validate a single bait against bahr patterns using LLM"""
        
        # Assume verses already have diacritics applied
        diacritized_verses = list(bait)
        
        # Concatenate diacritized verses with # for prosody analysis
        diacritized_bait = "#".join(diacritized_verses)
        
        try:
            # Get bahr instance to extract pattern
            bahr_instance = bahr_class()
            arabic_bahr_name = self._get_arabic_bahr_name(bahr_class)
            
            # Get the first expected pattern as reference (simplified approach)
            expected_patterns = bahr_instance.all_baits_combinations_patterns
            if expected_patterns:
                # Use the first pattern as reference - in practice, we'd want to show all valid patterns
                reference_pattern = expected_patterns[0]
                # Convert binary pattern to tafeelat representation for LLM
                bahr_pattern = self._convert_pattern_to_tafeelat(reference_pattern, bahr_class)
            else:
                bahr_pattern = "نمط غير معروف"
            
            # Use LLM to validate the bait
            validation_result = self._validate_bait_with_llm(
                diacritized_bait, 
                arabic_bahr_name, 
                bahr_pattern
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating bait: {e}")
            return BaitValidationResult(
                bait_text="#".join(bait),
                is_valid=False,
                pattern="",
                error_details=f"خطأ في التحقق: {str(e)}",
                diacritized_text=diacritized_bait
            )
    
    def _convert_pattern_to_tafeelat(self, pattern: str, bahr_class) -> str:
        """Convert binary pattern to tafeelat representation"""
        try:
            bahr_instance = bahr_class()
            # Get all valid combinations to find the one that matches this pattern
            combinations = bahr_instance.bait_combinations
            
            for combination in combinations:
                if isinstance(combination[0], Tafeela):
                    # Single shatr case
                    shatr = combination
                    combo_pattern = "".join("".join(map(str, tafeela.pattern)) for tafeela in shatr)
                    if combo_pattern == pattern:
                        return " ".join(str(tafeela) for tafeela in shatr)
                else:
                    # Two shatr case
                    first_shatr, second_shatr = combination
                    first_pattern = "".join("".join(map(str, tafeela.pattern)) for tafeela in first_shatr)
                    second_pattern = "".join("".join(map(str, tafeela.pattern)) for tafeela in second_shatr)
                    combo_pattern = first_pattern + second_pattern
                    if combo_pattern == pattern:
                        first_str = " ".join(str(tafeela) for tafeela in first_shatr)
                        second_str = " ".join(str(tafeela) for tafeela in second_shatr)
                        return f"{first_str}#{second_str}"
            
            # If no exact match found, return a simplified representation
            return f"نمط ثنائي: {pattern}"
            
        except Exception as e:
            logger.error(f"Error converting pattern to tafeelat: {e}")
            return f"نمط ثنائي: {pattern}"
    
    def _get_bahr_zehaf_elal_info(self, bahr_class) -> str:
        """Get zehaf and elal information for a bahr"""
        try:
            bahr_instance = bahr_class()
            info_parts = []
            
            # Get tafeelat structure
            tafeelat_names = [tafeela_class().name for tafeela_class in bahr_instance.tafeelat]
            info_parts.append(f"التفعيلات الأساسية: {' '.join(tafeelat_names)}")
            
            # Get allowed zehafs for each tafeela
            zehaf_info = []
            for i, tafeela_class in enumerate(bahr_instance.tafeelat):
                tafeela = tafeela_class()
                if tafeela.allowed_zehafs:
                    zehaf_names = [zehaf.__name__ for zehaf in tafeela.allowed_zehafs]
                    zehaf_info.append(f"التفعيلة {i+1} ({tafeela.name}): {', '.join(zehaf_names)}")
            
            if zehaf_info:
                info_parts.append("الزحافات المسموحة:")
                info_parts.extend(zehaf_info)
            
            # Get arod and dharb information
            if hasattr(bahr_instance, 'arod_dharbs_map') and bahr_instance.arod_dharbs_map:
                arod_info = []
                for ella_class, dharb_classes in bahr_instance.arod_dharbs_map.items():
                    ella_name = ella_class.__name__
                    dharb_names = [dharb.__name__ for dharb in dharb_classes]
                    arod_info.append(f"العروض ({ella_name}): {', '.join(dharb_names)}")
                
                if arod_info:
                    info_parts.append("العروض والضروب:")
                    info_parts.extend(arod_info)
            
            return "\n".join(info_parts)
            
        except Exception as e:
            logger.error(f"Error getting bahr zehaf/elal info: {e}")
            return "معلومات الزحاف والإعل غير متوفرة"
    
    def _validate_bait_with_llm(self, bait_text: str, bahr_name: str, bahr_pattern: str) -> BaitValidationResult:
        """Validate bait using LLM"""
        try:
            # Check if LLM is available
            if not self.llm:
                logger.warning("No LLM provider available, using fallback validation")
                return BaitValidationResult(
                    bait_text=bait_text,
                    is_valid=False,
                    pattern="",
                    error_details="لا يوجد مزود ذكاء اصطناعي متاح",
                    diacritized_text=bait_text
                )
            
            # Get zehaf and elal information
            bahr_class = self._get_bahr_class(bahr_name)
            if bahr_class:
                zehaf_elal_info = self._get_bahr_zehaf_elal_info(bahr_class)
            else:
                zehaf_elal_info = "معلومات غير متوفرة"
            
            # Format the prompt
            formatted_prompt = self.prompt_manager.format_prompt(
                "prosody_validation",
                bait_text=bait_text,
                bahr_name=bahr_name,
                bahr_pattern=bahr_pattern,
                zehaf_elal_info=zehaf_elal_info
            )
            
            # Get LLM response
            response_text = self.llm.generate(formatted_prompt)
            
            # Parse the JSON response
            try:
                # Extract JSON from the response (handle cases where LLM adds extra text)
                response_text = response_text.strip()
                
                # Try to find JSON in the response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    # Extract just the JSON part
                    response_text = response_text[json_start:json_end]
                else:
                    # Try to extract from markdown code blocks
                    if response_text.startswith("```json"):
                        response_text = response_text[7:]
                    if response_text.endswith("```"):
                        response_text = response_text[:-3]
                    response_text = response_text.strip()
                
                # If still no JSON found, try to create a fallback response
                if not response_text.startswith('{'):
                    logger.warning(f"LLM returned non-JSON response: {response_text[:100]}...")
                    # Create a fallback response based on the Arabic text
                    if "صحيح" in response_text or "مطابق" in response_text:
                        response_text = '{"is_valid": true, "pattern": "", "error_details": null}'
                    else:
                        response_text = '{"is_valid": false, "pattern": "", "error_details": "استجابة غير واضحة من الذكاء الاصطناعي"}'
                
                response = json.loads(response_text)
                
                if isinstance(response, dict):
                    is_valid = response.get("is_valid", False)
                    pattern = response.get("pattern", "")
                    error_details = response.get("error_details")
                    
                    return BaitValidationResult(
                        bait_text=bait_text,
                        is_valid=is_valid,
                        pattern=pattern,
                        error_details=error_details,
                        diacritized_text=bait_text
                    )
                else:
                    raise ValueError("Response is not a dictionary")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response: {e}")
                logger.error(f"Response text: {response_text}")
                return BaitValidationResult(
                    bait_text=bait_text,
                    is_valid=False,
                    pattern="",
                    error_details="فشل في تحليل استجابة الذكاء الاصطناعي",
                    diacritized_text=bait_text
                )
                
        except Exception as e:
            logger.error(f"Error in LLM validation: {e}")
            return BaitValidationResult(
                bait_text=bait_text,
                is_valid=False,
                pattern="",
                error_details=f"خطأ في التحقق: {str(e)}",
                diacritized_text=bait_text
            )
    
    def _generate_validation_summary(self, valid_baits: int, invalid_baits: int, bahr: str, bait_results: List[BaitValidationResult] = None) -> str:
        """Generate validation summary"""
        total_baits = valid_baits + invalid_baits
        
        if invalid_baits == 0:
            return f"جميع الأبيات ({total_baits}) صحيحة عروضياً على بحر {bahr}"
        elif valid_baits == 0:
            return f"جميع الأبيات ({total_baits}) خاطئة عروضياً على بحر {bahr}"
        else:
            # If less than 5 invalid baits, mention specific bait numbers
            if invalid_baits < 5 and bait_results:
                invalid_bait_numbers = []
                for i, result in enumerate(bait_results, 1):
                    if not result.is_valid:
                        invalid_bait_numbers.append(str(i))
                
                bait_numbers_str = "، ".join(invalid_bait_numbers)
                return f"{valid_baits} من {total_baits} أبيات صحيحة عروضياً على بحر {bahr}. الأبيات الخاطئة: {bait_numbers_str}"
            else:
                return f"{valid_baits} من {total_baits} أبيات صحيحة عروضياً على بحر {bahr}. عدد الأبيات الخاطئة: {invalid_baits}"

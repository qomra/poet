import logging
from typing import Dict, List, Optional, Any
from poet.models.poem import LLMPoem
from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
from poet.prompts.prompt_manager import PromptManager
from poet.evaluation.bohour.arudi_style import get_arudi_style

# Import bohour classes
from poet.evaluation.bohour.bahr import (
    Taweel, Madeed, Baseet, Wafer, Kamel, Hazaj, Rajaz, Ramal,
    Saree, Munsareh, Khafeef, Mudhare, Muqtadheb, Mujtath,
    Mutakareb, Mutadarak
)

logger = logging.getLogger(__name__)


class ProsodyValidator:
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
    
    def __init__(self):
        """Initialize ProsodyValidator"""
        self.prompt_manager = PromptManager()
    
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
        """Validate a single bait against bahr patterns"""
        
        # Assume verses already have diacritics applied
        diacritized_verses = list(bait)
        
        # Concatenate diacritized verses with # for prosody analysis
        diacritized_bait = "#".join(diacritized_verses)
        
        try:
            # Extract prosodic pattern using bohour
            try:
                arudi_result = get_arudi_style(diacritized_bait)
                logger.debug(f"Arudi result: {arudi_result}")
            except Exception as e:
                logger.error(f"Error getting arudi style: {e}")
                logger.error(f"Diacritized bait: {diacritized_bait}")
                return BaitValidationResult(
                    bait_text="#".join(bait),
                    is_valid=False,
                    pattern="",
                    error_details=f"فشل في استخراج النمط العروضي: {str(e)}",
                    diacritized_text=diacritized_bait
                )
            
            plain_text, pattern = arudi_result[0]
            logger.debug(f"Extracted pattern: {pattern}")
            logger.debug(f"Plain text: {plain_text}")
            
            # Get expected patterns for this bahr
            bahr_instance = bahr_class()
            expected_patterns = bahr_instance.all_baits_combinations_patterns
            
            # Check if pattern matches any expected pattern
            is_valid = pattern in expected_patterns
            
            error_details = None
            if not is_valid:
                # Show the actual bait text instead of technical patterns
                bait_text = "#".join(bait)
                # Get Arabic bahr name
                arabic_bahr_name = self._get_arabic_bahr_name(bahr_class)
                error_details = f"البيت '{bait_text}' لا يتبع وزن {arabic_bahr_name}"
            
            return BaitValidationResult(
                bait_text="#".join(bait),
                is_valid=is_valid,
                pattern=pattern,
                error_details=error_details,
                diacritized_text=diacritized_bait
            )
            
        except Exception as e:
            logger.error(f"Error validating bait: {e}")
            return BaitValidationResult(
                bait_text="#".join(bait),
                is_valid=False,
                pattern="",
                error_details=f"خطأ في التحقق: {str(e)}",
                diacritized_text=diacritized_bait
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
    


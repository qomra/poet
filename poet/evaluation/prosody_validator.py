import logging
from typing import Dict, List, Optional, Any
from poet.models.poem import LLMPoem
from poet.models.prosody import ProsodyValidationResult, BaitValidationResult
from poet.models.quality import QualityAssessment
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
    
    def __init__(self, llm_provider=None):
        """Initialize ProsodyValidator with optional LLM provider for diacritics"""
        self.llm_provider = llm_provider
        self.prompt_manager = PromptManager()
    
    def validate_poem(self, poem: LLMPoem, bahr: str) -> LLMPoem:
        """Validate poem prosody and update poem with validation results"""
        
        # Validate line count first
        if not poem.validate_line_count():
            logger.warning(f"Poem has invalid line count: {len(poem.verses)}")
            self._update_poem_quality(poem, line_count_issues=["عدد الأبيات يجب أن يكون زوجياً"])
            return poem
        
        # Get bahr class
        bahr_class = self._get_bahr_class(bahr)
        if not bahr_class:
            logger.error(f"Unknown bahr: {bahr}")
            self._update_poem_quality(poem, prosody_issues=[f"بحر غير معروف: {bahr}"])
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
        
        # Update quality assessment
        self._update_poem_quality(poem, prosody_result)
        
        return poem
    
    def _get_bahr_class(self, bahr_string: str):
        """Get bohour bahr class from string"""
        return self.BAHR_MAPPING.get(bahr_string)
    
    def _apply_diacritics(self, verses: tuple) -> list:
        """Apply diacritics to verses using LLM"""
        if not self.llm_provider:
            return list(verses)
        
        try:
            # Join verses with newlines for tashkeel processing
            verses_text = '\n'.join(verses)
            
            # Get tashkeel prompt
            prompt = self.prompt_manager.format_prompt("tashkeel", text=verses_text)
            response = self.llm_provider.generate(prompt)
            
            # Try to parse as JSON first
            import json
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end > 0:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                    diacritized_verses = result.get("diacritized_verses", [])
                    
                    if diacritized_verses and len(diacritized_verses) == len(verses):
                        return diacritized_verses
                    elif diacritized_verses:
                        # Fallback: use the verses as returned
                        return diacritized_verses
                
                # If JSON parsing failed, try to extract from plain text
                lines = response.strip().split('\n')
                diacritized_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip empty lines and common prefixes
                    if line and not line.startswith(('```', '---', '===')):
                        # Check if this line contains Arabic with diacritics
                        if any(char in line for char in ['َ', 'ِ', 'ُ', 'ْ', 'ً', 'ٍ', 'ٌ', 'ّ']):
                            diacritized_lines.append(line)
                
                if diacritized_lines and len(diacritized_lines) == len(verses):
                    return diacritized_lines
                
                # If no diacritized text found, return original
                logger.warning(f"No diacritized text found in LLM response: {response[:200]}...")
                return list(verses)
                
            except json.JSONDecodeError:
                # JSON parsing failed, try to extract diacritized text from plain response
                logger.warning(f"JSON parsing failed, trying to extract from plain text: {response[:200]}...")
                
                # Look for lines with diacritics
                lines = response.strip().split('\n')
                diacritized_lines = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith(('```', '---', '===')):
                        if any(char in line for char in ['َ', 'ِ', 'ُ', 'ْ', 'ً', 'ٍ', 'ٌ', 'ّ']):
                            diacritized_lines.append(line)
                
                if diacritized_lines and len(diacritized_lines) == len(verses):
                    return diacritized_lines
                
                return list(verses)
            
        except Exception as e:
            logger.error(f"Failed to apply diacritics: {e}")
            return list(verses)
    
    def _validate_bait(self, bait: tuple, bahr_class) -> BaitValidationResult:
        """Validate a single bait against bahr patterns"""
        
        # Apply diacritics to the bait tuple (list of verses)
        diacritized_verses = self._apply_diacritics(bait)
        
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
                # Show a few expected patterns for debugging
                sample_patterns = list(expected_patterns)[:3] if expected_patterns else []
                error_details = f"النمط المستخرج '{pattern}' لا يتطابق مع أنماط بحر {bahr_class.__name__}. أمثلة على الأنماط المتوقعة: {sample_patterns}"
            
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
    
    def _update_poem_quality(self, poem: LLMPoem, prosody_result: Optional[ProsodyValidationResult] = None, 
                           line_count_issues: Optional[List[str]] = None, 
                           prosody_issues: Optional[List[str]] = None):
        """Update poem quality assessment"""
        
        # Collect issues
        all_line_count_issues = line_count_issues or []
        all_prosody_issues = prosody_issues or []
        
        # Add prosody issues from validation result
        if prosody_result:
            for bait_result in prosody_result.bait_results:
                if not bait_result.is_valid and bait_result.error_details:
                    all_prosody_issues.append(bait_result.error_details)
        
        # Calculate overall score
        overall_score = 1.0
        if all_line_count_issues:
            overall_score -= 0.3
        if all_prosody_issues:
            overall_score -= min(0.7, len(all_prosody_issues) * 0.1)
        
        # Determine if acceptable - unknown bahr should make it unacceptable
        is_acceptable = overall_score >= 0.7 and not all_line_count_issues and not any("غير معروف" in issue for issue in all_prosody_issues)
        
        # Generate recommendations
        recommendations = []
        if all_line_count_issues:
            recommendations.append("تأكد من أن عدد الأبيات زوجي")
        if all_prosody_issues:
            recommendations.append("راجع الأوزان العروضية للأبيات")
        
        # Create quality assessment
        poem.quality = QualityAssessment(
            prosody_issues=all_prosody_issues,
            line_count_issues=all_line_count_issues,
            overall_score=overall_score,
            is_acceptable=is_acceptable,
            recommendations=recommendations
        )









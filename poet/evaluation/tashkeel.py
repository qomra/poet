# poet/evaluation/qafiya_evaluator.py

import json
import logging
from typing import List, Optional, Dict, Any
from poet.models.poem import LLMPoem
from poet.models.tashkeel import TashkeelValidationResult, TashkeelBaitResult
from poet.llm.base_llm import BaseLLM
from poet.prompts.prompt_manager import PromptManager

class TashkeelValidationError(Exception):
    """Raised when qafiya validation fails"""
    pass


class TashkeelEvaluator:
    """
    Validates tashkeel (diacritics) in Arabic poetry using LLM analysis.
    
    Analyzes the diacritics applied to each verse to ensure they are correct
    and identifies any misapplied diacritics.
    """
    
    def __init__(self, llm: BaseLLM, prompt_manager: Optional[PromptManager] = None):
        self.llm = llm
        self.prompt_manager = prompt_manager or PromptManager()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
    
    def evaluate_tashkeel(self, poem: LLMPoem) -> TashkeelValidationResult:
        """Check if tashkeel needs fixing"""
        # check if there is non-vowel character in the poem that is not diacritized 
        # or there is a shaddah character that is followed by a haraka
        
        baits = poem.get_baits()
        
        # If no baits (odd number of verses), return invalid result
        if not baits:
            return TashkeelValidationResult(
                overall_valid=False,
                total_baits=0,
                valid_baits=0,
                invalid_baits=0,
                bait_results=[],
                validation_summary="لا يمكن تقييم التشكيل لعدد فردي من الأبيات",
                issues=["لا يمكن تقييم التشكيل لعدد فردي من الأبيات"]
            )
        
        bait_results = []
        for i, bait in enumerate(baits):
            bait_result = self._evaluate_single_bait(i, bait)
            bait_results.append(bait_result)
        
        # Check if any bait has issues
        has_issues = any(not result.is_valid for result in bait_results)
        
        return TashkeelValidationResult(
            overall_valid=not has_issues,
            total_baits=len(baits),
            valid_baits=sum(1 for result in bait_results if result.is_valid),
            invalid_baits=sum(1 for result in bait_results if not result.is_valid),
            bait_results=bait_results,
            validation_summary="التشكيل صحيح" if not has_issues else "يوجد أخطاء في التشكيل",
            issues=[result.error_details for result in bait_results if not result.is_valid and result.error_details]
        )
    
    def _evaluate_single_bait(self, bait_number: int, bait: tuple) -> TashkeelBaitResult:
        """Evaluate a single bait for tashkeel"""
        harakat = ["\u0650", "\u064E", "\u064F", "\u0652", "\u064B", "\u064C", "\u064D"]  # kasra, fatha, damma, sukun, tanween variants
        shadda = "\u0651"
        vowels = ["\u0627", "\u064A", "\u0648", "\u0649"]  # alif, yaa, waw, alif maqsura
        
        text = "#".join(bait)
        issues = []
        
        i = 0
        while i < len(text):
            char = text[i]
            
            # Skip non-Arabic characters (spaces, punctuation, etc.)
            if not ('\u0600' <= char <= '\u06FF'):
                i += 1
                continue
            
            # Skip vowel characters
            if char in vowels:
                i += 1
                continue
            
            # Check for Arabic consonants (non-vowel Arabic characters)
            if '\u0621' <= char <= '\u064A':  # Arabic letter range
                # Special case: Allow lam (ل) in definite article (ال) to be without haraka
                if char == '\u0644':  # lam (ل)
                    # Check if this is part of definite article ال
                    if i > 0 and text[i-1] == '\u0627':  # preceded by alif (ا)
                        i += 1
                        continue
                
                # Look ahead to see if this consonant is diacritized
                has_diacritic = False
                j = i + 1
                
                # Check the next few characters for diacritics
                while j < len(text) and j < i + 3:  # Look at next 2 positions max
                    next_char = text[j]
                    if next_char in harakat or next_char == shadda:
                        has_diacritic = True
                        break
                    elif '\u0600' <= next_char <= '\u06FF' and next_char not in harakat and next_char != shadda:
                        # Hit another Arabic character that's not a diacritic
                        break
                    j += 1
                
                # If consonant has no diacritic, add issue
                if not has_diacritic:
                    issues.append(f"حرف '{char}' بدون تشكيل")
            
            # Check for shadda followed by haraka
            if char == shadda:
                # Look at the next character to see if it's a haraka
                if i + 1 < len(text) and text[i + 1] in harakat:
                    issues.append(f"شدة متبوعة بحركة: {char}{text[i + 1]}")
            
            i += 1
        
        # Return result based on whether issues were found
        if issues:
            return TashkeelBaitResult(
                bait_number=bait_number,
                is_valid=False,
                error_details="; ".join(issues)
            )
        else:
            return TashkeelBaitResult(
                bait_number=bait_number,
                is_valid=True,
                error_details=None
            )
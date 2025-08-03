#!/usr/bin/env python3
"""
Extract rhyme words from the ashaar dataset.

This script loads the ashaar dataset and extracts the last word from each verse
to build a comprehensive rhyme dictionary for Arabic poetry generation.
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Set, Dict, List, Tuple
from pyarabic.araby import strip_tashkeel

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Error: datasets library not available. Install with: pip install datasets")
    sys.exit(1)

# Arabic text processing patterns
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670\u0640]')  # Diacritics and tatweel
ARABIC_PUNCTUATION = re.compile(r'[،؛؟!.\-\s]+$')  # Common punctuation at end
ARABIC_LETTERS = re.compile(r'^[\u0621-\u063A\u0641-\u064A\u067E\u0686\u0698\u06A9\u06AF\u06BE\u06CC]+$')

# Arabic vowel letters to check
VOWEL_LETTERS = {"و", "ي", "ا"}

# Haraka/diacritic marks to skip
HARAKA_MARKS = {"ـ", "ٰ", "ٔ", "ٕ"}

TANWEEN_MARKS = {"ْ", "َ", "ِ", "ُ"}

SHADDA_MARKS = {"ّ"}

ARABIC_LETTERS_MAP = {
 
  "ا": "ا",
  "ء": "ء",
  "ب": "ب",
  "ت": "ت",
  "ث": "ث",
  "ج": "ج",
  "ح": "ح",
  "خ": "خ",
  "د": "د",
  "ذ": "ذ",
  "ر": "ر",
  "ز": "ز",
  "س": "س",
  "ش": "ش",
  "ص": "ص",
  "ض": "ض",
  "ط": "ط",
  "ظ": "ظ",
  "ع": "ع",
  "غ": "غ",
  "ف": "ف",
  "ق": "ق",
  "ك": "ك",
  "ل": "ل",
  "م": "م",
  "ن": "ن",
  "ه": "ه",
  "و": "و",
  "ي": "ي",
  
  # Hamza carriers map to base hamza
  "أ": "ء",
  "إ": "ء", 
  "ئ": "ء",
  "آ": "ء",
  "ؤ": "ء",
  
  # Alif variants map to base alif
  "ٱ": "ا", # Alif Wasla
  "ى": "ا", # Alif Maksura
  
  # Ba forms map to base ba
  "بـ": "ب", # initial
  "ـبـ": "ب", # medial
  "ـب": "ب", # final
  
  # Ta forms map to base ta
  "تـ": "ت", # initial
  "ـتـ": "ت", # medial
  "ـت": "ت", # final
  
  # Tha forms map to base tha
  "ثـ": "ث", # initial
  "ـثـ": "ث", # medial
  "ـث": "ث", # final
  
  # Jim forms map to base jim
  "جـ": "ج", # initial
  "ـجـ": "ج", # medial
  "ـج": "ج", # final
  
  # Ha forms map to base ha
  "حـ": "ح", # initial
  "ـحـ": "ح", # medial
  "ـح": "ح", # final
  
  # Kha forms map to base kha
  "خـ": "خ", # initial
  "ـخـ": "خ", # medial
  "ـخ": "خ", # final
  
  # Dal forms map to base dal
  "ـد": "د", # final only
  
  # Dhal forms map to base dhal
  "ـذ": "ذ", # final only
  
  # Ra forms map to base ra
  "ـر": "ر", # final only
  
  # Zay forms map to base zay
  "ـز": "ز", # final only
  
  # Sin forms map to base sin
  "سـ": "س", # initial
  "ـسـ": "س", # medial
  "ـس": "س", # final
  
  # Shin forms map to base shin
  "شـ": "ش", # initial
  "ـشـ": "ش", # medial
  "ـش": "ش", # final
  
  # Sad forms map to base sad
  "صـ": "ص", # initial
  "ـصـ": "ص", # medial
  "ـص": "ص", # final
  
  # Dad forms map to base dad
  "ضـ": "ض", # initial
  "ـضـ": "ض", # medial
  "ـض": "ض", # final
  
  # Ta forms map to base ta
  "طـ": "ط", # initial
  "ـطـ": "ط", # medial
  "ـط": "ط", # final
  
  # Za forms map to base za
  "ظـ": "ظ", # initial
  "ـظـ": "ظ", # medial
  "ـظ": "ظ", # final
  
  # Ain forms map to base ain
  "عـ": "ع", # initial
  "ـعـ": "ع", # medial
  "ـع": "ع", # final
  
  # Ghain forms map to base ghain
  "غـ": "غ", # initial
  "ـغـ": "غ", # medial
  "ـغ": "غ", # final
  
  # Fa forms map to base fa
  "فـ": "ف", # initial
  "ـفـ": "ف", # medial
  "ـف": "ف", # final
  
  # Qaf forms map to base qaf
  "قـ": "ق", # initial
  "ـقـ": "ق", # medial
  "ـق": "ق", # final
  
  # Kaf forms map to base kaf
  "كـ": "ك", # initial
  "ـكـ": "ك", # medial
  "ـك": "ك", # final
  
  # Lam forms map to base lam
  "لـ": "ل", # initial
  "ـلـ": "ل", # medial
  "ـل": "ل", # final
  "لا": "ل", # lam-alif ligature
  "لاـ": "ل", # lam-alif initial (if exists)
  "ـلاـ": "ل", # lam-alif medial (if exists)
  "ـلا": "ل", # lam-alif final
  
  # Mim forms map to base mim
  "مـ": "م", # initial
  "ـمـ": "م", # medial
  "ـم": "م", # final
  
  # Nun forms map to base nun
  "نـ": "ن", # initial
  "ـنـ": "ن", # medial
  "ـن": "ن", # final
  
  # Ha forms map to base ha
  "هـ": "ه", # initial
  "ـهـ": "ه", # medial
  "ـه": "ه", # final
  
  # Waw forms map to base waw
  "ـو": "و", # final only
  
  # Ya forms map to base ya
  "يـ": "ي", # initial
  "ـيـ": "ي", # medial
  "ـي": "ي", # final
  
  # Ta Marbuta maps to ha
  "ة": "ه",
  
  # Persian/Urdu letters
  "پـ": "پ", # Pe initial
  "ـپـ": "پ", # Pe medial
  "ـپ": "پ", # Pe final
  
  "چـ": "چ", # Che initial
  "ـچـ": "چ", # Che medial
  "ـچ": "چ", # Che final
  
  "ـژ": "ژ", # Zhe final only
  
  "کـ": "ک", # Keheh initial
  "ـکـ": "ک", # Keheh medial
  "ـک": "ک", # Keheh final
  
  "گـ": "گ", # Gaf initial
  "ـگـ": "گ", # Gaf medial
  "ـگ": "گ", # Gaf final
  
  "یـ": "ی", # Farsi Yeh initial
  "ـیـ": "ی", # Farsi Yeh medial
  "ـی": "ی", # Farsi Yeh final
}



def load_ashaar_dataset():
    """Load the ashaar dataset."""
    kb_path = project_root / "kb" / "ashaar"
    if not kb_path.exists():
        print(f"Error: Dataset path not found: {kb_path}")
        sys.exit(1)
    
    print(f"Loading ashaar dataset from: {kb_path}")
    try:
        dataset_dict = load_dataset(str(kb_path))
        
        # Get the main split
        if 'train' in dataset_dict:
            dataset = dataset_dict['train']
        else:
            split_name = list(dataset_dict.keys())[0]
            dataset = dataset_dict[split_name]
            print(f"Using split '{split_name}' as main split")
        
        print(f"Dataset size: {len(dataset)}")
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

def is_valid_verse(verse: str) -> bool:
    """Check if a verse is valid."""
    # check if empty
    if verse.strip() == "":
        return False
    # check if it has at least 5 arabic letters
    if len(re.findall(r'[\u0600-\u06FF]', verse)) < 5:
        return False
    
    return True

def clean_verse(verse: str) -> str:
    """Clean a verse."""
    # remove ـ, ‏, ‎, ‬
    verse = verse.replace("ـ", "").replace("‏", "").replace("‎", "").replace("‬", "")
    # remove punctuation
    verse = re.sub(ARABIC_PUNCTUATION, "", verse)
    return verse

def extract_last_letter(word: str) -> str:
    """Extract the last letter of a list of words. 
    If the last letter is a vowel, then return the last non-vowel letter"""
    last_index = len(word) - 1
    if last_index == 0:
        return None
    if word[last_index] not in ["ي", "ا"]:
        return word[last_index]
    while word[last_index] in VOWEL_LETTERS and last_index >= 0:
        last_index -= 1
    if last_index < 0:
        return word[-2]
    return word[last_index]

def identify_rhyme_pattern(words: List[str]) -> str:
    """Identify the rhyme pattern of a list of words."""
    # first ensure that all words ends with the same letter or the same letter with ا or ي or وا
    no_tashkeel_words = [strip_tashkeel(w) for w in words]
    no_tashkeel_words = [w for w in no_tashkeel_words if len(w) > 0]
    if len(no_tashkeel_words) == 0:
        return None
    last_letter = extract_last_letter(no_tashkeel_words[0])
    if last_letter is None:
        return None
    if len(no_tashkeel_words) == 1:
        return last_letter
    
    for word in no_tashkeel_words[1:]:
        if extract_last_letter(word) != last_letter:
            # not unified pattern
            return None
    

    return last_letter


def map_rhyme_to_poem(dataset):
    """Map the rhyme to the poem."""
    mapped_rhyme = {}
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            print(f"Processing item {i} of {len(dataset)}")
        verses = item["poem verses"]
        if len(verses) % 2 != 0 and len(verses) <= 1:
            continue
        
        ajz = [clean_verse(v.strip()) for v in verses[1::2]]
        ajz = [a for a in ajz if len(a) > 0]
        darb = [a.split()[-1] for a in ajz if len(a.split()) > 0]
        rhyme = identify_rhyme_pattern(darb)
        if rhyme is not None:
            # if all letters in the rhyme are alphabetic, then add it to the mapped_rhyme
            no_tashkeel_rhyme = strip_tashkeel(rhyme)
            if all(letter.isalpha() for letter in no_tashkeel_rhyme):
                # check if the rhyme is in the ARABIC_LETTERS_MAP
                if no_tashkeel_rhyme in ARABIC_LETTERS_MAP:
                    rhyme = ARABIC_LETTERS_MAP[no_tashkeel_rhyme]
                else:
                    print(f"Rhyme {rhyme} not found in ARABIC_LETTERS_MAP")
                    continue
                if rhyme not in mapped_rhyme:
                    mapped_rhyme[rhyme] = []
                mapped_rhyme[rhyme].append(i) 
    return mapped_rhyme


def save_rhyme_dictionary(all_rhyme_words: Dict[str, Set[str]],  output_path: Path):
    """Save the rhyme dictionary to JSON files."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all rhyme words
    all_words_file = output_path / "rhyme_map.json"
    with open(all_words_file, 'w', encoding='utf-8') as f:
        json.dump(all_rhyme_words, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(all_rhyme_words):,} rhyme words to: {all_words_file}")
    
def main():
    """Main extraction process."""
    print("Ashaar Rhyme Words Extractor")
    print("=" * 60)
    
    # Load dataset
    dataset = load_ashaar_dataset()
    
    # Extract rhyme words
    rhyme_map = map_rhyme_to_poem(dataset)
   
    
    # Save results
    output_path = project_root / "kb"
    save_rhyme_dictionary(rhyme_map, output_path)

   
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 
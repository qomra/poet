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

def clean_arabic_word(word: str) -> str:
    """
    Clean Arabic word by removing diacritics and punctuation.
    
    Args:
        word: Raw Arabic word
        
    Returns:
        Cleaned word or empty string if invalid
    """
    if not word:
        return ""
    
    # Remove diacritics
    word = ARABIC_DIACRITICS.sub('', word)
    
    # Remove trailing punctuation
    word = ARABIC_PUNCTUATION.sub('', word)
    
    # Strip whitespace
    word = word.strip()
    
    # Check if it's a valid Arabic word
    if not word or not ARABIC_LETTERS.match(word):
        return ""
    
    return word

def extract_last_word(verse: str) -> str:
    """
    Extract the last meaningful word from a verse.
    
    Args:
        verse: Arabic verse text
        
    Returns:
        Last word or empty string if none found
    """
    if not verse or not isinstance(verse, str):
        return ""
    
    # Split into words and process from right to left (last to first)
    words = verse.strip().split()
    
    for word in reversed(words):
        cleaned = clean_arabic_word(word)
        if cleaned:
            return cleaned
    
    return ""

def get_rhyme_letter(word: str) -> str:
    """
    Extract the rhyme letter (last letter) from a word.
    
    Args:
        word: Cleaned Arabic word
        
    Returns:
        Last letter or empty string
    """
    if not word:
        return ""
    
    # Return the last alphabetical character
    last_letter_index = len(word) - 1
    while last_letter_index >= 0:
        letter = word[last_letter_index]
        if letter.isalpha():
            return word[last_letter_index:]
        last_letter_index -= 1
    return ""

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

def add_words_to_dictionary(ryhme_dictionary,words: List[str], rhyme_letter: str):
    """Add words to the rhyme dictionary."""
    words = set(words)
    if rhyme_letter not in ryhme_dictionary:
        ryhme_dictionary[rhyme_letter] = words
    else:
        ryhme_dictionary[rhyme_letter] = ryhme_dictionary[rhyme_letter] | words

def check_word_valid(word: str, rhyme_letter: str) -> bool:
    """Check if a word is valid for a given rhyme letter."""
    # check if word is not empty
    if word.strip() == "":
        return False
    # check if word is not tashkeel
    if strip_tashkeel(word) == "":
        return False

    # check if word ends with rhyme letter
    if word.endswith(rhyme_letter):
        return True
    
    stripped_word = strip_tashkeel(word)
    if stripped_word.endswith(rhyme_letter):
        return True

    # loose check if word ends with rhyme_letter + ي or ا or وا
    if word.endswith(rhyme_letter + "ي") or word.endswith(rhyme_letter + "ا") or word.endswith(rhyme_letter + "وا"):
        return True

    return False

def extract_rhyme_pattern(verses: List[str]) -> str:
    """Extract rhyme pattern from poem"""
    lines = [line.strip() for line in verses]
    
    # Get last words of each line
    last_words = []
    for line in lines:
        # Remove punctuation and get last word
        try:
            words = line.split()
            last_words.append(words[-1])
        except Exception as e:
            print(verses)
            raise e
            continue
    # Try to identify common ending patterns
    for i in range(3, 0, -1):
        current_ending = last_words[0][-i:]
        if all(word.endswith(current_ending) for word in last_words):
            return current_ending
    
    # return last alphabetical character from first word that is not tashkeel
    last_word = last_words[0]
    if len(strip_tashkeel(last_word)) > 0:
        # find last alphabetical character index
        last_letter_index = len(last_word) - 1
        while last_letter_index >= 0:
            letter = last_word[last_letter_index]
            if letter.isalpha():
                return last_word[last_letter_index:]
            last_letter_index -= 1
        
    return ""

def is_valid_verse(verse: str) -> bool:
    """Check if a verse is valid."""
    # check if empty
    if verse.strip() == "":
        return False
    # check if it has at least 5 arabic letters
    if len(re.findall(r'[\u0600-\u06FF]', verse)) < 5:
        return False
    
    return True

def extract_rhyme_words_from_dataset(dataset, max_items: int = None) -> Tuple[Set[str], Dict[str, List[str]], Dict[str, int]]:
    """
    Extract rhyme words from the dataset.
    
    Args:
        dataset: Loaded ashaar dataset
        max_items: Maximum number of items to process (None for all)
        
    Returns:
        ryme_dictionary: Dictionary of rhyme words
    """
    ryhme_dictionary = {}
    
    processed_items = 0
    
    print(f"\nProcessing dataset items...")
    
    for i, item in enumerate(dataset):
        if max_items and processed_items >= max_items:
            break
        # Look for verse text in different possible fields
        verses = [verse.strip().replace("ـ", "").replace("‏", "").replace("‎", "").replace("‬", "") for verse in item["poem verses"] if is_valid_verse(verse.strip())]
        # remove empty verses
        verses = [verse for verse in verses if verse.strip() != ""]
        
        if len(verses) == 0:
            pass    
        elif len(verses) == 1:
            rhyme_letter = get_rhyme_letter(verses[0].split()[-1])
            if rhyme_letter != "":
                words = [verse.split()[-1] for verse in verses]
                add_words_to_dictionary(ryhme_dictionary, words, rhyme_letter)
        elif len(verses) % 2 != 0:
            pass
        else:
            # get 1,2, or 3 letters rhyme from the last 2-3 verses
            rhyme_letter = extract_rhyme_pattern(verses[-6:][1::2])
            if rhyme_letter != "":
                try:
                    words = [verse.split()[-1] for verse in verses[1::2] if check_word_valid(verse.split()[-1],rhyme_letter)]
                    add_words_to_dictionary(ryhme_dictionary, words, rhyme_letter)
                except Exception as e:
                    print(f"Error processing item {i}: {e}")
                    continue


        
        processed_items += 1
        
        # Progress update
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1:,} items, found {len(ryhme_dictionary):,} unique rhyme words")
        
    print(f"\nProcessing complete:")
    print(f"  Total items processed: {processed_items:,}")
    print(f"  Unique rhyme words found: {len(ryhme_dictionary):,}")
    
    return ryhme_dictionary

def save_rhyme_dictionary(all_rhyme_words: Dict[str, Set[str]],  output_path: Path):
    """Save the rhyme dictionary to JSON files."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all rhyme words
    all_words_file = output_path / "all_rhyme_words.json"
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
    all_rhyme_words = extract_rhyme_words_from_dataset(dataset)
    # change set to list
    all_rhyme_words = {k: list(v) for k, v in all_rhyme_words.items()}
    
    
    # Save results
    output_path = project_root / "kb"
    save_rhyme_dictionary(all_rhyme_words, output_path)

    # save keys to a file
    with open(project_root / "kb" / "rhyme_keys.txt", "w") as f:
        for key in all_rhyme_words.keys():
            f.write(key + "\n")
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 
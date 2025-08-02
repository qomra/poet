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
    
    # Return the last character
    return word[-1]

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

def add_words_to_dictionary(words: List[str], rhyme_letter: str):
    """Add words to the rhyme dictionary."""
    words = set(words)
    if rhyme_letter not in ryhme_dictionary:
        ryhme_dictionary[rhyme_letter] = words
    else:
        ryhme_dictionary[rhyme_letter] = ryhme_dictionary[rhyme_letter] + words


def extract_rhyme_pattern(poem: list[str]) -> str:
    """Extract rhyme pattern from poem"""
    lines = [line.strip() for line in poem]
    
    # Get last words of each line
    last_words = []
    for line in lines:
        # Remove punctuation and get last word
        words = line.split()
        last_words.append(words[-1])
    
    # Try to identify common ending patterns
    common_endings = {}
    for word in last_words:
        ending = word[-3:]  # Last 3 characters
        common_endings[ending] = common_endings.get(ending, 0) + 1

    if common_endings:
        most_common = max(common_endings.items(), key=lambda x: x[1])
        return most_common[0]
    
    return ""


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
    skipped_items = 0
    
    print(f"\nProcessing dataset items...")
    
    for i, item in enumerate(dataset):
        if max_items and processed_items >= max_items:
            break
        
        # Look for verse text in different possible fields
        verses = item["verses"]

        if len(verses) > 0:

            if len(verses) == 1:
                rhyme_letter = get_rhyme_letter(verses[0].split()[-1])
                if rhyme_letter:
                    words = [verse.split()[-1] for verse in verses]
                    add_words_to_dictionary(words, rhyme_letter)
            else:
                # get 1,2, or 3 letters rhyme from the last 2-3 verses
                rhyme_letter = extract_rhyme_pattern(verses[-3:])
                if rhyme_letter:
                    words = [verse.split()[-1] for verse in verses]
                    add_words_to_dictionary(words, rhyme_letter)

        
            processed_items += 1
            
            # Progress update
            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1:,} items, found {len(all_rhyme_words):,} unique rhyme words")
        
    print(f"\nProcessing complete:")
    print(f"  Total items processed: {processed_items:,}")
    print(f"  Items skipped: {skipped_items:,}")
    print(f"  Unique rhyme words found: {len(all_rhyme_words):,}")
    print(f"  Rhyme letters found: {len(rhyme_groups):,}")
    
    return all_rhyme_words, dict(rhyme_groups), dict(rhyme_letter_counts)

def analyze_rhyme_patterns(rhyme_groups: Dict[str, List[str]], rhyme_letter_counts: Dict[str, int]):
    """Analyze and display rhyme patterns."""
    print(f"\n{'='*60}")
    print("RHYME PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    # Sort rhyme letters by frequency
    sorted_letters = sorted(rhyme_letter_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 most common rhyme letters:")
    print(f"{'Letter':<8} {'Count':<10} {'Unique Words':<15} {'Examples'}")
    print(f"{'-'*8} {'-'*10} {'-'*15} {'-'*30}")
    
    for letter, count in sorted_letters[:20]:
        unique_words = len(set(rhyme_groups[letter]))
        examples = ', '.join(list(set(rhyme_groups[letter]))[:5])
        print(f"{letter:<8} {count:<10,} {unique_words:<15,} {examples}")
    
    # Find rhyme letters with most diverse vocabulary
    print(f"\nRhyme letters with most diverse vocabulary:")
    print(f"{'Letter':<8} {'Unique Words':<15} {'Total Count':<12} {'Examples'}")
    print(f"{'-'*8} {'-'*15} {'-'*12} {'-'*30}")
    
    diversity_sorted = sorted(rhyme_groups.items(), key=lambda x: len(set(x[1])), reverse=True)
    for letter, words in diversity_sorted[:15]:
        unique_count = len(set(words))
        total_count = len(words)
        examples = ', '.join(list(set(words))[:5])
        print(f"{letter:<8} {unique_count:<15,} {total_count:<12,} {examples}")

def save_rhyme_dictionary(all_rhyme_words: Set[str], rhyme_groups: Dict[str, List[str]], output_path: Path):
    """Save the rhyme dictionary to JSON files."""
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all rhyme words
    all_words_file = output_path / "all_rhyme_words.json"
    with open(all_words_file, 'w', encoding='utf-8') as f:
        json.dump(sorted(list(all_rhyme_words)), f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(all_rhyme_words):,} rhyme words to: {all_words_file}")
    
    # Save rhyme groups (deduplicated)
    rhyme_groups_dedup = {
        letter: sorted(list(set(words))) 
        for letter, words in rhyme_groups.items()
    }
    
    rhyme_groups_file = output_path / "rhyme_groups.json"
    with open(rhyme_groups_file, 'w', encoding='utf-8') as f:
        json.dump(rhyme_groups_dedup, f, ensure_ascii=False, indent=2)
    
    print(f"Saved rhyme groups for {len(rhyme_groups_dedup):,} letters to: {rhyme_groups_file}")
    
    # Save statistics
    stats = {
        "total_unique_words": len(all_rhyme_words),
        "total_rhyme_letters": len(rhyme_groups),
        "rhyme_letter_stats": {
            letter: {
                "unique_words": len(set(words)),
                "total_occurrences": len(words)
            }
            for letter, words in rhyme_groups.items()
        }
    }
    
    stats_file = output_path / "rhyme_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"Saved statistics to: {stats_file}")

def main():
    """Main extraction process."""
    print("Ashaar Rhyme Words Extractor")
    print("=" * 60)
    
    # Load dataset
    dataset = load_ashaar_dataset()
    
    # Extract rhyme words
    all_rhyme_words, rhyme_groups, rhyme_letter_counts = extract_rhyme_words_from_dataset(dataset)
    
    # Analyze patterns
    analyze_rhyme_patterns(rhyme_groups, rhyme_letter_counts)
    
    # Save results
    output_path = project_root / "data" / "rhyme_dictionary"
    save_rhyme_dictionary(all_rhyme_words, rhyme_groups, output_path)
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 
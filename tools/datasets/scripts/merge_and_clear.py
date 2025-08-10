from pathlib import Path
import json
from pyarabic.araby import strip_tashkeel

def process_rhyme_words(all_rhyme_words, strip_tashkeel_func):
    """
    Process rhyme words according to Arabic phonetic rules.
    
    For each key that has single key and haraka:
    1. Keep the word if the letter before the rhyme key is not و or ي or ا
    2. If the letter before the rhyme key is و or ي or ا, then check if the 
       rhyme + و or ي or ا is in the keys. If it is, move the word to that key,
       otherwise create a new key with the letter before the rhyme key and 
       the rhyme key and add the word to the new key
    """
    
    # Arabic vowel letters to check
    VOWEL_LETTERS = {"و", "ي", "ا"}
    
    # Haraka/diacritic marks to skip
    HARAKA_MARKS = {"ـ", "ٰ", "ٔ", "ٕ"}
    
    # Dictionary to store new keys and their words
    new_rhyme_words = {}
    
    # Dictionary to track words to remove from original keys
    words_to_remove = {}
    
    # Get keys that have single letter + haraka only
    single_letter_keys = [
        key for key in all_rhyme_words 
        if len(strip_tashkeel_func(key)) == 1
    ]
    
    print(f"Processing {len(single_letter_keys)} single-letter rhyme keys...")
    
    for i, key in enumerate(single_letter_keys):
        if i % 50 == 0:
            print(f"Processing {i} of {len(single_letter_keys)}")
        
        # Initialize removal list for this key
        words_to_remove[key] = []
        
        # Process each word in this rhyme key
        for word in all_rhyme_words[key]:
            # Get word without the last two letters (rhyme part)
            if len(word) < 3:  # Need at least 3 letters to have a letter before rhyme
                continue
                
            word_minus_rhyme = word[:-2]
            last_letter = word_minus_rhyme[-1]
            
            # Skip if last letter is a haraka mark
            if last_letter in HARAKA_MARKS:
                continue
            
            # Check if the last letter is a vowel letter (و, ي, ا)
            if last_letter in VOWEL_LETTERS:
                new_key = key + last_letter
                
                # Check if the new key already exists in original data
                if new_key in all_rhyme_words:
                    # Move word to existing key
                    if new_key not in new_rhyme_words:
                        new_rhyme_words[new_key] = set(all_rhyme_words[new_key])
                    new_rhyme_words[new_key].add(word)
                else:
                    # Create new key
                    if new_key not in new_rhyme_words:
                        new_rhyme_words[new_key] = set()
                    new_rhyme_words[new_key].add(word)
                
                # Mark word for removal from original key
                words_to_remove[key].append(word)
    
    # Remove words from original keys
    for key, words in words_to_remove.items():
        if words:  # Only process if there are words to remove
            remaining_words = [w for w in all_rhyme_words[key] if w not in words]
            all_rhyme_words[key] = remaining_words
    
    # Add new keys to all_rhyme_words (convert sets back to lists)
    for key, word_set in new_rhyme_words.items():
        if key in all_rhyme_words:
            # Merge with existing key, removing duplicates
            all_rhyme_words[key] = list(set(all_rhyme_words[key]) | word_set)
        else:
            # Create new key
            all_rhyme_words[key] = list(word_set)
    
    print(f"Processing complete. Created/updated {len(new_rhyme_words)} rhyme keys.")
    return all_rhyme_words

def clean_non_arabic_words(all_rhyme_words):
    """
    Remove words that contain non-Arabic characters from the rhyme dictionary.
    
    This function filters out words containing:
    - Latin/English letters (a-z, A-Z)
    - Numbers (0-9)  
    - Special characters like punctuation marks (., >, <, [, ], etc.)
    - Any character that is not a valid Arabic letter or diacritic
    
    Args:
        all_rhyme_words (dict): Dictionary with rhyme keys and lists of words
        
    Returns:
        dict: Cleaned dictionary with only pure Arabic words
    """
    
    # Define Arabic character ranges
    # Arabic letters: ا-ي (U+0627 to U+064A)
    # Arabic supplement: ﭐ-﴾ (U+FB50 to U+FDFF) 
    # Arabic presentation forms: ﷀ-﷿ (U+FDF0 to U+FDFF)
    # Arabic diacritics: ً-ٟ (U+064B to U+065F)
    # Additional Arabic characters: ٠-ۿ (U+0660 to U+06FF)
    
    def is_arabic_char(char):
        """Check if a character is a valid Arabic character."""
        code = ord(char)
        return (
            # Main Arabic block
            (0x0600 <= code <= 0x06FF) or
            # Arabic Supplement  
            (0x0750 <= code <= 0x077F) or
            # Arabic Extended-A
            (0x08A0 <= code <= 0x08FF) or  
            # Arabic Presentation Forms-A
            (0xFB50 <= code <= 0xFDFF) or
            # Arabic Presentation Forms-B  
            (0xFE70 <= code <= 0xFEFF) or
            # Spaces and common punctuation that might be acceptable
            char in {' ', '\u200C', '\u200D'}  # Zero-width non-joiner/joiner
        )
    
    def is_pure_arabic_word(word):
        """Check if a word contains only Arabic characters."""
        if not word or not word.strip():
            return False
            
        # Check each character in the word
        for char in word:
            if not is_arabic_char(char):
                return False
        return True
    
    # Track statistics
    total_words_before = sum(len(words) for words in all_rhyme_words.values())
    removed_words_count = 0
    removed_keys_count = 0
    
    # Clean the dictionary
    cleaned_rhyme_words = {}
    
    print("Cleaning non-Arabic words from rhyme dictionary...")
    
    for i, (key, words) in enumerate(all_rhyme_words.items()):
        if i % 1000 == 0:
            print(f"Processing {i} of {len(all_rhyme_words)} keys...")
        
        # Filter words to keep only pure Arabic ones
        arabic_words = [word for word in words if is_pure_arabic_word(word)]
        
        # Count removed words
        removed_words_count += len(words) - len(arabic_words)
        
        # Only keep the key if it has Arabic words remaining
        if arabic_words:
            cleaned_rhyme_words[key] = arabic_words
        else:
            removed_keys_count += 1
    
    # Print statistics
    total_words_after = sum(len(words) for words in cleaned_rhyme_words.values())
    
    print(f"\nCleaning complete!")
    print(f"Words before: {total_words_before:,}")
    print(f"Words after: {total_words_after:,}")
    print(f"Words removed: {removed_words_count:,}")
    print(f"Keys removed: {removed_keys_count:,}")
    print(f"Keys remaining: {len(cleaned_rhyme_words):,}")
    
    return cleaned_rhyme_words

def remove_prefixes(all_rhyme_words, strip_tashkeel_func):
    """
    Remove Arabic prefixes from words and eliminate duplicates.
    
    Prefixes handled (with or without diacritics):
    - وبال, كال, وال, فال, بال, ولل → ال
    - لل → ال  
    - وا, فا → ا
    
    Only applies to words with stripped length > 4 letters.
    """
    
    # Define prefix mappings (stripped versions)
    PREFIX_RULES = {
        "وبال": "ال",
        "كال": "ال", 
        "وال": "ال",
        "فال": "ال",
        "بال": "ال",
        "ولل": "ال",
        "لل": "ال",
        "وا": "ا",
        "فا": "ا"
    }
    
    print(f"Removing prefixes from {len(all_rhyme_words)} rhyme keys...")
    
    for i, (key, words) in enumerate(all_rhyme_words.items()):
        if i % 1000 == 0:
            print(f"Processing {i} of {len(all_rhyme_words)} keys...")
            
        # Use set to automatically handle duplicates
        processed_words = set()
        
        for word in words:
            # Check if word is long enough (stripped length > 4)
            if len(strip_tashkeel_func(word)) <= 4:
                processed_words.add(word)
                continue
                
            # Strip diacritics for prefix matching
            stripped_word = strip_tashkeel_func(word)
            
            # Check each prefix against stripped word
            modified = False
            for stripped_prefix, replacement in PREFIX_RULES.items():
                if stripped_word.startswith(stripped_prefix):
                    # Find the actual prefix length in the original word
                    # by counting characters until we match the stripped prefix
                    original_prefix_len = 0
                    stripped_count = 0
                    
                    for char in word:
                        original_prefix_len += 1
                        # Only count non-diacritic characters
                        if strip_tashkeel_func(char) == char and char != '':
                            stripped_count += 1
                        # Stop when we've matched the stripped prefix length
                        if stripped_count == len(stripped_prefix):
                            break
                    
                    # Replace prefix and add to set
                    new_word = replacement + word[original_prefix_len:]
                    processed_words.add(new_word)
                    modified = True
                    break
            
            # If no prefix matched, keep original word
            if not modified:
                processed_words.add(word)
        
        # Convert back to list
        all_rhyme_words[key] = list(processed_words)
    
    print("Prefix removal complete.")
    return all_rhyme_words

def distribute_single_letter_keys(all_rhyme_words, strip_tashkeel_func):
    """
    Distribute words from single letter keys (no harakat) to appropriate rhyme keys.
    
    For each word in a single letter key:
    1. If word ends with ا و ي: add to letter+vowel+harakat combinations
    2. Otherwise: add to single letter+harakat combinations
    
    Harakat used: ْ (sukun), َ (fatha), ِ (kasra), ُ (damma)
    """
    
    # Arabic harakat to add
    HARAKAT = ["ْ", "َ", "ِ", "ُ"]
    
    # Vowel letters that affect distribution
    VOWEL_LETTERS = {"ا", "و", "ي"}
    
    print("Distributing single letter keys...")
    
    # Find keys that are single letters without harakat
    single_letter_keys = []
    for key in all_rhyme_words:
        stripped_key = strip_tashkeel_func(key)
        if len(stripped_key) == 1:
            # Check if original key has no harakat (same as stripped)
            if key == stripped_key:
                single_letter_keys.append(key)
    
    print(f"Found {len(single_letter_keys)} single letter keys without harakat")
    
    # Dictionary to collect new distributions
    new_distributions = {}
    
    # Dictionary to track keys to remove
    keys_to_remove = []
    
    for key in single_letter_keys:
        print(f"Processing key: '{key}'")
        words = all_rhyme_words[key]
        
        for word in words:
            if not word:
                continue
                
            # Get the last character of the word
            last_char = word[-1]
            
            # Case 1: Word ends with vowel letter (ا و ي)
            if last_char in VOWEL_LETTERS:
                # Create keys: letter + vowel + harakat
                for harakat in HARAKAT:
                    new_key = key + last_char + harakat
                    
                    if new_key not in new_distributions:
                        new_distributions[new_key] = set()
                    new_distributions[new_key].add(word)
            
            # Case 2: Word doesn't end with vowel letter
            else:
                # Create keys: letter + harakat
                for harakat in HARAKAT:
                    new_key = key + harakat
                    
                    if new_key not in new_distributions:
                        new_distributions[new_key] = set()
                    new_distributions[new_key].add(word)
        
        # Mark original key for removal
        keys_to_remove.append(key)
    
    # Remove original single letter keys
    for key in keys_to_remove:
        del all_rhyme_words[key]
    
    # Add new distributions to all_rhyme_words
    for new_key, word_set in new_distributions.items():
        if new_key in all_rhyme_words:
            # Merge with existing key, avoiding duplicates
            existing_words = set(all_rhyme_words[new_key])
            combined_words = existing_words | word_set
            all_rhyme_words[new_key] = list(combined_words)
        else:
            # Create new key
            all_rhyme_words[new_key] = list(word_set)
    
    print(f"Distribution complete. Created/updated {len(new_distributions)} rhyme keys.")
    print(f"Removed {len(keys_to_remove)} original single letter keys.")
    
    return all_rhyme_words

if __name__ == "__main__":
    # project root
    project_root = Path(__file__).parent.parent.parent.parent
    # load all_rhyme_words_clean.json
    with open(project_root / "kb" / "all_rhyme_words_clean.json", "r") as f:
        all_rhyme_words = json.load(f)
    
    all_rhyme_words = process_rhyme_words(all_rhyme_words, strip_tashkeel)

    all_rhyme_words = clean_non_arabic_words(all_rhyme_words)

    all_rhyme_words = remove_prefixes(all_rhyme_words, strip_tashkeel)

    all_rhyme_words = distribute_single_letter_keys(all_rhyme_words, strip_tashkeel)

    # save all_rhyme_words to a file
    with open(project_root / "kb" / "all_rhyme_words_clean_merged.json", "w") as f:
        json.dump(all_rhyme_words, f, ensure_ascii=False, indent=4)
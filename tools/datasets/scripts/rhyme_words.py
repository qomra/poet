import sys  
from pathlib import Path
import json

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Error: datasets library not available. Install with: pip install datasets")
    sys.exit(1)


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



def get_rhyme_words(dataset, rhyme_keys):
    """Get the rhyme words for the given dataset and rhyme keys."""
    rhyme_words = {key: set([]) for key in rhyme_keys}
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            print(f"Processing item {i} of {len(dataset)}")
        if len(item["poem verses"]) % 2 != 0:
            continue
        for verse in item["poem verses"][1::2]:
            last_word = verse.split()
            if len(last_word) == 0:
                continue
            last_word = last_word[-1].strip()
            # remoove ـ, ‏, ‎, ‬
            last_word = last_word.replace("ـ", "").replace("‏", "").replace("‎", "").replace("‬", "")
            # check if last 3, 2, 1 letters are in rhyme_keys
            for i in range(3, 0, -1):
                # use try catch for faster 
                try:
                    rhyme_words[last_word[-i:]] |=  set([last_word])
                except Exception as e:
                    continue

    return rhyme_words


if __name__ == "__main__":
    # project root
    project_root = Path(__file__).parent.parent.parent.parent
    # load rhyme keys
    with open(project_root / "kb" / "rhyme_keys_clean.txt", "r") as f:
        rhyme_keys = f.readlines()
    rhyme_keys = [key.strip() for key in rhyme_keys]

    # load ashaar dataset
    dataset = load_ashaar_dataset()

    rhyme_words = get_rhyme_words(dataset, rhyme_keys)
    # convert to list
    rhyme_words = {key: list(rhyme_words[key]) for key in rhyme_words}
    # load all_rhyme_words
    with open(project_root / "kb" / "all_rhyme_words_clean.json", "w") as f:
        json.dump(rhyme_words, f, ensure_ascii=False, indent=4)
        
    
    
    
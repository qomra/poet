#!/usr/bin/env python3
"""
Explore the ashaar dataset to understand the structure and unique values
in main categorical columns.
"""

import sys
from pathlib import Path
from collections import Counter

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


def explore_ashaar_dataset():
    """Explore the ashaar dataset structure and categorical values."""
    
    # Load dataset
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
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)
    
    # Get first few examples to understand structure
    print("\n" + "="*60)
    print("DATASET STRUCTURE")
    print("="*60)
    
    if len(dataset) > 0:
        first_item = dataset[0]
        print("Keys in dataset:")
        for key in sorted(first_item.keys()):
            value = first_item[key]
            value_type = type(value).__name__
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
            elif isinstance(value, list):
                preview = f"[{len(value)} items]"
            else:
                preview = str(value)
            print(f"  {key:25} ({value_type:10}): {preview}")
    
    # Explore categorical columns
    categorical_columns = [
        'poem meter', 'poem theme', 'poet name', 'poet era', 
        'poet location', 'poem language type'
    ]
    
    print("\n" + "="*60)
    print("CATEGORICAL COLUMN ANALYSIS")
    print("="*60)
    
    for column in categorical_columns:
        if column not in first_item:
            print(f"\nColumn '{column}' not found in dataset")
            continue
            
        print(f"\n{column.upper()}:")
        print("-" * 40)
        
        # Collect all values for this column
        values = []
        for item in dataset:
            value = item.get(column, '')
            if value and isinstance(value, str):
                values.append(value.strip())
        
        # Count unique values
        counter = Counter(values)
        total_unique = len(counter)
        total_items = len(values)
        
        print(f"Total items: {total_items}")
        print(f"Unique values: {total_unique}")
        print(f"Coverage: {(len([v for v in values if v]) / len(values) * 100):.1f}%")
        
        # Show top 20 most common values
        print(f"\nTop 20 most common values:")
        for value, count in counter.most_common(20):
            percentage = (count / total_items) * 100
            print(f"  {count:6d} ({percentage:5.1f}%) - {value}")
        
        # Show some examples of less common values
        if total_unique > 20:
            print(f"\nSample of less common values:")
            less_common = [item for item, count in counter.items() if count <= 5][:10]
            for value in less_common:
                count = counter[value]
                print(f"  {count:6d} - {value}")


def analyze_meter_patterns():
    """Analyze meter naming patterns specifically."""
    
    kb_path = project_root / "kb" / "ashaar"
    dataset_dict = load_dataset(str(kb_path))
    dataset = dataset_dict['train'] if 'train' in dataset_dict else dataset_dict[list(dataset_dict.keys())[0]]
    
    print("\n" + "="*60)
    print("METER PATTERN ANALYSIS")
    print("="*60)
    
    meters = []
    for item in dataset:
        meter = item.get('poem meter', '')
        if meter and isinstance(meter, str):
            meters.append(meter.strip())
    
    meter_counter = Counter(meters)
    
    # Group by patterns
    with_bahr = [m for m in meter_counter.keys() if 'بحر' in m]
    without_bahr = [m for m in meter_counter.keys() if 'بحر' not in m and m]
    
    print(f"Meters with 'بحر': {len(with_bahr)}")
    print(f"Meters without 'بحر': {len(without_bahr)}")
    
    print(f"\nTop meters with 'بحر':")
    for meter in sorted(with_bahr, key=lambda x: meter_counter[x], reverse=True)[:10]:
        print(f"  {meter_counter[meter]:6d} - {meter}")
    
    print(f"\nTop meters without 'بحر':")
    for meter in sorted(without_bahr, key=lambda x: meter_counter[x], reverse=True)[:10]:
        print(f"  {meter_counter[meter]:6d} - {meter}")


def analyze_theme_patterns():
    """Analyze theme naming patterns specifically."""
    
    kb_path = project_root / "kb" / "ashaar"
    dataset_dict = load_dataset(str(kb_path))
    dataset = dataset_dict['train'] if 'train' in dataset_dict else dataset_dict[list(dataset_dict.keys())[0]]
    
    print("\n" + "="*60)
    print("THEME PATTERN ANALYSIS")
    print("="*60)
    
    themes = []
    for item in dataset:
        theme = item.get('poem theme', '')
        if theme and isinstance(theme, str):
            themes.append(theme.strip())
    
    theme_counter = Counter(themes)
    
    # Group by patterns
    with_qaseeda = [t for t in theme_counter.keys() if 'قصيدة' in t]
    without_qaseeda = [t for t in theme_counter.keys() if 'قصيدة' not in t and t]
    
    print(f"Themes with 'قصيدة': {len(with_qaseeda)}")
    print(f"Themes without 'قصيدة': {len(without_qaseeda)}")
    
    print(f"\nTop themes with 'قصيدة':")
    for theme in sorted(with_qaseeda, key=lambda x: theme_counter[x], reverse=True)[:10]:
        print(f"  {theme_counter[theme]:6d} - {theme}")
    
    print(f"\nTop themes without 'قصيدة':")
    for theme in sorted(without_qaseeda, key=lambda x: theme_counter[x], reverse=True)[:10]:
        print(f"  {theme_counter[theme]:6d} - {theme}")


if __name__ == "__main__":
    print("Ashaar Dataset Explorer")
    print("=" * 60)
    
    explore_ashaar_dataset()
    analyze_meter_patterns()
    analyze_theme_patterns()
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE")
    print("="*60) 
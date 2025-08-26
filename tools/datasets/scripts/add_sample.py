from datasets import Dataset
from collections import Counter
from pyarabic.araby import strip_tashkeel
import os
import pandas as pd
import glob

def most_common_letter(letters):
    return Counter(letters).most_common(1)[0][0]

def get_letter_name(letter):
    name_map = {
        'ا': 'الف',
        'ب': 'الباء',
        'ت': 'التاء',
        'ث': 'الثاء',
        'ج': 'الجيم',
        'ح': 'الحاء',
        'خ': 'الخاء',
        'د': 'الدال',
        'ذ': 'الذال',
        'ر': 'الراء',
        'ز': 'الزاي',
        'س': 'السين',
        'ش': 'الشين',
        'ص': 'الصاد',
        'ض': 'الضاد',
        'ط': 'الطاء',
        'ظ': 'الظاء',
        'ع': 'العين',
        'غ': 'الغين',
        'ف': 'الفاء',
        'ق': 'القاف',
        'ك': 'الكاف',
        'ل': 'اللام',
        'م': 'الميم',
        'ن': 'النون',
        'ه': 'الهاء',
        'و': 'الواو',
        'ي': 'الياء',
    }
    return name_map.get(letter, letter)

def get_rhyme_letter(poem):
    # poem is a list of verses
    if len(poem) < 2:
        return ""
    # get even verses
    even_verses = poem[1::2]
    # get the last letter of each verse
    last_letters = [strip_tashkeel(verse[-1]) for verse in even_verses if len(verse) > 0]
    # get the most common letter
    letter =  most_common_letter(last_letters)
    letter_name = get_letter_name(letter)
    return letter_name

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "..", "..", "dataset", "ashaar_original", "data")
output_path = os.path.join(script_dir, "..", "..", "..", "dataset", "ashaar")

print(f"Reading parquet files from: {dataset_path}")

# Read all parquet files
parquet_files = glob.glob(os.path.join(dataset_path, "*.parquet"))
if not parquet_files:
    print(f"No parquet files found in {dataset_path}")
    exit(1)

print(f"Found {len(parquet_files)} parquet files")

# Load and combine all data
all_data = []
for i, file_path in enumerate(parquet_files):
    print(f"Loading file {i+1}/{len(parquet_files)}: {os.path.basename(file_path)}")
    df = pd.read_parquet(file_path)
    all_data.append(df)

# Combine all dataframes
print("Combining all data...")
combined_df = pd.concat(all_data, ignore_index=True)
print(f"Total rows: {len(combined_df)}")

# Add rhyme column
print("Adding rhyme column...")
combined_df['rhyme'] = combined_df['poem verses'].apply(get_rhyme_letter)

# Convert to datasets Dataset
print("Converting to datasets Dataset...")
dataset = Dataset.from_pandas(combined_df)
# filter in بحر الكامل and rhyme القاف
dataset = dataset.filter(lambda x: x['poem meter'] == 'بحر الكامل' and x['rhyme'] == 'القاف')

# Save using save_to_disk
print(f"Saving dataset to: {output_path}")
dataset.save_to_disk(output_path)
print("Dataset saved successfully using save_to_disk!")

from datasets import Dataset
from collections import Counter
from pyarabic.araby import strip_tashkeel
import os
import pandas as pd
import glob
from datasets import load_dataset
from pathlib import Path

# project dir is 3 levels up
project_root = Path(__file__).parent.parent.parent.parent

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


dataset = load_dataset("arbml/ashaar")["train"]
dataset = dataset.add_column(name="rhyme", column=[""]*len(dataset))


# Add rhyme column
print("Adding rhyme column...")
dataset = dataset.map(lambda x: {"rhyme": get_rhyme_letter(x["poem verses"])})

# Save using save_to_disk
print(f"Saving dataset to: {project_root / 'dataset' / 'ashaar_with_rhymes'}")
dataset.save_to_disk(project_root / "dataset" / "ashaar_with_rhymes")
print("Dataset saved successfully using save_to_disk!")

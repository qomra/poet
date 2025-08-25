from datasets import load_dataset
from collections import Counter
from pyarabic.araby import strip_tashkeel

def most_common_letter(letters):
    return Counter(letters).most_common(1)[0][0]


def get_rhyme_letter(poem):
    # poem is a list of verses
    if len(poem) < 2:
        return ""
    # get even verses
    even_verses = poem[1::2]
    # get the last letter of each verse
    last_letters = [strip_tashkeel(verse[-1]) for verse in even_verses if len(verse) > 0]
    # get the most common letter
    return most_common_letter(last_letters)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "..", "..", "..", "dataset", "ashaar_original")
output_path = os.path.join(script_dir, "..", "..", "..", "dataset", "ashaar")
dataset = load_dataset(dataset_path)["train"]
print(dataset[0])
# add new column to dataset
dataset = dataset.map(lambda x: {"rhyme": get_rhyme_letter(x["poem verses"])})
# save dataset
dataset.save_to_disk(output_path)

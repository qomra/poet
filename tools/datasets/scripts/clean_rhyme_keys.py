from pathlib import Path
from pyarabic.araby import strip_tashkeel

if __name__ == "__main__":
    # project root
    project_root = Path(__file__).parent.parent.parent.parent
    # load rhyme keys
    with open(project_root / "kb" / "rhyme_keys.txt", "r") as f:
        rhyme_keys = f.readlines()

    
    n_keys = []
    for key in rhyme_keys:
        key = key.strip()
        # if key is not all alph when removed tashkeel, then ignore it
        stripped_key = strip_tashkeel(key.strip())
        # check if any letter is not arabic
        if not all(char in "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهويى" for char in stripped_key):
            continue


        tashkeel_marks = [
            "ْ",  # sukun
            "ٌ",  # dammatan (tanween damma)
            "ٍ",  # kasratan (tanween kasra)
            "ً",  # fathatan (tanween fatha)
            "ُ",  # damma
            "ِ",  # kasra
            "َ",  # fatha
            "ّ",  # shaddah
            "ـ",  # tatweel (kashida)
            "ٰ",  # alif khanjariyya
            "ْ",  # sukun
        ]

        # if key is only tashkeel, then ignore it
        if all(char in tashkeel_marks for char in key):
            continue

        # Define shaddah specifically
        shaddah = "ّ"

        # Check if key has exactly 2 tashkeel that are not shaddah, plus one alpha character
        tashkeel_chars = [char for char in key if char in tashkeel_marks]
        non_shaddah_tashkeel = [char for char in tashkeel_chars if char != shaddah]
        alpha_chars = [char for char in key if char not in tashkeel_marks and char.strip()]

        if len(non_shaddah_tashkeel) >= 2 and len(alpha_chars) == 1:
            continue
        
        # ا alone is not a valid key
        if key == "ا":
            continue


        n_keys.append(key)

    # sort n_keys
    n_keys.sort()

    # save n_keys to a file
    with open(project_root / "kb" / "rhyme_keys_clean.txt", "w") as f:
        for key in n_keys:
            f.write(key + "\n")
    print(f"Removed {len(n_keys)} keys")
        

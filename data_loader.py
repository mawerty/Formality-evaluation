import os
import glob
import pandas as pd
from typing import List, Tuple, Optional

from config import DATASET_DIR, LABEL_MAP

def load_datasets(
    dataset_dir: str = DATASET_DIR,
    subset_prefix: Optional[str] = None,
    subset_length: Optional[str] = None
) -> Tuple[List[str], List[int], List[str], List[str]]:
    """Loads data, optionally filtering by prefix."""
    all_sentences = []
    all_labels = []
    all_sentences_types = []
    all_sentences_lenghts = []

    search_pattern = os.path.join(dataset_dir, "*.txt")
    all_txt_files = glob.glob(search_pattern)

    if not all_txt_files:
        print(f"Error: No .txt files found in directory: {dataset_dir}")
        return [], []

    if subset_prefix:
        print(f"Filtering datasets for subset prefix: '{subset_prefix}'")
        txt_files = [f for f in all_txt_files if os.path.basename(f).startswith(subset_prefix + "_")]
        if not txt_files:
             print(f"Error: No .txt files found matching prefix '{subset_prefix}_' in {dataset_dir}")
             return [], []
    else:
        txt_files = all_txt_files

    if subset_length:
        print(f"Filtering datasets for word length: '{subset_length}'")
        txt_files = [f for f in txt_files if os.path.basename(f).endswith(f"_{subset_length}.txt")]
        if not txt_files:
            print(f"Error: No .txt files found matching length '{subset_length}' in {dataset_dir}")
            return [], []

        

    for file_path in txt_files:
        try:
            df = pd.read_csv(file_path, sep=';', header=None, names=['sentence', 'label'], 
                            engine='python', quotechar='"', encoding='utf-8', on_bad_lines='error')

            df.dropna(inplace=True)
            df['sentence'] = df['sentence'].str.strip()
            df['label'] = df['label'].str.strip().str.lower() 

            df = df[df['label'].isin(LABEL_MAP.keys())]
            df['label'] = df['label'].map(LABEL_MAP)

            sentences = df['sentence'].tolist()
            labels = df['label'].tolist()
            sentences_lenght = []
            sentences_types = []

            if "small" in file_path:
                sentences_lenght = ["small"] * len(sentences)
            elif "medium" in file_path:
                sentences_lenght = ["medium"] * len(sentences)
            elif "long" in file_path:
                sentences_lenght = ["long"] * len(sentences)
            else:
                print(f"Warning: No length found for file {file_path}. Defaulting to 'other'.")
                sentences_lenght = ["other"] * len(sentences)
            
            if "education" in file_path:
                sentences_types = ["education"] * len(sentences)
            elif "emergency" in file_path:
                sentences_types = ["emergency"] * len(sentences)
            elif "guides" in file_path:
                sentences_types = ["guides"] * len(sentences)
            elif "workplace" in file_path:
                sentences_types = ["workplace"] * len(sentences)
            else:
                print(f"Warning: No type found for file {file_path}. Defaulting to 'other'.")
                sentences_types = ["other"] * len(sentences)
            
            all_sentences.extend(sentences)
            all_labels.extend(labels)
            all_sentences_types.extend(sentences_types)
            all_sentences_lenghts.extend(sentences_lenght)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Total sentences loaded for subset '{subset_prefix or 'all'}': {len(all_sentences)}")

    if len(all_sentences) != len(all_labels):
         print("Warning: Mismatch between number of sentences and labels loaded.")
    all_labels_int = [int(l) for l in all_labels]
    return all_sentences, all_labels_int, all_sentences_types, all_sentences_lenghts

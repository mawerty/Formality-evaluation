import os
import torch

DATASET_DIR = "datasets/"
OUTPUT_DIR = "outputs"

LABEL_MAP = {"informal": 0, "formal": 1}
POSITIVE_LABEL = 1 # Which label represents "formal"

PLOT_ROC = True
PLOT_PR = True
PLOT_CONFUSION_MATRIX = True
PLOT_SCORE_DISTRIBUTION = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)
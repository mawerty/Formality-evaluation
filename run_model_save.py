import os
import json
import numpy as np
import argparse
import sys

from data_loader import load_datasets
from models.hf_model import HuggingFaceModel
from models.llm_llamaapi_model import LLMModelLLAMA
from models.llm_openai_model import LLMModelOpenAI
from models.baseline_model import BaselineModel
import config

AVAILABLE_MODELS = {
    "xlmr_classifier": (HuggingFaceModel, "s-nlp/xlmr_formality_classifier"),
    "deberta_large_ranker": (HuggingFaceModel, "s-nlp/deberta-large-formality-ranker"),
    "mdistilbert_ranker": (HuggingFaceModel, "s-nlp/mdistilbert-base-formality-ranker"),
    "gpt4o-mini": (LLMModelOpenAI, "gpt-4o-mini"),
    "gpt4o": (LLMModelOpenAI, "gpt-4o"),
    "llama11b": (LLMModelLLAMA, "llama3.2-11b-vision"),
    "llama70b": (LLMModelLLAMA, "llama3.3-70b"),
    "deepseek-v3": (LLMModelLLAMA, "deepseek-v3"),
    "baseline": (BaselineModel, "baseline_model")
}

def main(args):
    """Main function to check cache, load data, initialize model, predict, and save."""

    model_name = args.model

    # --- 1. Check for Existing Output File ---
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "outputs")
    output_filename = f"{model_name}.json"
    output_file = os.path.join(output_dir, output_filename)

    if os.path.exists(output_file):
        print(f"Output file already exists: {output_file}")
        print("Skipping execution. To re-run, delete the existing file.")
        sys.exit(0)

    # --- 2. Load Dataset  ---
    print("\nOutput file not found. Proceeding with execution...")
    print("Loading dataset...")
    sentences, true_labels, sentence_types, sentence_lengths = load_datasets(
        config.DATASET_DIR,
        subset_prefix=None,
        subset_length=None
    )
    if not sentences:
        print("Error: No data loaded. Check dataset path in config.py.")
        sys.exit(1)

    print(f"Loaded {len(sentences)} sentences.")

    # --- 3. Initialize Model ---
    model_class, model_identifier = AVAILABLE_MODELS[model_name]
    print(f"\nInitializing model: {model_class.__name__} (Identifier: {model_identifier})...")

    try:
        model = model_class(model_identifier)
        print(f"Model '{model_name}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing model '{model_name}': {e}")
        sys.exit(1)
    # --- 4. Predict ---
    print(f"\nCalculating scores using model '{model_name}'...")
    try:
        scores = model.predict(sentences)
        print(f"Calculated {len(scores)} scores.")

        os.makedirs(output_dir, exist_ok=True)

        output_data = [
            [
                sentences[i],
                int(true_labels[i]) if isinstance(true_labels[i], np.integer) else true_labels[i],
                scores[i],
                sentence_types[i],
                sentence_lengths[i]
            ]
            for i in range(len(sentences))
        ]

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved new results to {output_file}")

    except Exception as e:
        print(f"Error during prediction or saving for model '{model_name}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run formality detection model prediction if results don't exist.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(AVAILABLE_MODELS.keys()),
        help="Name of the model to run."
    )

    args = parser.parse_args()
    main(args)
    print("\nScript finished.")
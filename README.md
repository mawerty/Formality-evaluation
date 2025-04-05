# Formality Detection Model Evaluation Framework

This evaluation system evaluates formality models on specially created dataset, and compares it to other methods.

## Files & Folders

*   `/datasets/`: Dataset files are in there, you can add your own categories following structure of other files `data_loader.py` reads from here.
*   `/models/`: Code for the models (transformers, LLMs, baseline). Inherit from `base_model.py` if you make a new one.
*   `/outputs/`: Where model predictions get saved (`model_name.json`).
*   `/results/`: Plots and CSVs comparing models go here.
*   `run_model_save.py`: Run one model, save its output.
*   `calculate_plots.py`: Make plots/CSVs from data in `/outputs/`.
*   `run_model.ipynb`: Jupyter notebook to run on specific data subsets.
*   `data_loader.py`, `evaluator.py`: Helper code, don't run these directly.
*   `requirements.txt`: packages required to run the code.

## Setup

1.  Clone it: `git clone <url>`
2.  Go into folder: `cd <folder>`
3.  Make a venv: `python -m venv venv` then `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4.  Install packages: `pip install -r requirements.txt`
5.  API Keys: If using OpenAI API/LlamaAPI etc., enter your API key in files.

## How to Use

**Step 1: Get Model Outputs**

*   Run `run_model_save.py` with the model you want.
    ```bash
    python run_model_save.py --model=gpt4o
    python run_model_save.py --model=baseline
    # Check the script for other model names
    ```
*   It saves a `.json` file in `/outputs/`.
*   **IMPORTANT:** If `outputs/gpt4o.json` already exists, it WON'T run `gpt4o` again. Delete the file if you want to re-run it.

**Step 2: Make Plots & Compare**

*   Run `calculate_plots.py`.
    ```bash
    python calculate_plots.py
    ```
*   It looks at all `.json` files in `/outputs/` and makes plots and summaries in `/results/`.

**Step 3: Notebook (If you want)**

*   Open `run_model.ipynb` to run models on specific parts of the data and see results right away.

## Adding a New Model

1.  Make a `.py` file in `/models/`.
2.  Make a class that inherits from `BaseModel` (in `base_model.py`).
3.  Add your model to the `AVAILABLE_MODELS` dictionary in `run_model_save.py` and `run_model.ipynb`. Give it a key (for `--model`) and an ID(useful if you want to run multiple very similar models using one file).
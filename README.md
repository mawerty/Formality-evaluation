# Formality Detection Model Evaluation System

This project provides a framework to evaluate and compare various formality detection models using a custom dataset.

## Project Overview

*   **/datasets/**: Input dataset files.
*   **/models/**: Model implementations (inherit from `base_model.py`). Includes examples like transformers, LLMs, and a baseline.
*   **/outputs/**: Raw prediction JSON files (`model_name.json`) for each model run.
*   **/results/**: Generated comparison plots (`.png`) and metrics (`.csv`).
*   **`run_model_save.py`**: Script to run a model on the full dataset and save outputs.
*   **`calculate_plots.py`**: Script to generate plots and metrics from saved outputs.
*   **`run_model.ipynb`**: Jupyter notebook for interactive testing on data subsets.
*   **`config.py`**: Configuration (paths, etc.).
*   **`data_loader.py` / `evaluator.py`**: Helper utilities (not run directly).
*   **`requirements.txt`**: Dependencies.
*   **`Report.pdf`**: Detailed report on the methodology, dataset, and results.

## Setup

1.  **Clone:** `git clone https://github.com/mawerty/Codellama-tuning.git` && `cd Codellama-tuning`
2.  **Virtual Env (Recommended):** `python -m venv venv` && `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
3.  **Install Deps:** `pip install -r requirements.txt`
4.  **Datasets:** Datasets are avaible in `/datasets/` (check `config.py` and `data_loader.py`).

## Usage Workflow

1.  **Generate Model Predictions:**
    *   Run a specific model on the entire dataset.
    *   Outputs are saved to `/outputs/model_name.json`.
    *   **Note:** The script will *not* overwrite existing output files. Delete a file manually if you need to re-run that model.
    ```bash
    # See AVAILABLE_MODELS in run_model_save.py for names
    python run_model_save.py --model=baseline
    python run_model_save.py --model=gpt4o
    ```

2.  **Calculate Results:**
    *   Generate comparative plots and metrics from all `.json` files in `/outputs`.
    *   Results are saved to `/results`.
    ```bash
    python calculate_plots.py
    ```

3.  **Interactive Analysis (Optional):**
    *   Use `run_model.ipynb` to test models on specific data subsets and view results immediately.

## Adding a New Model

1.  Create a new Python file in `/models`.
2.  Define a class inheriting from `BaseModel` (see `models/base_model.py`) and implement the `predict` method.
3.  Register your model class and a unique key in the `AVAILABLE_MODELS` dictionary in `run_model_save.py`.

## Output Format (`/outputs/model_name.json`)

The JSON file contains a list, where each item corresponds to a sentence and includes:
`[sentence_text, true_label, model_score, sentence_type, sentence_length]`

## Detailed Report

For a comprehensive overview of the approach, dataset creation, model choices, evaluation methodology, and results analysis, please refer to **`Report.pdf`**.
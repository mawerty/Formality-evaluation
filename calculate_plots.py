import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys

SENTENCE_TYPES_ORDER = ["education", "emergency", "guides", "workplace"]
SENTENCE_LENGTHS_ORDER = ["small", "medium", "long"]

INPUT_DIR = "outputs"
OUTPUT_DIR_BASE = "results" 
INDIVIDUAL_PLOT_DIR = os.path.join(OUTPUT_DIR_BASE, "individual_eval_plots")
COMPARISON_PLOT_DIR = os.path.join(OUTPUT_DIR_BASE, "comparison_plots")
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, "metrics_csv")
LOG_FILE_PATH = os.path.join(OUTPUT_DIR_BASE, "analysis_log.txt")

# --- Font Size Constants (for specific elements if needed) ---
# Using Seaborn context primarily, but these can override if necessary
FONT_SIZE_ANNOTATION = 11 # For heatmap values, scatter labels etc.
FONT_SIZE_SPECIFIC_TEXT = 12 # For fallback text like "N/A"

# --- Setup Logging ---
os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("--- Analysis Script Started ---")
logger.info(f"Input directory: {INPUT_DIR}")
logger.info(f"Output directory base: {OUTPUT_DIR_BASE}")

# --- Import User's Evaluation Function ---
try:
    from evaluator import evaluate_model
    try:
        from evaluator import POSITIVE_LABEL
        logger.info(f"Imported POSITIVE_LABEL = {POSITIVE_LABEL} from evaluator.py")
    except ImportError:
        logger.warning("POSITIVE_LABEL not found in evaluator.py, defaulting to 1.")
        POSITIVE_LABEL = 1
    logger.info("Successfully imported evaluate_model from evaluator.py")
except ImportError:
    logger.error("Could not import 'evaluate_model' from evaluator.py. Analysis will fail.")
    sys.exit("Error: evaluator.py not found or evaluate_model function missing.")
except Exception as e:
    logger.error(f"An unexpected error occurred during import from evaluator.py: {e}", exc_info=True)
    sys.exit(f"Error importing evaluator: {e}")

# --- Helper Functions ---

def load_data_from_json(input_dir):
    """Loads and preprocesses data from JSON files in the input directory."""
    logger.info(f"Loading data from JSON files in: {input_dir}")
    all_data = []
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    if not json_files:
        logger.error(f"No .json files found in directory: {input_dir}")
        return None

    for file_path in json_files:
        model_source = os.path.splitext(os.path.basename(file_path))[0]
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and all(isinstance(item, list) and len(item) == 5 for item in data):
                     for item in data:
                         item[4] = str(item[4]).lower().replace("short", "small")
                         all_data.append(item + [model_source])
                else:
                    logger.warning(f"Skipping {file_path}: Unexpected format.")
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Skipping {file_path} ({model_source}): Error reading/parsing file: {e}")

    if not all_data:
        logger.error("No valid data loaded from JSON files.")
        return None

    df = pd.DataFrame(all_data, columns=['sentence', 'true_label', 'score', 'sentence_type', 'sentence_length', 'model_source'])

    # Data Cleaning and Type Conversion
    df['true_label'] = pd.to_numeric(df['true_label'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    initial_rows = len(df)
    df.dropna(subset=['true_label', 'score'], inplace=True)
    if len(df) < initial_rows:
         logger.warning(f"Dropped {initial_rows - len(df)} rows with missing true_label or score.")

    if df.empty:
        logger.error("DataFrame is empty after loading and cleaning.")
        return None

    df['true_label'] = df['true_label'].astype(int)
    df['sentence_type'] = pd.Categorical(df['sentence_type'].astype(str), categories=SENTENCE_TYPES_ORDER, ordered=True)
    df['sentence_length'] = pd.Categorical(df['sentence_length'].astype(str), categories=SENTENCE_LENGTHS_ORDER, ordered=True)
    df['model_source'] = pd.Categorical(df['model_source'].astype(str))

    unexpected_lengths = df[~df['sentence_length'].isin(SENTENCE_LENGTHS_ORDER)]['sentence_length'].unique().tolist()
    if unexpected_lengths:
         logger.warning(f"Found unexpected sentence lengths after standardization: {unexpected_lengths}. Check input data or SENTENCE_LENGTHS_ORDER.")

    logger.info(f"Loaded DataFrame shape: {df.shape}. Models: {df['model_source'].unique().tolist()}")
    return df


def plot_evaluation_summary_individual(evaluation_results, title_prefix, output_path_prefix):
    """Generates and saves the 4 standard evaluation plots as separate files."""
    if evaluation_results is None or "error" in evaluation_results:
        logger.warning(f"Skipping individual plots for '{title_prefix}': Evaluation failed or returned error: {evaluation_results.get('error', 'None')}.")
        return

    can_plot_something = any(evaluation_results.get(key) is not None for key in ['confusion_matrix', 'roc_curve', 'pr_curve', 'scores'])
    if not can_plot_something:
        logger.warning(f"Skipping individual plots for '{title_prefix}': Missing all essential data for plotting.")
        return

    threshold = evaluation_results.get('threshold_applied', 'N/A')
    threshold_str = f"{threshold:.4f}" if isinstance(threshold, (int, float)) else str(threshold)
    metrics = evaluation_results.get('metrics', {})
    roc_auc = metrics.get('roc_auc', np.nan)
    avg_precision = metrics.get('average_precision', metrics.get('pr_auc', np.nan))

    # Using context should handle most font sizes, explicit title needed if context is set outside
    base_title = f"{title_prefix}\n(Thr={threshold_str})"

    plot_functions = {
        "confusion_matrix": plot_cm,
        "roc_curve": plot_roc,
        "pr_curve": plot_pr,
        "score_distribution": plot_dist
    }

    for plot_key, plot_func in plot_functions.items():
        required_data_key = 'scores' if plot_key == 'score_distribution' else plot_key
        if evaluation_results.get(required_data_key) is not None:
             plot_path = f"{output_path_prefix}_{plot_key}.png"
             try:
                  plot_func(evaluation_results, base_title, metrics, threshold, roc_auc, avg_precision, plot_path)
             except Exception as e:
                  logger.error(f"Failed to generate {plot_key} plot for '{title_prefix}': {e}", exc_info=False)


# --- Individual Plotting Functions (Adjusted for Context/Specific Fonts) ---

def plot_cm(eval_results, base_title, metrics, threshold, roc_auc, avg_precision, plot_path):
    fig, ax = plt.subplots(figsize=(8, 7)) # Slightly larger
    try:
        cm = np.array(eval_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=[f'Pred {i}' for i in range(cm.shape[1])],
                   yticklabels=[f'Actual {i}' for i in range(cm.shape[0])],
                   annot_kws={"size": FONT_SIZE_ANNOTATION}) # Explicit size for numbers
        ax.set_title(f'{base_title}\nConfusion Matrix') # Context should handle size
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        plt.tight_layout(); plt.savefig(plot_path)
    finally: plt.close(fig)

def plot_roc(eval_results, base_title, metrics, threshold, roc_auc, avg_precision, plot_path):
    fig, ax = plt.subplots(figsize=(8, 7)) # Slightly larger
    try:
        roc_data = eval_results['roc_curve']
        fpr, tpr = roc_data.get('fpr'), roc_data.get('tpr')
        if fpr is not None and tpr is not None:
            ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC={roc_auc:.3f})') # Thicker line
        else: ax.text(0.5, 0.5, "ROC N/A", ha='center', va='center', fontsize=FONT_SIZE_SPECIFIC_TEXT) # Explicit size for N/A
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{base_title}\nReceiver Operating Characteristic') # Context handles size
        ax.legend(loc="lower right") # Context handles size
        plt.tight_layout(); plt.savefig(plot_path)
    finally: plt.close(fig)

def plot_pr(eval_results, base_title, metrics, threshold, roc_auc, avg_precision, plot_path):
     fig, ax = plt.subplots(figsize=(8, 7)) # Slightly larger
     try:
         pr_data = eval_results['pr_curve']
         precision, recall = pr_data.get('precision'), pr_data.get('recall')
         true_labels = np.array(eval_results.get('true_labels', []))
         baseline = np.mean(true_labels == POSITIVE_LABEL) if len(true_labels) > 0 else 0.0
         if precision is not None and recall is not None:
             ax.plot(recall, precision, color='blue', lw=2.5, label=f'PR (AP={avg_precision:.3f})') # Thicker line
         else: ax.text(0.5, 0.5, "PR N/A", ha='center', va='center', fontsize=FONT_SIZE_SPECIFIC_TEXT) # Explicit size for N/A
         ax.plot([0, 1], [baseline, baseline], color='red', linestyle='--', label=f'Baseline ({baseline:.3f})')
         ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
         ax.set_xlabel('Recall')
         ax.set_ylabel('Precision')
         ax.set_title(f'{base_title}\nPrecision-Recall Curve') # Context handles size
         ax.legend(loc="lower left") # Context handles size
         plt.tight_layout(); plt.savefig(plot_path)
     finally: plt.close(fig)

def plot_dist(eval_results, base_title, metrics, threshold, roc_auc, avg_precision, plot_path):
     fig, ax = plt.subplots(figsize=(10, 7)) # Slightly larger
     try:
         scores = np.array(eval_results.get('scores', []))
         true_labels = np.array(eval_results.get('true_labels', []))
         if len(scores) > 0 and len(scores) == len(true_labels):
             unique_labels = sorted(np.unique(true_labels))
             palette = sns.color_palette('viridis', n_colors=len(unique_labels))
             sns.histplot(x=scores, hue=true_labels, element='step', stat='density',
                          common_norm=False, palette=palette, ax=ax, bins=50, hue_order=unique_labels)

             if isinstance(threshold, (int, float)):
                 ax.axvline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')

             legend_elements = []
             if isinstance(threshold, (int, float)):
                 legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', label=f'Threshold ({threshold:.3f})'))

             for i, label in enumerate(unique_labels):
                 label_scores = scores[true_labels == label]
                 if len(label_scores) > 0:
                    mean_score = label_scores.mean()
                    line = ax.axvline(mean_score, color=palette[i % len(palette)], linestyle=':', alpha=0.7, label=f'Mean ({label}): {mean_score:.3f}')
                    legend_elements.append(line)

             ax.legend(title='Class / Info') # Context handles size
             ax.set_title(f'{base_title}\nScore Distribution by True Class') # Context handles size
             ax.set_xlabel('Prediction Score')
             ax.set_ylabel('Density')
             plt.tight_layout(); plt.savefig(plot_path)
         else:
             logger.debug(f"Skipping Score Dist plot for '{base_title}': Insufficient or mismatched scores/labels.")
     finally: plt.close(fig)


# --- Analysis Functions ---

def generate_individual_plots(df, output_dir):
    """Generates individual evaluation plots for different data slices."""
    if df is None or df.empty:
        logger.warning("Individual plot generation skipped: DataFrame is empty.")
        return

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"--- Generating Individual Evaluation Plots (Output Dir: {output_dir}) ---")

    analysis_groups = {
        "by_model": ['model_source'], "by_type": ['sentence_type'], "by_length": ['sentence_length'],
        "by_model_and_type": ['model_source', 'sentence_type'], "by_model_and_length": ['model_source', 'sentence_length']
    }

    for dirname, group_cols in analysis_groups.items():
        plot_subdir = os.path.join(output_dir, dirname)
        os.makedirs(plot_subdir, exist_ok=True)
        logger.info(f"--- Generating plots grouped by: {', '.join(group_cols)} ---")

        grouped = df.groupby(group_cols, observed=False)
        if grouped.ngroups == 0: logger.info(f"    No groups found for {', '.join(group_cols)}. Skipping."); continue

        for group_keys, group_df in grouped:
            if isinstance(group_keys, tuple):
                keys_str_list = [str(k) for k in group_keys]
                group_name_title = ", ".join(f"{col}={key}" for col, key in zip(group_cols, keys_str_list))
                group_name_file = "_".join(f"{col}-{key}" for col, key in zip(group_cols, keys_str_list))
            else:
                group_name_title = f"{group_cols[0]}={group_keys}"
                group_name_file = f"{group_cols[0]}-{group_keys}"
            group_name_file = "".join(c if c.isalnum() else "_" for c in group_name_file).replace("__", "_")

            if len(group_df) == 0: continue
            if len(pd.unique(group_df['true_label'])) < 2:
                 logger.debug(f"    Skipping group '{group_name_title}': Only one class present (Size: {len(group_df)}).")
                 continue
            try:
                eval_results = evaluate_model(group_df['score'].tolist(), group_df['true_label'].tolist())
                plot_path_prefix = os.path.join(plot_subdir, f"eval_summary_{group_name_file}")
                plot_evaluation_summary_individual(eval_results, title_prefix=group_name_title, output_path_prefix=plot_path_prefix)
            except Exception as e:
                logger.error(f"  Error evaluating or plotting group '{group_name_title}': {e}", exc_info=False)
    logger.info("--- Individual Plot Generation Complete ---")


def generate_comparison_plots_and_csv(df, plot_output_dir, csv_output_dir):
    """Generates comparison plots across models (sorted by accuracy, consistent colors, larger fonts)
       and detailed/summary CSV files."""
    if df is None or df.empty:
        logger.warning("Comparison plot/CSV generation skipped: DataFrame is empty.")
        return

    os.makedirs(plot_output_dir, exist_ok=True)
    os.makedirs(csv_output_dir, exist_ok=True)
    logger.info(f"--- Generating Comparison Plots & Metrics CSVs (Sorted by Accuracy) ---")
    logger.info(f"--- Output Dirs: {plot_output_dir}, {csv_output_dir} ---")

    # --- Prepare Detailed Metrics Data ---
    detailed_metrics_list = []
    grouping_cols = ['model_source', 'sentence_type', 'sentence_length']
    all_models_overall_eval_results = {}
    logger.info("Calculating metrics for detailed scenarios (Model x Type x Length)...")
    for group_keys, group_df in df.groupby(grouping_cols, observed=False):
        if group_df.empty: continue
        model_name, sent_type, sent_len = group_keys
        group_size = len(group_df)
        if len(pd.unique(group_df['true_label'])) < 2: logger.debug(f"    Skipping metrics for group {group_keys}: Only one class present."); continue
        try:
            eval_results = evaluate_model(group_df['score'].tolist(), group_df['true_label'].tolist())
            if eval_results and "error" not in eval_results and 'metrics' in eval_results:
                metrics = eval_results['metrics']; row_data = { 'model_source': model_name, 'sentence_type': sent_type, 'sentence_length': sent_len, 'size': group_size, 'threshold_applied': eval_results.get('threshold_applied') }; row_data.update(metrics); detailed_metrics_list.append(row_data)
            else: logger.warning(f"  Skipping metrics for group {group_keys} due to evaluation error or missing metrics dict: {eval_results.get('error', 'N/A')}")
        except Exception as e: logger.warning(f"  Error evaluating group {group_keys}: {e}", exc_info=False)

    if not detailed_metrics_list: logger.error("No detailed metrics could be calculated. Aborting comparison plots and CSV generation."); return
    detailed_metrics_df = pd.DataFrame(detailed_metrics_list)
    try:
        detailed_metrics_df['model_source'] = pd.Categorical(detailed_metrics_df['model_source'], categories=df['model_source'].cat.categories, ordered=False)
        detailed_metrics_df['sentence_type'] = pd.Categorical(detailed_metrics_df['sentence_type'], categories=SENTENCE_TYPES_ORDER, ordered=True)
        detailed_metrics_df['sentence_length'] = pd.Categorical(detailed_metrics_df['sentence_length'], categories=SENTENCE_LENGTHS_ORDER, ordered=True)
    except Exception as e: logger.warning(f"Could not set categorical types on detailed_metrics_df: {e}")

    # --- Save Detailed CSV ---
    detailed_csv_path = os.path.join(csv_output_dir, "detailed_metrics_all_scenarios.csv")
    try: detailed_metrics_df.sort_values(by=['model_source', 'sentence_type', 'sentence_length']).to_csv(detailed_csv_path, index=False, float_format='%.5f'); logger.info(f"Saved detailed metrics CSV to: {detailed_csv_path}")
    except Exception as e: logger.error(f"Failed to save detailed metrics CSV: {e}", exc_info=True)

    # --- Calculate Overall Metrics per Model ---
    logger.info("Calculating overall metrics per model...")
    summary_metrics_list = []
    all_input_models = df['model_source'].unique().tolist()
    for model_name in all_input_models:
         group_df = df[df['model_source'] == model_name]
         if group_df.empty: all_models_overall_eval_results[model_name] = {"error": "No data"}; continue
         group_size = len(group_df)
         if len(pd.unique(group_df['true_label'])) < 2: logger.warning(f"Skipping overall metrics for model '{model_name}': Only one class present."); all_models_overall_eval_results[model_name] = {"error": "Single class"}; summary_metrics_list.append({'model_source': model_name, 'size': group_size, 'accuracy': np.nan}); continue
         try:
             eval_results = evaluate_model(group_df['score'].tolist(), group_df['true_label'].tolist())
             all_models_overall_eval_results[model_name] = eval_results
             if eval_results and "error" not in eval_results and 'metrics' in eval_results:
                 metrics = eval_results['metrics']; row_data = {'model_source': model_name, 'size': group_size, 'threshold_applied': eval_results.get('threshold_applied')}; row_data.update(metrics); summary_metrics_list.append(row_data)
                 if 'accuracy' not in row_data: row_data['accuracy'] = np.nan
             else: logger.warning(f"Could not get valid overall metrics for model {model_name}. Eval error: {eval_results.get('error', 'N/A')}"); all_models_overall_eval_results[model_name] = {"error": f"Eval error: {eval_results.get('error', 'N/A')}"}; summary_metrics_list.append({'model_source': model_name, 'size': group_size, 'accuracy': np.nan})
         except Exception as e: logger.warning(f"Error evaluating overall metrics for model {model_name}: {e}", exc_info=False); all_models_overall_eval_results[model_name] = {"error": f"Evaluation exception: {e}"}; summary_metrics_list.append({'model_source': model_name, 'size': 0, 'accuracy': np.nan})

    if not summary_metrics_list: logger.error("No models found or processed for overall metrics. Cannot generate comparison plots."); return
    summary_metrics_df = pd.DataFrame(summary_metrics_list).set_index('model_source')

    # --- Determine Model Order based on Overall Accuracy ---
    if 'accuracy' in summary_metrics_df.columns:
        sorted_models_df = summary_metrics_df.sort_values(by='accuracy', ascending=False, na_position='last'); sorted_model_names = sorted_models_df.index.tolist(); logger.info(f"Model order (by overall accuracy): {sorted_model_names}")
    else: logger.warning("Overall 'accuracy' metric not found in summary. Sorting models alphabetically."); sorted_model_names = sorted(summary_metrics_df.index.tolist())
    num_models = len(sorted_model_names)

    # --- Create Stable Color Map ---
    palette_name = 'viridis' if num_models > 10 else 'tab10'; 
    try: colors = sns.color_palette(palette_name, num_models)
    except ValueError: logger.warning(f"Could not generate {num_models} distinct colors from '{palette_name}'. Using fallback."); colors = sns.color_palette("husl", num_models)
    model_color_map = {model: colors[i] for i, model in enumerate(sorted_model_names)}; logger.debug(f"Model color map created: {model_color_map}")

    # --- Save Summary CSV ---
    summary_csv_path = os.path.join(csv_output_dir, "summary_metrics_per_model_sorted.csv")
    try: sorted_models_df.reset_index().to_csv(summary_csv_path, index=False, float_format='%.5f'); logger.info(f"Saved summary metrics per model CSV (sorted by accuracy) to: {summary_csv_path}")
    except Exception as e: logger.error(f"Failed to save sorted summary metrics CSV: {e}", exc_info=True)

    # --- Generate Comparison Plots (Now using Seaborn context) ---
    logger.info("Generating comparison plots with consistent order and colors...")
    # THEME AND CONTEXT SET HERE BEFORE PLOTS
    sns.set_theme(style="whitegrid")
    sns.set_context("talk") # <<< SET LARGER FONT CONTEXT
    logger.info("Seaborn context set to 'talk' for larger plot elements.")

    key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    # --- Plot 1: Overall ROC Curve Overlay ---
    try:
        fig_roc, ax_roc = plt.subplots(figsize=(12, 9)) # Adjusted size ok
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance')
        plot_count = 0
        for model_name in sorted_model_names:
            eval_results = all_models_overall_eval_results.get(model_name)
            if eval_results and "error" not in eval_results and eval_results.get('roc_curve') and 'metrics' in eval_results:
                roc_data = eval_results['roc_curve']; metrics = eval_results['metrics']; fpr, tpr = roc_data.get('fpr'), roc_data.get('tpr'); roc_auc = metrics.get('roc_auc')
                if fpr is not None and tpr is not None and roc_auc is not None and not np.isnan(roc_auc):
                    color = model_color_map.get(model_name, 'gray'); ax_roc.plot(fpr, tpr, lw=2.5, label=f'{model_name} (AUC={roc_auc:.3f})', color=color); plot_count += 1
        if plot_count > 0:
            ax_roc.set_xlim([0.0, 1.0]); ax_roc.set_ylim([0.0, 1.05]); ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate'); ax_roc.set_title('Overall Model ROC Comparison (Sorted by Accuracy)')
            ax_roc.legend(loc="lower right") # Context should handle size
            ax_roc.grid(True); plt.tight_layout(); save_path = os.path.join(plot_output_dir, "comparison_overall_roc_overlay_sorted.png")
            plt.savefig(save_path); logger.info(f"  Saved: Overall ROC Overlay plot to {save_path}")
        else: logger.warning("  Skipping ROC overlay plot: No valid ROC curves to plot.")
        plt.close(fig_roc)
    except Exception as e: logger.error(f"  Failed ROC Overlay Plot: {e}", exc_info=True)

    # --- Plot 2: Overall Accuracy Comparison (Bar Chart) ---
    plottable_summary_metrics = []
    if not summary_metrics_df.empty: plottable_summary_metrics = [m for m in key_metrics if m in summary_metrics_df.columns and not summary_metrics_df[m].isnull().all()]

    if 'accuracy' in summary_metrics_df.columns and not summary_metrics_df['accuracy'].isnull().all():
         try:
             plot_data = sorted_models_df['accuracy'].dropna()
             if not plot_data.empty:
                 fig, ax = plt.subplots(figsize=(max(8, len(plot_data) * 1.0), 7)) # Wider bars possible
                 colors_for_plot = [model_color_map.get(model, 'gray') for model in plot_data.index]
                 plot_data.plot(kind='bar', ax=ax, color=colors_for_plot)
                 ax.set_title('Overall Model Accuracy Comparison (Sorted)'); ax.set_ylabel('Accuracy'); ax.set_xlabel('Model Source')
                 ax.tick_params(axis='x', rotation=45) # Context handles size
                 ax.grid(axis='y', linestyle='--'); ax.set_ylim(bottom=0, top=max(1.0, plot_data.max() * 1.05) if not plot_data.empty else 1.0)
                 plt.tight_layout(); save_path = os.path.join(plot_output_dir, "comparison_overall_accuracy_sorted.png")
                 plt.savefig(save_path); logger.info(f"  Saved: Overall Accuracy plot to {save_path}")
                 plt.close(fig)
             else: logger.warning("  Skipping overall accuracy plot: No non-NaN accuracy values found.")
         except Exception as e: logger.error(f"  Failed Plot 2 (Accuracy Bar): {e}", exc_info=True)
    else: logger.warning("  Skipping overall accuracy plot: 'accuracy' metric missing or all NaN.")

    # --- Plot 3: Key Metrics Comparison (Grouped Bar Chart) ---
    if plottable_summary_metrics and not summary_metrics_df.empty:
         try:
             plot_df = summary_metrics_df.reindex(sorted_model_names)[plottable_summary_metrics].dropna(how='all', axis=0)
             if not plot_df.empty:
                 fig, ax = plt.subplots(figsize=(max(12, len(plot_df) * 1.5), 8)) # Wider bars possible
                 plot_df.plot(kind='bar', ax=ax, width=0.8)
                 ax.set_title('Overall Model Metrics Comparison (Sorted by Accuracy)'); ax.set_ylabel('Score'); ax.set_xlabel('Model Source')
                 ax.tick_params(axis='x', rotation=45) # Context handles size
                 ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left') # Context handles size
                 ax.grid(axis='y', linestyle='--'); ax.set_ylim(bottom=0); plt.tight_layout()
                 save_path = os.path.join(plot_output_dir, "comparison_overall_metrics_sorted.png")
                 plt.savefig(save_path); logger.info(f"  Saved: Overall Metrics plot to {save_path}")
                 plt.close(fig)
             else: logger.warning("  Skipping overall metrics plot: No models left after selecting metrics and dropping NaNs.")
         except Exception as e: logger.error(f"  Failed Plot 3 (Grouped Metrics Bar): {e}", exc_info=True)
    else: logger.warning("  Skipping overall metrics plot: No plottable key metrics or no summary data.")

    # --- Plot 4: Accuracy vs. Dataset Size (Scatter Plot) ---
    if 'accuracy' in summary_metrics_df.columns and 'size' in summary_metrics_df.columns and not summary_metrics_df.empty:
         try:
             plot_df_size = summary_metrics_df.reindex(sorted_model_names)[['accuracy', 'size']].dropna()
             if not plot_df_size.empty:
                  fig, ax = plt.subplots(figsize=(12, 8)) # Adjusted size ok
                  sns.scatterplot(data=plot_df_size, x='size', y='accuracy', ax=ax, s=150, # Larger points
                                  hue=plot_df_size.index, palette=model_color_map,
                                  hue_order=sorted_model_names, legend=False)
                  if len(plot_df_size) < 25: # Allow more labels
                      for model, row in plot_df_size.iterrows():
                          ax.text(row['size'] * 1.01, row['accuracy'], str(model),
                                  fontsize=FONT_SIZE_ANNOTATION, # Specific size for labels
                                  color=model_color_map.get(model, 'black'))
                  ax.set_title('Model Accuracy vs. Overall Dataset Size'); ax.set_ylabel('Accuracy'); ax.set_xlabel('Number of Samples (Overall)')
                  ax.grid(True, linestyle='--'); plt.tight_layout()
                  save_path = os.path.join(plot_output_dir, "comparison_accuracy_vs_size.png")
                  plt.savefig(save_path); logger.info(f"  Saved: Accuracy vs Size plot to {save_path}")
                  plt.close(fig)
             else: logger.warning("  Skipping Accuracy vs Size plot: No valid (accuracy, size) pairs.")
         except Exception as e: logger.error(f"  Failed Plot 4 (Accuracy vs Size): {e}", exc_info=True)
    else: logger.warning("  Skipping Accuracy vs Size plot: Missing 'accuracy' or 'size' or no summary data.")

    # --- Plot 5a & 5b: Accuracy by Model and Type/Length (Bar Charts) ---
    if not detailed_metrics_df.empty and 'accuracy' in detailed_metrics_df.columns:
        try: # Plot 5a (by Type)
             fig, ax = plt.subplots(figsize=(max(12, num_models * 1.2), 8)) # Wider bars
             sns.barplot(data=detailed_metrics_df, x='model_source', y='accuracy', hue='sentence_type', ax=ax, palette='tab10', order=sorted_model_names, hue_order=SENTENCE_TYPES_ORDER, ci=None)
             ax.set_title('Accuracy by Model and Sentence Type (Models Sorted by Overall Accuracy)'); ax.set_ylabel('Accuracy'); ax.set_xlabel('Model Source')
             ax.tick_params(axis='x', rotation=45) # Context handles size
             ax.legend(title='Sentence Type', bbox_to_anchor=(1.02, 1), loc='upper left') # Context handles size
             ax.grid(axis='y', linestyle='--'); ax.set_ylim(bottom=0); plt.tight_layout()
             save_path = os.path.join(plot_output_dir, "comparison_accuracy_by_model_type_sorted.png")
             plt.savefig(save_path); logger.info(f"  Saved: Accuracy by Model/Type plot to {save_path}")
             plt.close(fig)
        except Exception as e: logger.error(f"  Failed Plot 5a (Accuracy by Type): {e}", exc_info=True)
        try: # Plot 5b (by Length)
             fig, ax = plt.subplots(figsize=(max(12, num_models * 1.2), 8)) # Wider bars
             sns.barplot(data=detailed_metrics_df, x='model_source', y='accuracy', hue='sentence_length', ax=ax, palette='rocket', order=sorted_model_names, hue_order=SENTENCE_LENGTHS_ORDER, ci=None)
             ax.set_title('Accuracy by Model and Sentence Length (Models Sorted by Overall Accuracy)'); ax.set_ylabel('Accuracy'); ax.set_xlabel('Model Source')
             ax.tick_params(axis='x', rotation=45) # Context handles size
             ax.legend(title='Sentence Length', bbox_to_anchor=(1.02, 1), loc='upper left') # Context handles size
             ax.grid(axis='y', linestyle='--'); ax.set_ylim(bottom=0); plt.tight_layout()
             save_path = os.path.join(plot_output_dir, "comparison_accuracy_by_model_length_sorted.png")
             plt.savefig(save_path); logger.info(f"  Saved: Accuracy by Model/Length plot to {save_path}")
             plt.close(fig)
        except Exception as e: logger.error(f"  Failed Plot 5b (Accuracy by Length): {e}", exc_info=True)
    else: logger.warning("  Skipping detailed accuracy breakdown plots: 'accuracy' metric missing or no detailed data.")

    logger.info("--- Comparison Plot and CSV Generation Complete ---")


# --- Main Execution ---
if __name__ == "__main__":

    logger.info("Starting main execution block.")

    # Set Seaborn theme and context *once* globally before plotting
    try:
        sns.set_theme(style="whitegrid")
        sns.set_context("talk") # Options: paper, notebook, talk, poster
        logger.info("Set global Seaborn theme to 'whitegrid' and context to 'talk'.")
    except Exception as e:
        logger.warning(f"Could not set global Seaborn theme/context: {e}")


    # 1. Load Data
    master_df = load_data_from_json(INPUT_DIR)

    if master_df is not None and not master_df.empty:
        # 2. Generate Individual Evaluation Plots
        generate_comparison_plots_and_csv(master_df, COMPARISON_PLOT_DIR, CSV_OUTPUT_DIR)
        # These will now also use the 'talk' context set above
        generate_individual_plots(master_df, INDIVIDUAL_PLOT_DIR)

        # 3. Generate Comparison Plots and Metrics CSVs
        # The context is already set, no need to set it again inside the function
    else:
        logger.error("Execution stopped: Failed to load valid data or data is empty.")

    logger.info("--- Analysis Script Finished ---")
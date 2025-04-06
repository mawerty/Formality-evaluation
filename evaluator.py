import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)
from typing import List, Dict, Any, Tuple, Optional

from config import POSITIVE_LABEL

def calculate_metrics(
    y_true: List[int],
    y_pred_binary: List[int],
    y_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """Calculates standard classification metrics."""
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred_binary)
    metrics["precision"] = precision_score(y_true, y_pred_binary, pos_label=POSITIVE_LABEL, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred_binary, pos_label=POSITIVE_LABEL, zero_division=0)
    metrics["f1_score"] = f1_score(y_true, y_pred_binary, pos_label=POSITIVE_LABEL, zero_division=0)

    # Calculate AUC only if scores are provided and meaningful
    if y_scores is not None and len(np.unique(y_true)) > 1 and len(np.unique(y_scores)) > 1:
         # Ensure scores are passed correctly for roc_auc_score
         metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
    else:
        metrics["roc_auc"] = np.nan

    return metrics

def find_optimal_threshold(
    y_true: List[int],
    scores: List[float],
    strategy: str = "maximize_j" # e.g., 'maximize_f1', 'maximize_j'
) -> Tuple[float, float]:
    """
    Finds an 'optimal' threshold based on a chosen strategy using the provided data.

    Args:
        y_true: Ground truth labels.
        scores: Model output scores for the positive class.
        strategy: Method to use ('maximize_f1', 'maximize_j').

    Returns:
        A tuple containing (best_threshold, best_metric_value).

    Note: Optimizing threshold on the test set can lead to inflated performance metrics.
          Ideally, use a separate validation set.
    """
    if len(np.unique(y_true)) < 2 or len(np.unique(scores)) < 2:
        print("Warning: Cannot find optimal threshold (single class or no score variance). Returning default 0.5.")
        return 0.5, np.nan

    scores_arr = np.array(scores)
    y_true_arr = np.array(y_true)

    best_threshold = 0.5
    best_metric_value = -np.inf

    if strategy == "maximize_f1":
        precision, recall, thresholds = precision_recall_curve(y_true_arr, scores_arr, pos_label=POSITIVE_LABEL)
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9) 
        valid_indices = ~np.isnan(f1_scores) # Handle potential NaNs if precision+recall is 0
        if np.any(valid_indices):
            best_idx = np.argmax(f1_scores[valid_indices])
            best_threshold = thresholds[valid_indices][best_idx]
            best_metric_value = f1_scores[valid_indices][best_idx]
        else:
            print("Warning: Could not compute valid F1 scores for threshold optimization. Returning default 0.5.")
            best_threshold = 0.5
            best_metric_value = np.nan


    elif strategy == "maximize_j":
        fpr, tpr, thresholds = roc_curve(y_true_arr, scores_arr, pos_label=POSITIVE_LABEL)
        youden_j = tpr - fpr
        valid_indices = ~np.isnan(youden_j) # Should generally not be NaN here, but good practice
        if np.any(valid_indices):
            best_idx = np.argmax(youden_j[valid_indices])

            best_threshold = thresholds[valid_indices][best_idx]
            best_metric_value = youden_j[valid_indices][best_idx]

            if best_threshold > np.max(scores_arr):
                 best_threshold = np.max(scores_arr)

        else:
            print("Warning: Could not compute valid Youden's J scores for threshold optimization. Returning default 0.5.")
            best_threshold = 0.5
            best_metric_value = np.nan

    else:
        raise ValueError(f"Unknown threshold optimization strategy: {strategy}")

    print(f"Optimal threshold found using '{strategy}': {best_threshold:.4f} (Metric Value: {best_metric_value:.4f})")
    return best_threshold, best_metric_value


def apply_threshold(
    scores: List[float],
    threshold: float
) -> List[int]:
    """Applies a fixed threshold to scores to get binary predictions."""
    scores_arr = np.array(scores)
    binary_predictions = (scores_arr >= threshold).astype(int).tolist()
    return binary_predictions

def evaluate_model(
    scores: List[float],
    true_labels: List[int],
    threshold_strategy: str = "maximize_j", # e.g., 'fixed', 'maximize_f1', 'maximize_j'
    fixed_threshold_value: float = 0.5, # Only used if threshold_strategy is 'fixed'
) -> Dict[str, Any]:
    """
    Evaluates a model, returning metrics, scores, predictions etc.
    Allows finding an optimal threshold or using a fixed one.
    """
    print(f"Starting evaluation with threshold strategy: {threshold_strategy}")
    results = {}
    results["scores"] = np.array(scores)
    results["true_labels"] = np.array(true_labels)

    if not scores or len(scores) != len(true_labels):
        print(f"Error: Scores list is empty or length mismatch with true_labels.")
        return {"error": "Input data error"}

    scores_arr = np.array(scores)
    true_labels_arr = np.array(true_labels)

    actual_threshold = fixed_threshold_value
    threshold_type_used = "fixed"

    if threshold_strategy == "fixed":
        actual_threshold = fixed_threshold_value
        print(f"Using pre-defined fixed threshold: {actual_threshold:.4f}")
        threshold_type_used = f"fixed ({fixed_threshold_value})"

    elif threshold_strategy in ["maximize_f1", "maximize_j"]:
        optimal_threshold, _ = find_optimal_threshold(true_labels_arr, scores_arr, strategy=threshold_strategy)
        actual_threshold = optimal_threshold
        threshold_type_used = f"optimized ({threshold_strategy})"
        print(f"Using data-driven threshold ({threshold_strategy}): {actual_threshold:.4f}")

    else:
         raise ValueError(f"Unknown threshold_strategy: {threshold_strategy}")


    binary_predictions = apply_threshold(scores_arr, actual_threshold)
    results["threshold_strategy_requested"] = threshold_strategy
    results["threshold_applied"] = actual_threshold
    results["threshold_type_used"] = threshold_type_used 
    results["binary_predictions"] = np.array(binary_predictions)

    results["metrics"] = calculate_metrics(true_labels_arr, results["binary_predictions"], scores_arr)

    # Calculate data for curves if possible
    results["roc_curve"] = None
    results["pr_curve"] = None
    if len(np.unique(true_labels_arr)) > 1 and len(np.unique(scores_arr)) > 1:
        fpr, tpr, roc_thresholds = roc_curve(true_labels_arr, scores_arr, pos_label=POSITIVE_LABEL)
        results["roc_curve"] = {"fpr": fpr, "tpr": tpr, "thresholds": roc_thresholds}

        precision, recall, pr_thresholds = precision_recall_curve(true_labels_arr, scores_arr, pos_label=POSITIVE_LABEL)
        results["pr_curve"] = {"precision": precision, "recall": recall, "thresholds": pr_thresholds}
    else:
        print("Warning: Cannot compute ROC/PR curves (single class or no score variance).")


    # Calculate confusion matrix
    # Ensure labels=[0, 1] matches your actual label encoding
    results["confusion_matrix"] = confusion_matrix(true_labels_arr, results["binary_predictions"], labels=[0, POSITIVE_LABEL])

    # Add the optimal F1/J value if calculated
    if threshold_strategy in ["maximize_f1", "maximize_j"]:
         _, metric_val = find_optimal_threshold(true_labels_arr, scores_arr, strategy=threshold_strategy)
         results[f"optimized_{threshold_strategy}_value"] = metric_val


    print(f"Evaluation complete. Final metrics (at threshold {actual_threshold:.4f}): {results['metrics']}")
    return results

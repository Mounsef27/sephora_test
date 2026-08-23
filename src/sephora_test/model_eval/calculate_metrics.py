import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_all_metrics(
    dataset_name: str,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Calculate normalized classification metrics.

    Args:
        dataset_name (str):
            Name of the evaluated dataset, for example "train" or "test".

        y_pred (np.ndarray):
            Predicted class labels.

        y_true (np.ndarray):
            Ground-truth class labels.

    Returns:
        dict:
            Dictionary containing the dataset name and its classification
            metrics, including accuracy, balanced accuracy, F1 scores,
            precision, recall, normalized confusion matrix, and
            classification report.
    """
    metrics = {
        dataset_name: {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "precision_macro": precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
                normalize="true",
            ),
            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
            ),
        }
    }

    return metrics

import pickle
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def create_artifact_directory(
    base_directory: str,
    run_datetime: str,
) -> Path:
    """Create the artifact directory for an experiment run.

    Args:
        base_directory (str):
            Base directory where experiment artifacts are stored.

        run_datetime (str):
            Datetime identifier of the experiment run.

    Returns:
        Path:
            Path to the created artifact directory.
    """
    artifact_directory = Path(base_directory) / run_datetime

    artifact_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return artifact_directory


def save_results_pickle(
    results: Dict,
    output_path: str | Path,
) -> None:
    """Save experiment results into a pickle file.

    Args:
        results (Dict):
            Dictionary containing experiment results.

        output_path (str | Path):
            Path of the output pickle file.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("wb") as file:
        pickle.dump(
            results,
            file,
        )


def save_test_predictions_csv(
    results: Dict,
    first_name: str,
    last_name: str,
    output_directory: str | Path,
) -> Path:
    """Save test predictions in the required CSV format.

    Args:
        results (Dict):
            Dictionary containing experiment results.

        first_name (str):
            Candidate first name.

        last_name (str):
            Candidate last name.

        output_directory (str | Path):
            Directory where the prediction CSV will be saved.

    Returns:
        Path:
            Path to the generated prediction CSV file.
    """
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_processed = results["test_processed"].copy()

    predictions = test_processed[
        [
            "productId",
            "nature_code_predicted",
            "target_code_predicted",
        ]
    ].copy()

    predictions = predictions.rename(
        columns={
            "nature_code_predicted": "nature_predict",
            "target_code_predicted": "target_predict",
        }
    )

    output_path = (
        output_directory
        / f"{first_name.upper()}_{last_name.upper()}_prediction.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_train_metrics_txt(
    metrics: Dict,
    output_path: str | Path,
) -> None:
    """Save training metrics into a text file.

    Args:
        metrics (Dict):
            Dictionary containing training metrics.

        output_path (str | Path):
            Path of the output text file.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for dataset_name, dataset_metrics in metrics.items():
            file.write(f"===== {dataset_name} =====\n\n")

            for metric_name, metric_value in dataset_metrics.items():
                if metric_name == "confusion_matrix":
                    continue

                if metric_name == "classification_report":
                    file.write(f"{metric_name}:\n{metric_value}\n\n")
                else:
                    file.write(f"{metric_name}: {metric_value}\n")


def save_confusion_matrix_image(
    confusion_matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    """Save a confusion matrix as an image.

    Args:
        confusion_matrix (np.ndarray):
            Confusion matrix to plot.

        labels (list[str]):
            Class labels.

        output_path (str | Path):
            Path of the output image.

        title (str):
            Figure title.
    """
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 8),
    )

    image = ax.imshow(
        confusion_matrix,
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticklabels(
        labels,
    )

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def assign_train_validation_split(
    data: DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DataFrame:
    data = data.copy()

    stratify_column = (
        data["nature_code"].astype(str) + "_" + data["target_code"].astype(str)
    )

    train_indices, validation_indices = train_test_split(
        data.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_column,
    )

    data["sample"] = "train"

    data.loc[
        validation_indices,
        "sample",
    ] = "validation"

    return data

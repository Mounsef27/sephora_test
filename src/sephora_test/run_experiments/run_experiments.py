"""Run the full Sephora experiment pipeline."""

from datetime import datetime
from typing import Dict

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from sephora_test.apply_transformation.clean_text import clean_text
from sephora_test.apply_transformation.label_construction import (
    LabelConstructor,
)
from sephora_test.config.labels_map import (
    mapping_nature_code,
    mapping_target_code,
)
from sephora_test.logger import setup_logger
from sephora_test.model_eval.calculate_metrics import calculate_all_metrics
from sephora_test.utils.utils import (
    create_artifact_directory,
    save_confusion_matrix_image,
    save_results_pickle,
    save_test_predictions_csv,
    save_train_metrics_txt,
)

logger = setup_logger(__name__)


config: Dict = {
    "first_name": "Mounsef",
    "last_name": "Debache",
    "method_name": "keyword_label_constructor",
    "run_datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "data_paths": {
        "training_data": (
            "/home/ubuntu/Desktop/sephora_test/data/training_set.csv"
        ),
        "test_data": ("/home/ubuntu/Desktop/sephora_test/data/testing_set.csv"),
    },
    "artifacts_path": ("/home/ubuntu/Desktop/sephora_test/src/artifacts"),
}


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean product name and description columns.

    Args:
        df (pd.DataFrame):
            Input product dataset.

    Returns:
        pd.DataFrame:
            Dataset containing cleaned product name and description columns.
    """
    logger.info(
        "Starting text preprocessing for dataframe with shape %s.",
        df.shape,
    )

    df = df.copy()

    df["cleanName"] = df["productName"].fillna("").apply(clean_text)

    df["cleanDescription"] = df["longDescription"].fillna("").apply(clean_text)

    logger.info("Text preprocessing completed.")

    return df


def build_pipeline() -> Pipeline:
    """Build the sklearn pipeline used for the experiment.

    Returns:
        Pipeline:
            Pipeline containing text preprocessing, nature code construction,
            and target code construction.
    """
    logger.info("Building sklearn pipeline.")

    pipeline = Pipeline(
        steps=[
            (
                "text_cleaning",
                FunctionTransformer(
                    preprocess_dataframe,
                    validate=False,
                ),
            ),
            (
                "nature_constructor",
                LabelConstructor(
                    label_map=mapping_nature_code,
                    columns_list_to_use=[
                        "cleanName",
                        "cleanDescription",
                    ],
                    output_column_name="nature_code_predicted",
                ),
            ),
            (
                "target_constructor",
                LabelConstructor(
                    label_map=mapping_target_code,
                    columns_list_to_use=[
                        "cleanName",
                        "cleanDescription",
                    ],
                    output_column_name="target_code_predicted",
                ),
            ),
        ]
    )

    logger.info("Sklearn pipeline successfully built.")

    return pipeline


def run_experiments(
    experiment_config: Dict,
) -> Dict:
    """Run the complete Sephora experiment.

    Args:
        experiment_config (Dict):
            Experiment configuration containing metadata and data paths.

    Returns:
        Dict:
            Dictionary containing processed datasets and evaluation metrics.
    """
    logger.info("Starting Sephora experiment.")

    logger.info(
        "Method: %s",
        experiment_config["method_name"],
    )

    logger.info(
        "Run datetime: %s",
        experiment_config["run_datetime"],
    )

    logger.info("Loading training dataset.")

    train_df = pd.read_csv(experiment_config["data_paths"]["training_data"])

    logger.info(
        "Training dataset loaded with shape %s.",
        train_df.shape,
    )

    logger.info("Loading testing dataset.")

    test_df = pd.read_csv(experiment_config["data_paths"]["test_data"])

    logger.info(
        "Testing dataset loaded with shape %s.",
        test_df.shape,
    )

    pipeline = build_pipeline()

    logger.info("Fitting pipeline on training dataset.")

    pipeline.fit(train_df)

    logger.info("Running pipeline on training dataset.")

    train_processed = pipeline.transform(train_df)

    logger.info(
        "Training pipeline completed with shape %s.",
        train_processed.shape,
    )

    logger.info("Running pipeline on testing dataset.")

    test_processed = pipeline.transform(test_df)

    logger.info(
        "Testing pipeline completed with shape %s.",
        test_processed.shape,
    )

    logger.info("Evaluating nature_code predictions.")

    nature_eval_df = train_processed[
        [
            "nature_code",
            "nature_code_predicted",
        ]
    ].dropna()

    nature_metrics = calculate_all_metrics(
        dataset_name="train_nature",
        y_true=nature_eval_df["nature_code"].to_numpy(),
        y_pred=nature_eval_df["nature_code_predicted"].to_numpy(),
    )

    logger.info(
        "Nature code accuracy: %.4f",
        nature_metrics["train_nature"]["accuracy"],
    )

    logger.info(
        "Nature code balanced accuracy: %.4f",
        nature_metrics["train_nature"]["balanced_accuracy"],
    )

    logger.info(
        "Nature code macro F1: %.4f",
        nature_metrics["train_nature"]["f1_macro"],
    )

    logger.info(
        "Nature code weighted F1: %.4f",
        nature_metrics["train_nature"]["f1_weighted"],
    )

    logger.info(
        "Nature code classification report:\n%s",
        nature_metrics["train_nature"]["classification_report"],
    )

    logger.info(
        "Nature code normalized confusion matrix:\n%s",
        nature_metrics["train_nature"]["confusion_matrix"],
    )

    logger.info("Evaluating target_code predictions.")

    target_eval_df = train_processed[
        [
            "target_code",
            "target_code_predicted",
        ]
    ].dropna()

    target_metrics = calculate_all_metrics(
        dataset_name="train_target",
        y_true=target_eval_df["target_code"].to_numpy(),
        y_pred=target_eval_df["target_code_predicted"].to_numpy(),
    )

    logger.info(
        "Target code accuracy: %.4f",
        target_metrics["train_target"]["accuracy"],
    )

    logger.info(
        "Target code balanced accuracy: %.4f",
        target_metrics["train_target"]["balanced_accuracy"],
    )

    logger.info(
        "Target code macro F1: %.4f",
        target_metrics["train_target"]["f1_macro"],
    )

    logger.info(
        "Target code weighted F1: %.4f",
        target_metrics["train_target"]["f1_weighted"],
    )

    logger.info(
        "Target code classification report:\n%s",
        target_metrics["train_target"]["classification_report"],
    )

    logger.info(
        "Target code normalized confusion matrix:\n%s",
        target_metrics["train_target"]["confusion_matrix"],
    )

    results = {
        "config": experiment_config,
        "pipeline": pipeline,
        "train_processed": train_processed,
        "test_processed": test_processed,
        "nature_metrics": nature_metrics,
        "target_metrics": target_metrics,
    }

    logger.info("Creating artifact directory.")

    artifact_directory = create_artifact_directory(
        base_directory=experiment_config["artifacts_path"],
        run_datetime=experiment_config["run_datetime"],
    )

    logger.info(
        "Artifact directory created at %s.",
        artifact_directory,
    )

    logger.info("Saving complete experiment results.")

    save_results_pickle(
        results=results,
        output_path=(artifact_directory / "results.pkl"),
    )

    logger.info("Saving test predictions CSV.")

    save_test_predictions_csv(
        results=results,
        first_name=experiment_config["first_name"],
        last_name=experiment_config["last_name"],
        output_directory=artifact_directory,
    )

    logger.info("Saving nature training metrics.")

    save_train_metrics_txt(
        metrics=results["nature_metrics"],
        output_path=(artifact_directory / "nature_metrics.txt"),
    )

    logger.info("Saving target training metrics.")

    save_train_metrics_txt(
        metrics=results["target_metrics"],
        output_path=(artifact_directory / "target_metrics.txt"),
    )

    nature_labels = sorted(nature_eval_df["nature_code"].dropna().unique())

    target_labels = sorted(target_eval_df["target_code"].dropna().unique())

    logger.info("Saving nature confusion matrix image.")

    save_confusion_matrix_image(
        confusion_matrix=(
            results["nature_metrics"]["train_nature"]["confusion_matrix"]
        ),
        labels=list(nature_labels),
        output_path=(artifact_directory / "nature_confusion_matrix.png"),
        title="Nature Code - Confusion Matrix",
    )

    logger.info("Saving target confusion matrix image.")

    save_confusion_matrix_image(
        confusion_matrix=(
            results["target_metrics"]["train_target"]["confusion_matrix"]
        ),
        labels=list(target_labels),
        output_path=(artifact_directory / "target_confusion_matrix.png"),
        title="Target Code - Confusion Matrix",
    )

    logger.info(
        "All experiment artifacts successfully saved in %s.",
        artifact_directory,
    )

    logger.info("Experiment completed successfully.")

    return results


def main() -> None:
    """Run the Sephora experiment."""
    run_experiments(
        experiment_config=config,
    )


if __name__ == "__main__":
    main()

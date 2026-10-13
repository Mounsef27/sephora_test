from datetime import datetime
from typing import Dict

from pandas import DataFrame

from sephora_test.logger import setup_logger
from sephora_test.model_eval.calculate_metrics import (
    calculate_all_metrics,
)
from sephora_test.run_experiments.build_ml_pipeline import (
    build_ml_pipeline,
)
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
    "method_name": "tfidf_linear_svc",
    "run_datetime": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    "data_paths": {
        "training_data": (
            "/home/ubuntu/Desktop/sephora_test/data/training_set.csv"
        ),
        "test_data": ("/home/ubuntu/Desktop/sephora_test/data/testing_set.csv"),
    },
    "artifacts_path": ("/home/ubuntu/Desktop/sephora_test/src/artifacts"),
    "validation_size": 0.2,
    "random_state": 42,
    "nature_model": {
        "class_weight": None,
    },
    "target_model": {
        "class_weight": "balanced",
    },
}


def run_ml_pipeline(
    training_data: DataFrame,
    testing_data: DataFrame,
    experiment_config: Dict,
) -> Dict:
    """Run the TF-IDF and LinearSVC experiment pipeline."""

    logger.info("Starting ML experiment.")

    logger.info(
        "Method: %s",
        experiment_config["method_name"],
    )

    logger.info(
        "Run datetime: %s",
        experiment_config["run_datetime"],
    )

    logger.info(
        "Training dataset shape: %s.",
        training_data.shape,
    )

    logger.info(
        "Testing dataset shape: %s.",
        testing_data.shape,
    )

    logger.info("Creating training and validation datasets.")

    train_data = training_data[training_data["sample"] == "train"].copy()

    validation_data = training_data[
        training_data["sample"] == "validation"
    ].copy()

    testing_data = testing_data.copy()

    logger.info(
        "Training sample shape: %s.",
        train_data.shape,
    )

    logger.info(
        "Validation sample shape: %s.",
        validation_data.shape,
    )

    logger.info("Building nature classification pipeline.")

    nature_pipeline = build_ml_pipeline(
        class_weight=experiment_config["nature_model"]["class_weight"],
    )

    logger.info("Building target classification pipeline.")

    target_pipeline = build_ml_pipeline(
        class_weight=experiment_config["target_model"]["class_weight"],
    )

    logger.info("Fitting nature classification pipeline.")

    nature_pipeline.fit(
        train_data,
        train_data["nature_code"],
    )

    logger.info("Nature classification pipeline fitted successfully.")

    logger.info("Fitting target classification pipeline.")

    target_pipeline.fit(
        train_data,
        train_data["target_code"],
    )

    logger.info("Target classification pipeline fitted successfully.")

    logger.info("Running nature predictions on validation dataset.")

    validation_data["nature_code_predicted"] = nature_pipeline.predict(
        validation_data
    )

    logger.info("Running target predictions on validation dataset.")

    validation_data["target_code_predicted"] = target_pipeline.predict(
        validation_data
    )

    logger.info("Calculating nature validation metrics.")

    nature_metrics = calculate_all_metrics(
        dataset_name="nature_validation",
        y_true=validation_data["nature_code"],
        y_pred=validation_data["nature_code_predicted"],
    )

    logger.info(
        "Nature validation accuracy: %.4f",
        nature_metrics["nature_validation"]["accuracy"],
    )

    logger.info(
        "Nature validation balanced accuracy: %.4f",
        nature_metrics["nature_validation"]["balanced_accuracy"],
    )

    logger.info(
        "Nature validation macro F1: %.4f",
        nature_metrics["nature_validation"]["f1_macro"],
    )

    logger.info("Calculating target validation metrics.")

    target_metrics = calculate_all_metrics(
        dataset_name="target_validation",
        y_true=validation_data["target_code"],
        y_pred=validation_data["target_code_predicted"],
    )

    logger.info(
        "Target validation accuracy: %.4f",
        target_metrics["target_validation"]["accuracy"],
    )

    logger.info(
        "Target validation balanced accuracy: %.4f",
        target_metrics["target_validation"]["balanced_accuracy"],
    )

    logger.info(
        "Target validation macro F1: %.4f",
        target_metrics["target_validation"]["f1_macro"],
    )

    logger.info("Running inference on testing dataset.")

    testing_data["nature_code_predicted"] = nature_pipeline.predict(
        testing_data
    )

    testing_data["target_code_predicted"] = target_pipeline.predict(
        testing_data
    )

    logger.info("Testing dataset inference completed successfully.")

    results = {
        "config": experiment_config,
        "nature_pipeline": nature_pipeline,
        "target_pipeline": target_pipeline,
        "train_data": train_data,
        "validation_data": validation_data,
        "test_processed": testing_data,
        "nature_metrics": nature_metrics,
        "target_metrics": target_metrics,
    }

    logger.info("Creating artifact directory.")

    artifact_directory = create_artifact_directory(
        base_directory=experiment_config["artifacts_path"],
        run_datetime=experiment_config["run_datetime"],
    )

    logger.info(
        "Artifact directory: %s.",
        artifact_directory,
    )

    method_name = experiment_config["method_name"]

    logger.info("Saving complete experiment pickle.")

    pickle_path = artifact_directory / f"{method_name}_results.pkl"

    save_results_pickle(
        results=results,
        output_path=pickle_path,
    )

    logger.info(
        "Experiment pickle saved to %s.",
        pickle_path,
    )

    logger.info("Saving nature validation metrics.")

    save_train_metrics_txt(
        metrics=nature_metrics,
        output_path=(artifact_directory / f"{method_name}_nature_metrics.txt"),
    )

    logger.info("Saving target validation metrics.")

    save_train_metrics_txt(
        metrics=target_metrics,
        output_path=(artifact_directory / f"{method_name}_target_metrics.txt"),
    )

    nature_labels = sorted(validation_data["nature_code"].dropna().unique())

    target_labels = sorted(validation_data["target_code"].dropna().unique())

    logger.info("Saving nature confusion matrix.")

    save_confusion_matrix_image(
        confusion_matrix=(
            nature_metrics["nature_validation"]["confusion_matrix"]
        ),
        labels=list(nature_labels),
        output_path=(
            artifact_directory / f"{method_name}_nature_confusion_matrix.png"
        ),
        title=("Nature Code - Validation Confusion Matrix"),
    )

    logger.info("Saving target confusion matrix.")

    save_confusion_matrix_image(
        confusion_matrix=(
            target_metrics["target_validation"]["confusion_matrix"]
        ),
        labels=list(target_labels),
        output_path=(
            artifact_directory / f"{method_name}_target_confusion_matrix.png"
        ),
        title=("Target Code - Validation Confusion Matrix"),
    )

    logger.info("Saving submission file.")

    submission_path = save_test_predictions_csv(
        results=results,
        first_name=experiment_config["first_name"],
        last_name=experiment_config["last_name"],
        output_directory=artifact_directory,
    )

    logger.info(
        "Submission file saved to %s.",
        submission_path,
    )

    logger.info("All ML artifacts successfully saved.")

    logger.info("ML experiment completed successfully.")

    return results

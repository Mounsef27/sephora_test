import pickle  # nosec B403

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from sephora_test.utils.utils import (
    assign_train_validation_split,
    create_artifact_directory,
    save_confusion_matrix_image,
    save_results_pickle,
    save_test_predictions_csv,
    save_train_metrics_txt,
)


class TestUtils:
    def test_create_artifact_directory(self, tmp_path):
        base_directory = tmp_path / "artifacts"

        result = create_artifact_directory(
            base_directory=str(base_directory),
            run_datetime="run_001",
        )

        assert result == base_directory / "run_001"
        assert result.is_dir()

    def test_create_artifact_directory_already_exists(self, tmp_path):
        expected_path = tmp_path / "run_001"
        expected_path.mkdir()

        result = create_artifact_directory(
            base_directory=str(tmp_path),
            run_datetime="run_001",
        )

        assert result == expected_path
        assert result.is_dir()

    def test_save_results_pickle(self, tmp_path):
        results = {
            "accuracy": 0.85,
            "predictions": ["A", "B"],
        }
        output_path = tmp_path / "artifacts" / "results.pkl"

        save_results_pickle(
            results=results,
            output_path=output_path,
        )

        assert output_path.is_file()

        # Load only the file created by this test.
        with output_path.open("rb") as file:
            saved_results = pickle.load(file)  # nosec B301

        assert saved_results == results

    def test_save_test_predictions_csv(self, tmp_path):
        original = pd.DataFrame({
            "productId": [101, 102],
            "nature_code_predicted": ["A", "B"],
            "target_code_predicted": ["X", "Y"],
            "extra_column": [1, 2],
        })
        results = {"test_processed": original.copy(deep=True)}
        output_directory = tmp_path / "predictions"

        output_path = save_test_predictions_csv(
            results=results,
            first_name="Mounsef",
            last_name="Debache",
            output_directory=output_directory,
        )

        assert output_path == (
            output_directory / "MOUNSEF_DEBACHE_prediction.csv"
        )
        assert output_path.is_file()

        saved_predictions = pd.read_csv(output_path)
        expected_predictions = pd.DataFrame({
            "productId": [101, 102],
            "nature_predict": ["A", "B"],
            "target_predict": ["X", "Y"],
        })

        pd.testing.assert_frame_equal(
            saved_predictions,
            expected_predictions,
        )
        pd.testing.assert_frame_equal(
            results["test_processed"],
            original,
        )

    def test_save_train_metrics_txt(self, tmp_path):
        metrics = {
            "train": {
                "accuracy": 0.95,
                "classification_report": "Train report",
                "confusion_matrix": np.array([[9, 1], [0, 10]]),
            },
            "validation": {
                "accuracy": 0.85,
                "classification_report": "Validation report",
                "confusion_matrix": np.array([[8, 2], [1, 9]]),
            },
        }
        output_path = tmp_path / "metrics" / "metrics.txt"

        save_train_metrics_txt(
            metrics=metrics,
            output_path=output_path,
        )

        assert output_path.is_file()

        content = output_path.read_text(encoding="utf-8")

        assert "===== train =====" in content
        assert "===== validation =====" in content
        assert "accuracy: 0.95" in content
        assert "accuracy: 0.85" in content
        assert "classification_report:\nTrain report" in content
        assert "classification_report:\nValidation report" in content
        assert "confusion_matrix" not in content

    def test_save_confusion_matrix_image(self, tmp_path):
        confusion_matrix = np.array([[8, 2], [1, 9]])
        output_path = tmp_path / "plots" / "confusion_matrix.png"
        figures_before = set(plt.get_fignums())

        save_confusion_matrix_image(
            confusion_matrix=confusion_matrix,
            labels=["A", "B"],
            output_path=output_path,
            title="Validation confusion matrix",
        )

        assert output_path.is_file()

        image = plt.imread(output_path)

        assert image.shape[0] > 0
        assert image.shape[1] > 0
        assert set(plt.get_fignums()) == figures_before

    @pytest.fixture
    def split_data(self):
        # Four class combinations with ten examples each.
        return pd.DataFrame(
            {
                "productId": range(40),
                "nature_code": np.repeat(["A", "A", "B", "B"], 10),
                "target_code": np.repeat(["X", "Y", "X", "Y"], 10),
            },
            index=range(100, 140),
        )

    def test_assign_train_validation_split(self, split_data):
        result = assign_train_validation_split(
            data=split_data,
            test_size=0.2,
            random_state=42,
        )

        train = result[result["sample"] == "train"]
        validation = result[result["sample"] == "validation"]

        assert len(train) == 32
        assert len(validation) == 8
        assert set(result["sample"]) == {"train", "validation"}
        assert train.index.intersection(validation.index).empty
        assert result.index.equals(split_data.index)

        pd.testing.assert_frame_equal(
            result.drop(columns="sample"),
            split_data,
        )

    def test_assign_train_validation_split_is_reproducible(
        self,
        split_data,
    ):
        first_result = assign_train_validation_split(
            data=split_data,
            random_state=42,
        )
        second_result = assign_train_validation_split(
            data=split_data,
            random_state=42,
        )

        pd.testing.assert_frame_equal(first_result, second_result)

    def test_assign_train_validation_split_preserves_original(
        self,
        split_data,
    ):
        original = split_data.copy(deep=True)

        assign_train_validation_split(data=split_data)

        pd.testing.assert_frame_equal(split_data, original)

    def test_assign_train_validation_split_is_stratified(
        self,
        split_data,
    ):
        result = assign_train_validation_split(
            data=split_data,
            test_size=0.2,
            random_state=42,
        )

        validation = result[result["sample"] == "validation"]
        counts = validation.groupby(["nature_code", "target_code"]).size()

        assert counts.to_dict() == {
            ("A", "X"): 2,
            ("A", "Y"): 2,
            ("B", "X"): 2,
            ("B", "Y"): 2,
        }

    @pytest.mark.parametrize("test_size", [-0.1, 0.0, 1.0, 1.5])
    def test_assign_train_validation_split_invalid_size(
        self,
        split_data,
        test_size,
    ):
        with pytest.raises(ValueError):
            assign_train_validation_split(
                data=split_data,
                test_size=test_size,
            )

    def test_assign_train_validation_split_rare_class(self):
        data = pd.DataFrame({
            "nature_code": ["A"] * 9 + ["B"],
            "target_code": ["X"] * 10,
        })

        with pytest.raises(ValueError):
            assign_train_validation_split(data=data)

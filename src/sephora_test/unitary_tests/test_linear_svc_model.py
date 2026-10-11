import numpy as np
import pytest

from sephora_test.models.linear_svc_model import LinearSVCModel


class TestLinearSVCModel:
    @pytest.fixture
    def training_data(self):
        features = np.array([
            [-3.0, -2.0],
            [-2.0, -3.0],
            [-2.0, -2.0],
            [2.0, 2.0],
            [2.0, 3.0],
            [3.0, 2.0],
        ])
        target = np.array(["A", "A", "A", "B", "B", "B"])

        return features, target

    def test_default_parameters(self):
        model = LinearSVCModel()

        assert model.class_weight is None
        assert model.random_state == 42
        assert not hasattr(model, "model_")

    def test_fit_returns_self(self, training_data):
        features, target = training_data
        model = LinearSVCModel()

        result = model.fit(features, target)

        assert result is model
        assert model.is_fitted_ is True

    def test_parameters_are_transmitted(self, training_data):
        features, target = training_data
        weights = {"A": 1.0, "B": 2.0}
        model = LinearSVCModel(
            class_weight=weights,
            random_state=123,
        )

        model.fit(features, target)

        assert model.model_.class_weight == weights
        assert model.model_.random_state == 123

    def test_predict_before_fit_raises(self):
        model = LinearSVCModel()

        with pytest.raises(
            RuntimeError,
            match="must be fitted before predict",
        ):
            model.predict(np.array([[1.0, 2.0]]))

    def test_predict_on_new_data(self, training_data):
        features, target = training_data
        model = LinearSVCModel()
        model.fit(features, target)

        new_features = np.array([
            [-4.0, -4.0],
            [4.0, 4.0],
        ])

        predictions = model.predict(new_features)

        np.testing.assert_array_equal(
            predictions,
            np.array(["A", "B"]),
        )

    def test_fit_with_mismatched_lengths_raises(self, training_data):
        features, target = training_data
        model = LinearSVCModel()

        with pytest.raises(ValueError):
            model.fit(features, target[:-1])

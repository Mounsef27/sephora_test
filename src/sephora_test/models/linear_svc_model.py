from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import LinearSVC


class LinearSVCModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        class_weight=None,
        random_state: int = 42,
    ):
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(
        self,
        features,
        target,
    ) -> "LinearSVCModel":
        self.model_ = LinearSVC(
            class_weight=self.class_weight,
            random_state=self.random_state,
        )

        self.model_.fit(
            features,
            target,
        )

        self.is_fitted_ = True

        return self

    def predict(
        self,
        features,
    ):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("LinearSVCModel must be fitted before predict.")

        return self.model_.predict(features)

"""This module contains a transformer that constructs labels
(`nature_code` or `target_code`) from cleaned product text columns.
"""

from typing import Dict, List

from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from sephora_test.logger import setup_logger

logger = setup_logger(__name__)


class LabelConstructor(BaseEstimator, TransformerMixin):
    """Construct labels from product text columns.

    The transformer first searches for keywords in the first configured
    text column. If no label is found, it searches in the remaining
    configured text columns.

    Args:
        label_map (Dict[str, List[str]]):
            Dictionary mapping each output label to associated keywords.

        columns_list_to_use (List[str]):
            Columns used to construct the label.

        output_column_name (str):
            Name of the output prediction column.
    """

    def __init__(
        self,
        label_map: Dict[str, List[str]],
        columns_list_to_use: List[str],
        output_column_name: str,
    ) -> None:
        """Initialize the LabelConstructor."""
        self.label_map = label_map
        self.columns_list_to_use = columns_list_to_use
        self.output_column_name = output_column_name

        logger.info(
            "LabelConstructor initialized for output column '%s'.",
            self.output_column_name,
        )

    def fit(
        self,
        features: DataFrame,
        target=None,
    ) -> "LabelConstructor":
        """Fit the transformer.

        Args:
            features (DataFrame):
                Input dataframe.

            target:
                Optional target values.

        Returns:
            LabelConstructor:
                Fitted transformer.
        """
        logger.info(
            "Fitting LabelConstructor on %d rows.",
            len(features),
        )

        logger.info(
            "Columns used for label construction: %s.",
            self.columns_list_to_use,
        )

        self.is_fitted_ = True

        return self

    def __sklearn_is_fitted__(self) -> bool:
        """Return whether the transformer has been fitted."""
        return getattr(
            self,
            "is_fitted_",
            False,
        )

    def _predict_label(
        self,
        row,
    ) -> str | None:
        """Predict a label from name first, then description."""

        name_column = self.columns_list_to_use[0]

        name_text = str(row.get(name_column, "")).lower()

        for label, keywords in self.label_map.items():
            for keyword in keywords:
                if keyword.lower() in name_text:
                    return label

        description_text = " ".join(
            str(row[column])
            for column in self.columns_list_to_use[1:]
            if column in row and row[column] is not None
        ).lower()

        for label, keywords in self.label_map.items():
            for keyword in keywords:
                if keyword.lower() in description_text:
                    return label

        return None

    def NatureConstructor(
        self,
        features: DataFrame,
        target=None,
    ) -> DataFrame:
        """Construct the nature_code prediction column."""
        logger.info(
            "Starting nature_code construction for %d rows.",
            len(features),
        )

        features = features.copy()

        features[self.output_column_name] = features.apply(
            self._predict_label,
            axis=1,
        )

        predicted_count = features[self.output_column_name].notna().sum()

        unmatched_count = features[self.output_column_name].isna().sum()

        logger.info(
            "nature_code construction completed: "
            "%d labels predicted, %d rows unmatched.",
            predicted_count,
            unmatched_count,
        )

        return features

    def TargetConstructor(
        self,
        features: DataFrame,
        target=None,
    ) -> DataFrame:
        """Construct the target_code prediction column."""
        logger.info(
            "Starting target_code construction for %d rows.",
            len(features),
        )

        features = features.copy()

        features[self.output_column_name] = features.apply(
            self._predict_label,
            axis=1,
        )

        predicted_count = features[self.output_column_name].notna().sum()

        unmatched_count = features[self.output_column_name].isna().sum()

        logger.info(
            "target_code construction completed: "
            "%d labels predicted, %d rows unmatched.",
            predicted_count,
            unmatched_count,
        )

        return features

    def transform(
        self,
        features: DataFrame,
    ) -> DataFrame:
        """Apply the configured label construction transformation."""
        check_is_fitted(
            self,
            attributes=["is_fitted_"],
        )

        logger.info(
            "Applying sklearn transform for output column '%s'.",
            self.output_column_name,
        )

        if self.output_column_name == "nature_code_predicted":
            return self.NatureConstructor(features)

        if self.output_column_name == "target_code_predicted":
            return self.TargetConstructor(features)

        raise ValueError(
            f"Unsupported output column: {self.output_column_name}"
        )

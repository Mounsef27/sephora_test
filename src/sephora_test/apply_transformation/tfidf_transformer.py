from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted


class TfidfTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns_list_to_use: list[str],
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.98,
        sublinear_tf: bool = True,
    ):
        self.columns_list_to_use = columns_list_to_use
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

    def _combine_text_columns(
        self,
        features: DataFrame,
    ):
        return (
            features[self.columns_list_to_use]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )

    def fit(
        self,
        features: DataFrame,
        target=None,
    ) -> "TfidfTransformer":
        combined_text = self._combine_text_columns(features)

        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
        )

        self.vectorizer_.fit(combined_text)

        self.is_fitted_ = True

        return self

    def transform(
        self,
        features: DataFrame,
    ):
        check_is_fitted(
            self,
            attributes=[
                "vectorizer_",
                "is_fitted_",
            ],
        )

        combined_text = self._combine_text_columns(features)

        return self.vectorizer_.transform(combined_text)

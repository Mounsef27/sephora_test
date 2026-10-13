import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError

from sephora_test.apply_transformation.tfidf_transformer import (
    TfidfTransformer,
)


class TestTfidfTransformer:
    @pytest.fixture
    def data(self):
        return pd.DataFrame({
            "cleanName": ["parfum floral", "crème visage", "parfum doux"],
            "cleanDescription": [
                "rose légère",
                "hydratation intense",
                "vanille légère",
            ],
        })

    @pytest.fixture
    def transformer(self):
        # Small datasets need less restrictive frequency thresholds.
        return TfidfTransformer(
            columns_list_to_use=["cleanName", "cleanDescription"],
            min_df=1,
            max_df=1.0,
        )

    def test_combine_text_columns(self, transformer):
        data = pd.DataFrame(
            {
                "cleanName": ["parfum", "crème"],
                "cleanDescription": ["floral", "visage"],
                "unused": ["ignored", "ignored"],
            },
            index=[10, 20],
        )

        combined = transformer._combine_text_columns(data)

        expected = pd.Series(
            ["parfum floral", "crème visage"],
            index=[10, 20],
        )
        pd.testing.assert_series_equal(combined, expected)

    def test_combine_missing_and_numeric_values(self, transformer):
        data = pd.DataFrame({
            "cleanName": [None, "parfum", 123],
            "cleanDescription": ["floral", np.nan, "produit"],
        })

        combined = transformer._combine_text_columns(data)

        assert combined.tolist() == [
            " floral",
            "parfum ",
            "123 produit",
        ]

    def test_fit_returns_self(self, transformer, data):
        assert transformer.fit(data) is transformer
        assert transformer.is_fitted_ is True
        assert len(transformer.vectorizer_.vocabulary_) > 0

    def test_vectorizer_parameters(self, data):
        transformer = TfidfTransformer(
            columns_list_to_use=["cleanName", "cleanDescription"],
            ngram_range=(1, 1),
            min_df=1,
            max_df=1.0,
            sublinear_tf=False,
        )

        transformer.fit(data)
        parameters = transformer.vectorizer_.get_params()

        assert parameters["ngram_range"] == (1, 1)
        assert parameters["min_df"] == 1
        assert parameters["max_df"] == 1.0
        assert parameters["sublinear_tf"] is False

    def test_transform_before_fit_raises(self, transformer, data):
        with pytest.raises(NotFittedError):
            transformer.transform(data)

    def test_transform_returns_sparse_matrix(self, transformer, data):
        transformer.fit(data)

        result = transformer.transform(data)

        assert issparse(result)
        assert result.shape == (
            len(data),
            len(transformer.vectorizer_.vocabulary_),
        )
        assert np.isfinite(result.data).all()
        assert (result.data >= 0).all()
        assert result.nnz > 0

    def test_transform_does_not_learn_new_words(self, transformer, data):
        transformer.fit(data)
        original_vocabulary = transformer.vectorizer_.vocabulary_.copy()

        new_data = pd.DataFrame({
            "cleanName": ["parfum nouveaumot"],
            "cleanDescription": ["floral"],
        })

        result = transformer.transform(new_data)

        assert transformer.vectorizer_.vocabulary_ == original_vocabulary
        assert "nouveaumot" not in transformer.vectorizer_.vocabulary_
        assert result.shape == (1, len(original_vocabulary))
        assert result.nnz > 0

    def test_unknown_words_produce_zero_vector(self, transformer, data):
        transformer.fit(data)
        new_data = pd.DataFrame({
            "cleanName": ["zzzinconnu"],
            "cleanDescription": ["yyyabsent"],
        })

        result = transformer.transform(new_data)

        assert result.shape[0] == 1
        assert result.nnz == 0

    def test_missing_values_during_transform(self, transformer, data):
        transformer.fit(data)
        new_data = pd.DataFrame({
            "cleanName": [None, "parfum"],
            "cleanDescription": [None, None],
        })

        result = transformer.transform(new_data)

        assert result.shape[0] == 2
        assert result[0].nnz == 0
        assert result[1].nnz > 0

    def test_input_dataframe_is_preserved(self, transformer, data):
        original = data.copy(deep=True)

        transformer.fit_transform(data)

        pd.testing.assert_frame_equal(data, original)

    def test_missing_column_raises(self, transformer):
        data = pd.DataFrame({"cleanName": ["parfum"]})

        with pytest.raises(KeyError):
            transformer.fit(data)

    def test_fit_with_only_empty_text_raises(self, transformer):
        data = pd.DataFrame({
            "cleanName": ["", None],
            "cleanDescription": [None, ""],
        })

        with pytest.raises(ValueError, match="empty vocabulary"):
            transformer.fit(data)

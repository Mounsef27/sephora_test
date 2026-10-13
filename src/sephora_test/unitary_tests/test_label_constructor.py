import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

# Adapte cet import à ton projet.
from sephora_test.apply_transformation.label_construction import (
    LabelConstructor,
)


class TestLabelConstructor:
    @pytest.fixture
    def constructor(self):
        return LabelConstructor(
            label_map={
                "PERFUME": ["parfum", "fragrance"],
                "CREAM": ["crème", "cream"],
            },
            columns_list_to_use=["cleanName", "cleanDescription"],
            output_column_name="nature_code_predicted",
        )

    def test_fit_returns_self(self, constructor):
        data = pd.DataFrame({
            "cleanName": ["parfum"],
            "cleanDescription": [""],
        })

        assert constructor.fit(data) is constructor
        assert constructor.__sklearn_is_fitted__() is True

    def test_not_fitted_initially(self, constructor):
        assert constructor.__sklearn_is_fitted__() is False

    def test_transform_before_fit_raises(self, constructor):
        data = pd.DataFrame({
            "cleanName": ["parfum"],
            "cleanDescription": [""],
        })

        with pytest.raises(NotFittedError):
            constructor.transform(data)

    @pytest.mark.parametrize(
        ("name", "description", "expected"),
        [
            # Match in the name.
            ("parfum floral", "", "PERFUME"),
            # Alternative keyword.
            ("fragrance florale", "", "PERFUME"),
            # Case-insensitive matching.
            ("PARFUM FLORAL", "", "PERFUME"),
            # Fall back to the description.
            ("produit beauté", "une crème douce", "CREAM"),
            # Name takes priority over description.
            ("crème visage", "parfum floral", "CREAM"),
            # No matching keyword.
            ("produit", "description inconnue", None),
            # Empty text.
            ("", "", None),
            # Missing values.
            (None, None, None),
        ],
    )
    def test_label_matching(
        self,
        constructor,
        name,
        description,
        expected,
    ):
        data = pd.DataFrame({
            "cleanName": [name],
            "cleanDescription": [description],
        })

        result = constructor.fit_transform(data)
        actual = result["nature_code_predicted"].iloc[0]

        if expected is None:
            assert pd.isna(actual)
        else:
            assert actual == expected

    @pytest.mark.parametrize(
        "output_column",
        ["nature_code_predicted", "target_code_predicted"],
    )
    def test_supported_output_columns(self, output_column):
        constructor = LabelConstructor(
            label_map={"PERFUME": ["parfum"]},
            columns_list_to_use=["cleanName", "cleanDescription"],
            output_column_name=output_column,
        )
        data = pd.DataFrame({
            "cleanName": ["parfum"],
            "cleanDescription": [""],
        })

        result = constructor.fit_transform(data)

        assert result[output_column].tolist() == ["PERFUME"]

    def test_input_dataframe_is_preserved(self, constructor):
        data = pd.DataFrame(
            {
                "cleanName": ["parfum", "crème"],
                "cleanDescription": ["floral", "visage"],
            },
            index=[10, 20],
        )
        original = data.copy(deep=True)

        result = constructor.fit_transform(data)

        pd.testing.assert_frame_equal(data, original)
        pd.testing.assert_frame_equal(
            result.drop(columns="nature_code_predicted"),
            original,
        )

    def test_searches_all_remaining_columns(self):
        constructor = LabelConstructor(
            label_map={"PERFUME": ["parfum"]},
            columns_list_to_use=[
                "cleanName",
                "cleanDescription",
                "details",
            ],
            output_column_name="nature_code_predicted",
        )
        data = pd.DataFrame({
            "cleanName": ["produit"],
            "cleanDescription": ["description"],
            "details": ["parfum floral"],
        })

        result = constructor.fit_transform(data)

        assert result["nature_code_predicted"].iloc[0] == "PERFUME"

    def test_missing_description_column(self, constructor):
        data = pd.DataFrame({"cleanName": ["parfum"]})

        result = constructor.fit_transform(data)

        assert result["nature_code_predicted"].iloc[0] == "PERFUME"

    def test_first_matching_label_wins(self, constructor):
        data = pd.DataFrame({
            "cleanName": ["crème parfum"],
            "cleanDescription": [""],
        })

        result = constructor.fit_transform(data)

        # PERFUME is first in label_map, even though crème
        # appears first in the text.
        assert result["nature_code_predicted"].iloc[0] == "PERFUME"

    def test_empty_label_map(self):
        constructor = LabelConstructor(
            label_map={},
            columns_list_to_use=["cleanName"],
            output_column_name="nature_code_predicted",
        )
        data = pd.DataFrame({"cleanName": ["parfum"]})

        result = constructor.fit_transform(data)

        assert result["nature_code_predicted"].isna().all()

    def test_unsupported_output_column_raises(self):
        constructor = LabelConstructor(
            label_map={"PERFUME": ["parfum"]},
            columns_list_to_use=["cleanName"],
            output_column_name="invalid_column",
        )
        data = pd.DataFrame({"cleanName": ["parfum"]})
        constructor.fit(data)

        with pytest.raises(
            ValueError,
            match="Unsupported output column",
        ):
            constructor.transform(data)

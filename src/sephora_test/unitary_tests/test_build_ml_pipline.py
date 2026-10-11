import pytest
from sklearn.pipeline import Pipeline

from sephora_test.apply_transformation.tfidf_transformer import (
    TfidfTransformer,
)
from sephora_test.models.linear_svc_model import LinearSVCModel
from sephora_test.run_experiments.build_ml_pipeline import build_ml_pipeline


class TestBuildMLPipeline:
    def test_pipeline_structure(self):
        pipeline = build_ml_pipeline()

        assert isinstance(pipeline, Pipeline)
        assert list(pipeline.named_steps) == ["tfidf", "classifier"]
        assert isinstance(
            pipeline.named_steps["tfidf"],
            TfidfTransformer,
        )
        assert isinstance(
            pipeline.named_steps["classifier"],
            LinearSVCModel,
        )

    def test_tfidf_parameters(self):
        pipeline = build_ml_pipeline()
        transformer = pipeline.named_steps["tfidf"]

        assert transformer.columns_list_to_use == [
            "cleanName",
            "cleanDescription",
        ]
        assert transformer.ngram_range == (1, 2)
        assert transformer.min_df == 2
        assert transformer.max_df == 0.98
        assert transformer.sublinear_tf is True

    @pytest.mark.parametrize(
        "class_weight",
        [None, "balanced", {"A": 1.0, "B": 2.0}],
    )
    def test_class_weight(self, class_weight):
        pipeline = build_ml_pipeline(class_weight=class_weight)

        classifier = pipeline.named_steps["classifier"]

        assert classifier.class_weight == class_weight

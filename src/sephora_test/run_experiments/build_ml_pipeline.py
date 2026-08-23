from sklearn.pipeline import Pipeline

from sephora_test.apply_transformation.tfidf_transformer import (
    TfidfTransformer,
)
from sephora_test.models.linear_svc_model import (
    LinearSVCModel,
)


def build_ml_pipeline(
    class_weight=None,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfTransformer(
                    columns_list_to_use=[
                        "cleanName",
                        "cleanDescription",
                    ],
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVCModel(
                    class_weight=class_weight,
                ),
            ),
        ]
    )

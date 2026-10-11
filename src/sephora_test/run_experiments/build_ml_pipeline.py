from sklearn.pipeline import Pipeline

from sephora_test.apply_transformation.tfidf_transformer import (
    TfidfTransformer,
)
from sephora_test.models.linear_svc_model import LinearSVCModel


def build_ml_pipeline(class_weight=None) -> Pipeline:
    """Construit un pipeline de classification de texte.

    Args:
        class_weight: Pondération des classes transmise à LinearSVCModel.

    Returns:
        Pipeline non entraîné combinant TF-IDF et LinearSVC.
    """
    tfidf_transformer = TfidfTransformer(
        columns_list_to_use=["cleanName", "cleanDescription"],
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
    )

    classifier = LinearSVCModel(class_weight=class_weight)

    pipeline = Pipeline(
        steps=[
            ("tfidf", tfidf_transformer),
            ("classifier", classifier),
        ]
    )

    return pipeline

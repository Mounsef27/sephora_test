Référence API
=============

Les transformations sont documentées dans :doc:`apply_transformation`.

Modèle LinearSVC
--------------------------------------------------------------------------------

.. autoclass:: sephora_test.models.linear_svc_model.LinearSVCModel
   :members: fit, predict
   :undoc-members:
   :show-inheritance:

Construction du pipeline
--------------------------------------------------------------------------------

.. autofunction:: sephora_test.run_experiments.build_ml_pipeline.build_ml_pipeline

Expérience de machine learning
--------------------------------------------------------------------------------

.. autofunction:: sephora_test.run_experiments.run_ml_pipeline.run_ml_pipeline

Baseline par mots-clés
--------------------------------------------------------------------------------

.. autofunction:: sephora_test.run_experiments.run_experiments.run_experiments

Métriques de classification
--------------------------------------------------------------------------------

.. autofunction:: sephora_test.model_eval.calculate_metrics.calculate_all_metrics

Séparation des données
--------------------------------------------------------------------------------

.. autofunction:: sephora_test.utils.utils.assign_train_validation_split

Gestion des artefacts
--------------------------------------------------------------------------------

.. autofunction:: sephora_test.utils.utils.create_artifact_directory

.. autofunction:: sephora_test.utils.utils.save_results_pickle

.. autofunction:: sephora_test.utils.utils.save_test_predictions_csv

.. autofunction:: sephora_test.utils.utils.save_train_metrics_txt

.. autofunction:: sephora_test.utils.utils.save_confusion_matrix_image

Journalisation
--------------------------------------------------------------------------------

.. autofunction:: sephora_test.logger.setup_logger

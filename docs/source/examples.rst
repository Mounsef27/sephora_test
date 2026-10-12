Exemple complet : entraînement des deux cibles
==============================================

Cet exemple charge les CSV, prépare les textes, sépare les données,
entraîne les deux modèles et sauvegarde les résultats.

Exécutez-le depuis la racine du dépôt avec :

* ``data/training_set.csv`` ;
* ``data/testing_set.csv``.

Les colonnes attendues sont décrites dans le :doc:`guide`.

Code complet
------------

.. code-block:: python

   from copy import deepcopy
   from datetime import datetime

   import pandas as pd

   from sephora_test.apply_transformation.clean_text import clean_text
   from sephora_test.run_experiments.run_ml_pipeline import (
       config,
       run_ml_pipeline,
   )
   from sephora_test.utils.utils import assign_train_validation_split

   def prepare(data):
       data = data.copy()

       for source, destination in [
           ("productName", "cleanName"),
           ("longDescription", "cleanDescription"),
       ]:
           data[destination] = (
               data[source]
               .fillna("")
               .astype(str)
               .map(clean_text)
           )

       return data

   training = prepare(pd.read_csv("data/training_set.csv"))
   testing = prepare(pd.read_csv("data/testing_set.csv"))

   settings = deepcopy(config)
   settings["artifacts_path"] = "artifacts"
   settings["run_datetime"] = datetime.now().strftime(
       "%Y-%m-%d_%H-%M-%S"
   )

   training = assign_train_validation_split(
       training,
       test_size=settings["validation_size"],
       random_state=settings["random_state"],
   )

   results = run_ml_pipeline(
       training_data=training,
       testing_data=testing,
       experiment_config=settings,
   )

   print(results["nature_metrics"])
   print(results["target_metrics"])

Fonctionnement
--------------

``run_ml_pipeline`` reçoit des DataFrames déjà préparés.
Elle ne charge pas les CSV et ne crée pas la colonne ``sample``.

Elle entraîne deux pipelines indépendants sur les lignes ``train``,
évalue leurs prédictions sur les lignes ``validation``, puis prédit
les deux attributs du jeu de test.

Les chemins ``data_paths`` de la configuration ne sont pas utilisés
par cette fonction pour charger les données.

Résultats retournés
-------------------

Le dictionnaire contient :

* ``config`` ;
* ``nature_pipeline`` ;
* ``target_pipeline`` ;
* ``train_data`` ;
* ``validation_data`` ;
* ``test_processed`` ;
* ``nature_metrics`` ;
* ``target_metrics``.

Fichiers générés
----------------

Un sous-dossier horodaté est créé dans ``artifacts``.

Il contient :

* les résultats et les modèles sérialisés au format pickle ;
* les métriques au format texte ;
* les matrices de confusion au format PNG ;
* le CSV des prédictions.

Le CSV utilise les colonnes ``productId``, ``nature_predict``
et ``target_predict``.

Réutiliser les modèles
----------------------

.. code-block:: python

   new_products = prepare(pd.read_csv("data/testing_set.csv"))

   nature_predictions = results["nature_pipeline"].predict(
       new_products
   )

   target_predictions = results["target_pipeline"].predict(
       new_products
   )

Les modèles retournés restent entraînés sur le sous-ensemble ``train``.
La fonction ne les réentraîne pas sur l'ensemble des données après
l'évaluation.

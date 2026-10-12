Documentation de sephora_test
=============================

``sephora_test`` permet de classifier des produits à partir de leur nom
et de leur description textuelle.

Deux attributs sont prédits :

* ``nature_code`` : la catégorie du produit ;
* ``target_code`` : la cible du produit.

Le package propose une baseline par mots-clés et une approche
de machine learning combinant TF-IDF et LinearSVC.

Quick Start
-----------

Installation
~~~~~~~~~~~~

Le projet nécessite Python 3.13 ou supérieur et utilise ``uv``.

Depuis la racine du dépôt :

.. code-block:: bash

   uv sync
   uv run python -m nltk.downloader punkt punkt_tab stopwords

Entraîner un premier modèle
~~~~~~~~~~~~~~~~~~~~~~~~~~

Cet exemple utilise des textes fictifs déjà nettoyés pour présenter
l'utilisation du pipeline. Il ne constitue pas une évaluation du modèle.

.. code-block:: python

   import pandas as pd

   from sephora_test.run_experiments.build_ml_pipeline import (
       build_ml_pipeline,
   )

   products = pd.DataFrame({
       "cleanName": [
           "mascara noir",
           "mascara volume",
           "serum visage",
           "serum hydratant",
       ],
       "cleanDescription": [
           "maquillage cils volume",
           "maquillage cils longueur",
           "soin visage hydratant",
           "soin visage douceur",
       ],
   })

   labels = ["MASCARAS", "MASCARAS", "SERUM", "SERUM"]

   pipeline = build_ml_pipeline()
   pipeline.fit(products, labels)

Prédire la catégorie d'un produit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   new_products = pd.DataFrame({
       "cleanName": ["mascara"],
       "cleanDescription": ["maquillage cils"],
   })

   predictions = pipeline.predict(new_products)
   print(predictions)

Le pipeline apprend le vocabulaire TF-IDF et le classificateur lors de
``fit``. Lors de ``predict``, il réutilise ces composants entraînés.

Les données doivent contenir ``cleanName`` et ``cleanDescription``.
Le :doc:`guide` explique comment préparer ces colonnes à partir
des textes bruts.

Découvrir le package
-------------------

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   guide
   examples
   api/apply_transformation
   api/reference

Index et recherche
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

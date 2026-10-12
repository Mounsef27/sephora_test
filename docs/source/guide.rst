Guide utilisateur
=================

Format des données
------------------

Les données brutes utilisent les colonnes suivantes :

``productName``
    Nom du produit.

``longDescription``
    Description textuelle du produit.

``nature_code``
    Catégorie réelle, nécessaire pour entraîner le modèle de catégorie.

``target_code``
    Cible réelle, nécessaire pour entraîner le modèle de cible.

``productId``
    Identifiant nécessaire pour exporter les prédictions.

Le pipeline ML attend les colonnes préparées ``cleanName`` et
``cleanDescription``.

Nettoyage du texte
------------------

La fonction ``clean_text`` effectue les opérations suivantes :

* décodage des entités HTML ;
* suppression des balises HTML ;
* normalisation des espaces ;
* remplacement des caractères de ponctuation ;
* conversion en minuscules ;
* suppression des mots vides français.

Elle ne réalise pas de lemmatisation.

.. code-block:: python

   from sephora_test.apply_transformation.clean_text import clean_text

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

.. note::

   Convertissez les valeurs manquantes et non textuelles avant
   d'appeler ``clean_text``. Dans l'implémentation actuelle, le message
   de journalisation pour une entrée invalide ne suffit pas à interrompre
   son traitement.

Ressources NLTK
~~~~~~~~~~~~~~~

Le module de nettoyage utilise les mots vides français de NLTK.
Il tente de télécharger les ressources manquantes à son importation.

Pour préparer l'environnement :

.. code-block:: bash

   uv run python -m nltk.downloader punkt punkt_tab stopwords

Séparation entraînement et validation
-------------------------------------

``assign_train_validation_split`` ajoute une colonne ``sample`` à
une copie du DataFrame.

Les valeurs possibles sont :

* ``train`` ;
* ``validation``.

La stratification utilise la combinaison de ``nature_code`` et
``target_code``. Chaque combinaison doit avoir suffisamment
d'observations pour permettre la séparation demandée.

.. code-block:: python

   from sephora_test.utils.utils import assign_train_validation_split

   training = assign_train_validation_split(
       training,
       test_size=0.2,
       random_state=42,
   )

Utilisez un index unique. La graine fixe rend la séparation reproductible
pour un même jeu de données.

Le vocabulaire TF-IDF doit être appris uniquement sur l'entraînement.
La validation et le test utilisent ensuite le pipeline déjà entraîné.

Représentation TF-IDF
---------------------

``TfidfTransformer`` concatène les colonnes configurées dans leur ordre.
Les valeurs manquantes sont remplacées par des chaînes vides.

Les paramètres par défaut sont :

``ngram_range=(1, 2)``
    Utilisation des mots isolés et des groupes de deux mots.

``min_df=2``
    Suppression des termes présents dans moins de deux documents.

``max_df=0.98``
    Suppression des termes présents dans plus de 98 % des documents.

``sublinear_tf=True``
    Utilisation d'une pondération logarithmique des fréquences.

.. note::

   Sur un corpus très petit, le filtrage peut supprimer tout le
   vocabulaire. Pour explorer un petit jeu de données, vous pouvez
   construire un transformeur avec ``min_df=1`` et ``max_df=1.0``.

Classification avec LinearSVC
-----------------------------

``build_ml_pipeline`` assemble deux étapes :

* ``tfidf`` : représentation du texte ;
* ``classifier`` : classification avec ``LinearSVCModel``.

.. code-block:: python

   from sephora_test.run_experiments.build_ml_pipeline import (
       build_ml_pipeline,
   )

   nature_pipeline = build_ml_pipeline()
   target_pipeline = build_ml_pipeline(class_weight="balanced")

Les deux cibles sont entraînées séparément.
``class_weight="balanced"`` pondère les classes inversement
à leur fréquence dans les données d'entraînement.

Le wrapper propose ``fit`` et ``predict``.
Il ne fournit pas de méthode de prédiction des probabilités.

Baseline par mots-clés
----------------------

``LabelConstructor`` cherche les mots-clés dans la première colonne
configurée, puis dans les autres colonnes combinées.

La recherche :

* ignore la casse ;
* utilise des correspondances par sous-chaîne ;
* retourne la première correspondance selon l'ordre du dictionnaire
  et des mots-clés ;
* retourne ``None`` si aucun mot-clé ne correspond.

``fit`` marque le transformeur comme entraîné sans apprendre de paramètres.

Les noms de colonnes de sortie pris en charge par ``transform`` sont
``nature_code_predicted`` et ``target_code_predicted``.

Évaluation
----------

``calculate_all_metrics`` retourne un dictionnaire indexé par
le nom du jeu évalué.

Les métriques disponibles sont :

* accuracy ;
* balanced accuracy ;
* F1 macro ;
* F1 pondéré ;
* précision macro ;
* rappel macro ;
* rapport de classification ;
* matrice de confusion normalisée par classe réelle.

Examinez les métriques par classe, notamment lorsque les classes
sont déséquilibrées.

Construire la documentation
---------------------------

.. code-block:: bash

   uv run sphinx-build -E -W -b html docs/source docs/build/html
   xdg-open docs/build/html/index.html

L'option ``-E`` reconstruit l'environnement Sphinx.
L'option ``-W`` transforme les warnings en erreurs de construction.

# Sephora Data Science Technical Test

<p align="center">
  <strong>Product classification with rule-based NLP and TF-IDF + LinearSVC</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-blue">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.9-orange">
  <img alt="uv" src="https://img.shields.io/badge/package%20manager-uv-6f42c1">
  <img alt="Ruff" src="https://img.shields.io/badge/code%20quality-Ruff-d7ff64">
  <img alt="mypy" src="https://img.shields.io/badge/type%20checking-mypy-2a6db2">
</p>

---

## Overview

This repository contains the solution developed for the **Sephora Data Science Technical Test**.

The objective is to predict two product attributes from textual information:

- **`nature_code`** — multiclass product-category classification.
- **`target_code`** — binary target classification (`FEMME` / `HOMME`).

The available textual features are mainly:

- `productName`
- `longDescription`

Two approaches are implemented:

1. **Rule-based keyword baseline**
2. **TF-IDF + LinearSVC machine-learning pipeline**

The project is designed as a reusable Python package with preprocessing, sklearn-compatible transformers and estimators, evaluation utilities, logging, experiment tracking, artifact generation, notebooks, and code-quality checks.

---

## Best validation results

The TF-IDF + LinearSVC approach is evaluated on a **stratified 80/20 train-validation split** of the provided training dataset.

### `nature_code`

| Metric | Score |
|---|---:|
| Accuracy | **0.9241** |
| Balanced Accuracy | **0.9241** |
| Macro F1 | **0.9223** |
| Weighted F1 | **0.9222** |
| Macro Precision | **0.9234** |
| Macro Recall | **0.9241** |

### `target_code`

| Metric | Score |
|---|---:|
| Accuracy | **0.9852** |
| Balanced Accuracy | **0.9426** |
| Macro F1 | **0.9515** |
| Weighted F1 | **0.9850** |
| Macro Precision | **0.9609** |
| Macro Recall | **0.9426** |

Despite the strong class imbalance in `target_code`, the model reaches a **0.89 recall for the HOMME class** and a **0.91 F1-score** on validation.

---

## `nature_code` class performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| COFFRET | 0.88 | 0.70 | 0.78 |
| CREM JOUR | 0.84 | 0.85 | 0.85 |
| EAUPARFUM | 0.93 | 0.98 | 0.95 |
| EAUTOIL | 0.92 | 1.00 | 0.96 |
| MASCARAS | 0.96 | 0.96 | 0.96 |
| NETTOYANT | 0.94 | 0.93 | 0.93 |
| PALETYEUX | 0.98 | 0.98 | 0.98 |
| RAL | 0.98 | 1.00 | 0.99 |
| SERUM | 0.87 | 0.85 | 0.86 |
| SHAMPOINGS | 0.93 | 0.98 | 0.95 |

The main remaining difficulty is the **COFFRET** class, which has lower recall than the other categories.

---

## ML architecture

```text
productName + longDescription
             │
             ▼
       Text preprocessing
             │
             ▼
    cleanName + cleanDescription
             │
             ▼
     Custom TF-IDF Transformer
             │
             ▼
          LinearSVC
          /       \
         /         \
 nature_code     target_code
 multiclass       binary
```

The two targets are modeled independently:

- `nature_code`: standard `LinearSVC`
- `target_code`: `LinearSVC(class_weight="balanced")`

This allows the binary classifier to better handle the strong `FEMME` / `HOMME` imbalance.

---

## Validation strategy

Only the provided **training dataset** is split.

```text
training_set.csv
      │
      ▼
stratified split
      │
  ┌───┴─────────────┐
  ▼                 ▼
Train 80%      Validation 20%
  │                 │
  │                 ▼
  │             Evaluation
  │
  ▼
TF-IDF fit
  │
  ▼
LinearSVC fit
```

The split is stratified using the combination of:

```text
nature_code + target_code
```

This helps preserve the distribution of both prediction targets, including the minority `HOMME` class.

The provided `testing_set.csv` is used **only for final inference** and never for model fitting or parameter selection.

---

## Text preprocessing

The preprocessing pipeline prepares both text columns:

```text
productName      → cleanName
longDescription  → cleanDescription
```

The cleaning utilities handle:

- missing values,
- HTML entities,
- HTML tags,
- text normalization,
- punctuation processing,
- French stopwords,
- whitespace normalization.

The cleaned columns are then jointly transformed by TF-IDF.

---

## sklearn-compatible components

The project implements custom components compatible with the sklearn API.

### TF-IDF transformer

`TfidfTransformer` inherits from:

```python
BaseEstimator
TransformerMixin
```

It combines:

```text
cleanName + cleanDescription
```

and learns the TF-IDF representation on the training subset only.

### LinearSVC model

`LinearSVCModel` inherits from:

```python
BaseEstimator
ClassifierMixin
```

It wraps sklearn's `LinearSVC` while remaining reusable inside a standard sklearn `Pipeline`.

### Complete ML pipeline

```python
Pipeline(
    steps=[
        ("tfidf", TfidfTransformer(...)),
        ("classifier", LinearSVCModel(...)),
    ]
)
```

This guarantees that the same fitted TF-IDF vocabulary is reused for validation and test inference.

---

## Rule-based baseline

A first baseline is implemented using configurable keyword mappings for:

- `nature_code`
- `target_code`

The baseline follows a deterministic strategy using cleaned product names and descriptions.

This provides:

- an interpretable reference solution,
- a simple benchmark,
- a useful way to inspect discriminative vocabulary before moving to statistical learning.

The ML implementation is kept separate so both approaches remain reproducible.

---

## Project structure

```text
sephora_test/
│
├── data/
│   ├── training_set.csv
│   └── testing_set.csv
│
├── src/
│   ├── artifacts/
│   │   ├── 2026-08-23_18-23-19/
│   │   └── 2026-08-23_22-50-34/
│   │
│   └── sephora_test/
│       ├── apply_transformation/
│       │   ├── clean_text.py
│       │   ├── label_construction.py
│       │   └── tfidf_transformer.py
│       │
│       ├── config/
│       │   ├── data_paths.py
│       │   └── labels_map.py
│       │
│       ├── model_eval/
│       │   └── calculate_metrics.py
│       │
│       ├── models/
│       │   └── linear_svc_model.py
│       │
│       ├── run_experiments/
│       │   ├── build_ml_pipeline.py
│       │   ├── run_experiments.py
│       │   └── run_ml_pipeline.py
│       │
│       ├── utils/
│       │   └── utils.py
│       │
│       └── logger.py
│
├── _static/
│   ├── docs/
│   │   └── Data Science Technical Test.pdf
│   │
│   └── _notebooks/
│       ├── exploration.ipynb
│       ├── package_execution.ipynb
│       └── package_execution_svc.ipynb
│
├── .pre-commit-config.yaml
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Installation

The project uses **uv** for dependency and environment management.

### 1. Clone the repository

```bash
git clone git@github.com:Mounsef27/sephora_test.git
cd sephora_test
```

### 2. Synchronize the environment

```bash
uv sync
```

This installs the package dependencies declared in `pyproject.toml`.

### 3. Development dependencies

The development environment includes:

- Jupyter / IPython kernel
- Ruff
- mypy
- pre-commit
- NLTK
- dill

---

## Quick start

The easiest way to reproduce the experiments is through the notebooks.

### Rule-based pipeline

Open:

```text
_static/_notebooks/package_execution.ipynb
```

### TF-IDF + LinearSVC pipeline

Open:

```text
_static/_notebooks/package_execution_svc.ipynb
```

The ML notebook performs:

```text
Load data
   ↓
Text preprocessing
   ↓
Stratified train-validation assignment
   ↓
TF-IDF transformation
   ↓
LinearSVC training
   ↓
Validation evaluation
   ↓
Test inference
   ↓
Artifact generation
```

---

## ML experiment configuration

The ML experiment is configured in `run_ml_pipeline.py`.

```python
config = {
    "method_name": "tfidf_linear_svc",
    "validation_size": 0.2,
    "random_state": 42,
    "nature_model": {
        "class_weight": None,
    },
    "target_model": {
        "class_weight": "balanced",
    },
}
```

The fixed `random_state` makes the validation split reproducible.

---

## Generated artifacts

Every experiment creates a timestamped directory under:

```text
src/artifacts/
```

For the TF-IDF + LinearSVC experiment:

```text
2026-08-23_22-50-34/
├── MOUNSEF_DEBACHE_prediction.csv
├── tfidf_linear_svc_nature_confusion_matrix.png
├── tfidf_linear_svc_nature_metrics.txt
├── tfidf_linear_svc_results.pkl
├── tfidf_linear_svc_target_confusion_matrix.png
└── tfidf_linear_svc_target_metrics.txt
```

### Pickle artifact

The complete experiment is serialized in:

```text
tfidf_linear_svc_results.pkl
```

It contains:

```python
{
    "config",
    "nature_pipeline",
    "target_pipeline",
    "train_data",
    "validation_data",
    "test_processed",
    "nature_metrics",
    "target_metrics",
}
```

The trained sklearn pipelines are therefore preserved alongside the predictions, configuration, and evaluation results.

---

## Submission file

The final prediction file follows the required structure:

```text
productId
nature_predict
target_predict
```

and is generated as:

```text
MOUNSEF_DEBACHE_prediction.csv
```

---

## Evaluation metrics

The project reports:

- Accuracy
- Balanced Accuracy
- Macro F1
- Weighted F1
- Macro Precision
- Macro Recall
- Classification Report
- Normalized Confusion Matrix

Balanced Accuracy and Macro F1 are especially important for `target_code` because of the class imbalance.

---

## Code quality

The repository uses automated checks through **pre-commit**.

Run all checks with:

```bash
uv run pre-commit run --all-files
```

Configured checks include:

```text
✓ large-file detection
✓ merge-conflict detection
✓ YAML validation
✓ private-key detection
✓ end-of-file normalization
✓ trailing-whitespace cleanup
✓ Ruff linting
✓ Ruff formatting
✓ mypy type checking
```

Ruff is configured with:

```toml
[tool.ruff]
preview = true
line-length = 80

[tool.ruff.lint]
select = ["E", "F", "I", "B"]
```

---

## Main dependencies

| Package | Purpose |
|---|---|
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | TF-IDF, pipelines, LinearSVC, metrics |
| BeautifulSoup4 | HTML cleaning |
| lxml | HTML parsing |
| NLTK | French stopwords and tokenization |
| Matplotlib | Confusion-matrix visualization |
| dill / pickle | Experiment serialization |
| uv | Environment and dependency management |

Python requirement:

```text
Python >= 3.13
```

---

## Reproducibility

The project emphasizes reproducibility through:

- fixed random seed,
- deterministic train-validation assignment,
- sklearn pipelines,
- serialized fitted models,
- timestamped experiment folders,
- saved configuration,
- saved validation metrics,
- saved confusion matrices,
- versioned dependencies with `uv.lock`.

---

## Potential improvements

Several extensions could further improve performance:

- Tune `LinearSVC` regularization parameter `C`.
- Compare word-level and character-level TF-IDF.
- Use independent TF-IDF representations for `productName` and `longDescription`.
- Give higher weight to `productName`.
- Add cross-validation.
- Tune class weighting for `nature_code`.
- Build a hybrid system combining high-confidence business rules with ML.
- Analyze errors on the `COFFRET`, `CREM JOUR`, and `SERUM` classes.
- Compare with logistic regression, Naive Bayes, or transformer-based embeddings.

---

## Author

**Mounsef Debache**

Sephora Data Science Technical Test
Machine Learning · NLP · Data Science

GitHub: [Mounsef27](https://github.com/Mounsef27)

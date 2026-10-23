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
  <img alt="pytest" src="https://img.shields.io/badge/tests-pytest-green">
  <img alt="Bandit" src="https://img.shields.io/badge/security-Bandit-yellow">
  <img alt="Sphinx" src="https://img.shields.io/badge/docs-Sphinx-blue">
  <img alt="GitHub Actions" src="https://img.shields.io/badge/CI-GitHub%20Actions-2088FF">
</p>

---

## Overview

This repository contains the solution developed for the **Sephora Data Science Technical Test**.

The objective is to predict two product attributes from textual information:

- **`nature_code`** — multiclass product-category classification.
- **`target_code`** — binary target classification (`FEMME` / `HOMME`).

The main textual features are:

- `productName`
- `longDescription`

Two approaches are implemented:

1. **Rule-based keyword baseline**
2. **TF-IDF + LinearSVC machine-learning pipeline**

The project is designed as a reusable Python package and includes:

- text preprocessing,
- sklearn-compatible transformers and estimators,
- experiment orchestration,
- model evaluation,
- artifact generation,
- model serialization,
- automated tests,
- code coverage,
- linting and formatting,
- static type checking,
- security analysis,
- pre-commit hooks,
- GitHub Actions CI,
- Sphinx documentation,
- notebooks,
- reproducible dependency management with `uv`.

The goal is not only to train an accurate model, but also to structure the solution as a maintainable and reproducible Machine Learning project.

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

Despite the strong class imbalance in `target_code`, the model reaches:

```text
HOMME Recall = 0.89
HOMME F1     = 0.91
```

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

Using balanced class weights helps the binary classifier better handle the `FEMME` / `HOMME` class imbalance.

---

## Validation strategy

Only the provided training dataset is split.

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

The split is stratified using the combination:

```text
nature_code + target_code
```

This helps preserve the distribution of both prediction targets, including the minority `HOMME` class.

The provided `testing_set.csv` is used **only for final inference** and never for model fitting or parameter selection.

---

## Text preprocessing

The preprocessing pipeline transforms:

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

## Rule-based baseline

A first baseline is implemented using configurable keyword mappings for:

- `nature_code`
- `target_code`

The baseline follows a deterministic strategy using cleaned product names and descriptions.

It provides:

- an interpretable reference solution,
- a simple benchmark,
- a way to inspect discriminative vocabulary,
- deterministic predictions,
- a simple mechanism for incorporating business knowledge.

The rule-based and ML approaches are kept separate so both remain independently reproducible.

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

This helps prevent data leakage.

### LinearSVC model

`LinearSVCModel` inherits from:

```python
BaseEstimator
ClassifierMixin
```

It wraps sklearn's `LinearSVC` while remaining reusable inside an sklearn pipeline.

### Complete ML pipeline

```python
Pipeline(
    steps=[
        ("tfidf", TfidfTransformer(...)),
        ("classifier", LinearSVCModel(...)),
    ]
)
```

This guarantees that the same fitted TF-IDF vocabulary is reused during validation and test inference.

---

## ML experiment configuration

The experiment is configured in `run_ml_pipeline.py`.

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

Balanced Accuracy and Macro F1 are particularly important for `target_code` because of the class imbalance.

---

## Generated artifacts

Each experiment creates a timestamped directory under:

```text
src/artifacts/
```

Example:

```text
2026-08-23_22-50-34/
├── MOUNSEF_DEBACHE_prediction.csv
├── tfidf_linear_svc_nature_confusion_matrix.png
├── tfidf_linear_svc_nature_metrics.txt
├── tfidf_linear_svc_results.pkl
├── tfidf_linear_svc_target_confusion_matrix.png
└── tfidf_linear_svc_target_metrics.txt
```

### Serialized experiment

The complete ML experiment is serialized in:

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

The trained pipelines are therefore preserved alongside configuration, predictions, validation data, and evaluation results.

---

## Submission file

The final prediction file follows the required structure:

```text
productId
nature_predict
target_predict
```

The generated file is:

```text
MOUNSEF_DEBACHE_prediction.csv
```

---

## Project structure

```text
sephora_test/
│
├── .github/
│   └── workflows/
│       ├── ...
│
├── data/
│   ├── training_set.csv
│   └── testing_set.csv
│
├── docs/
│   ├── source/
│   │   ├── index.rst
│   │   ├── guide.rst
│   │   ├── conf.py
│   │   └── api/
│   └── build/
│
├── logs/
│
├── src/
│   ├── artifacts/
│   │   └── <timestamped experiments>/
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
│       ├── unitary_tests/
│       │   └── ...
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
├── .coverage
├── .coveragerc
├── .gitignore
├── .pre-commit-config.yaml
├── .pre-commit-hooks.yaml
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Installation

The project uses **uv** for dependency and environment management.

### 1. Clone the repository

Using SSH:

```bash
git clone git@github.com:Mounsef27/sephora_test.git
cd sephora_test
```

Or HTTPS:

```bash
git clone https://github.com/Mounsef27/sephora_test.git
cd sephora_test
```

### 2. Checkout the development branch

```bash
git switch dev
```

### 3. Synchronize the environment

```bash
uv sync
```

This installs the dependencies declared in:

```text
pyproject.toml
uv.lock
```

### 4. Verify Python

```bash
uv run python --version
```

Python requirement:

```text
Python >= 3.13
```

---

## Quick start

### Run tests

```bash
uv run pytest
```

### Run pre-commit

```bash
uv run pre-commit run --all-files
```

### Run Ruff

```bash
uv run ruff check .
```

### Run formatting

```bash
uv run ruff format .
```

### Run type checking

```bash
uv run mypy src
```

### Run Bandit

```bash
uv run bandit -r src
```

### Build documentation

```bash
uv run sphinx-build -E -W -b html docs/source docs/build/html
```

---

## Notebooks

The project includes notebooks for exploration and experiment execution.

### Exploratory analysis

```text
_static/_notebooks/exploration.ipynb
```

Used for:

- dataset exploration,
- class distributions,
- text analysis,
- preprocessing exploration.

### Rule-based pipeline

```text
_static/_notebooks/package_execution.ipynb
```

### TF-IDF + LinearSVC pipeline

```text
_static/_notebooks/package_execution_svc.ipynb
```

The core ML logic remains inside the Python package rather than being implemented only inside notebooks.

---

## Unit testing

Automated tests are used to verify reusable components and preprocessing behavior.

Run:

```bash
uv run pytest
```

or:

```bash
uv run pytest -v
```

Tests are located under:

```text
src/sephora_test/unitary_tests/
```

---

## Code coverage

Coverage can be generated with:

```bash
uv run coverage run -m pytest
```

Display the terminal report:

```bash
uv run coverage report -m
```

Generate the HTML report:

```bash
uv run coverage html
```

The generated report is available under:

```text
htmlcov/
```

---

## Code quality

The project uses several complementary tools.

### Ruff

Ruff is used for:

- linting,
- import ordering,
- formatting,
- common Python error detection.

Run:

```bash
uv run ruff check .
```

Apply formatting:

```bash
uv run ruff format .
```

The project uses configuration defined in `pyproject.toml`.

Example:

```toml
[tool.ruff]
preview = true
line-length = 80

[tool.ruff.lint]
select = ["E", "F", "I", "B"]
```

---

## Static type checking

The project uses **mypy** for static type analysis.

Run:

```bash
uv run mypy src
```

Static typing helps detect potential inconsistencies before runtime.

---

## Security checks

The project uses **Bandit** for Python security analysis.

Run:

```bash
uv run bandit -r src
```

Bandit helps identify potentially unsafe patterns such as:

- unsafe subprocess usage,
- hardcoded secrets,
- risky deserialization,
- insecure temporary file usage,
- other security-sensitive Python patterns.

---

## Pre-commit hooks

The repository uses **pre-commit** to automatically run quality checks.

Install hooks with:

```bash
uv run pre-commit install
```

Run all configured hooks:

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
✓ Bandit security checks
```

This helps detect issues before code is pushed to GitHub.

---

## Continuous Integration

GitHub Actions workflows are stored under:

```text
.github/workflows/
```

The CI pipeline automatically validates the project when changes are pushed.

Conceptually:

```text
Developer
    │
    ▼
git commit
    │
    ▼
pre-commit
    │
    ▼
git push
    │
    ▼
GitHub Actions
    │
    ├── Environment setup
    ├── Dependency installation
    ├── Pre-commit checks
    ├── Unit tests
    └── Documentation build
```

The CI workflow uses `uv` to reproduce the development environment.

Typical CI commands include:

```bash
uv python install
uv sync --locked
uv run pre-commit run --all-files
uv run pytest --maxfail=1 --disable-warnings -q
```

Documentation can also be validated in CI using:

```bash
uv run sphinx-build -E -W -b html docs/source docs/build/html
```

Using `-W` turns documentation warnings into build errors.

---

## Dependency review

The repository includes dependency review as an additional security mechanism.

Dependency changes can introduce:

- vulnerabilities,
- incompatible versions,
- unexpected transitive dependencies.

Dependency review helps identify these issues before changes are merged.

---

## Documentation

Technical documentation is built with **Sphinx**.

Documentation sources are located under:

```text
docs/source/
```

Build the documentation with:

```bash
uv run sphinx-build -E -W -b html docs/source docs/build/html
```

Generated HTML documentation is available under:

```text
docs/build/html/
```

The documentation includes:

- project overview,
- quick start,
- usage guide,
- API documentation,
- module references.

---

## Main dependencies

| Package / Tool | Purpose |
|---|---|
| pandas | Data manipulation |
| NumPy | Numerical operations |
| scikit-learn | TF-IDF, pipelines, LinearSVC, metrics |
| BeautifulSoup4 | HTML cleaning |
| lxml | HTML parsing |
| NLTK | French stopwords and tokenization |
| Matplotlib | Confusion-matrix visualization |
| dill / pickle | Experiment serialization |
| pytest | Unit testing |
| coverage | Test coverage |
| Ruff | Linting and formatting |
| mypy | Static type checking |
| Bandit | Security analysis |
| pre-commit | Automated quality checks |
| Sphinx | Technical documentation |
| GitHub Actions | Continuous Integration |
| uv | Environment and dependency management |

---

## Reproducibility

The project emphasizes reproducibility through:

- fixed random seed,
- deterministic train-validation assignment,
- sklearn pipelines,
- TF-IDF fitted only on training data,
- serialized fitted models,
- timestamped experiment folders,
- saved configurations,
- saved validation metrics,
- saved confusion matrices,
- versioned dependencies with `uv.lock`,
- automated tests,
- CI validation.

This makes it easier to trace:

```text
Code
  +
Dependencies
  +
Configuration
  +
Model
  +
Metrics
  +
Predictions
```

for each experiment.

---

## Engineering principles

### Separation of concerns

The project separates:

```text
Preprocessing
     ≠
Model
     ≠
Evaluation
     ≠
Experiment orchestration
```

### Reusable components

Custom components follow the sklearn API when appropriate.

### Data leakage prevention

TF-IDF is fitted only on the training subset.

### Experiment traceability

Models, metrics, predictions, and configurations are saved as artifacts.

### Automated quality control

Testing, linting, formatting, security analysis, documentation checks, and CI are integrated into the development workflow.

### Reproducible environments

Dependencies are managed using:

```text
pyproject.toml
uv.lock
```

### Documentation as code

Documentation lives alongside the source code and is automatically buildable with Sphinx.

---

## Development workflow

A typical development workflow is:

```bash
git switch dev
git pull origin dev
git switch -c feat/my-feature
```

Develop the feature and then run:

```bash
uv run pytest
uv run pre-commit run --all-files
```

Commit:

```bash
git add .
git commit -m "feat: add my feature"
```

Push:

```bash
git push origin feat/my-feature
```

The feature can then be integrated through a Pull Request.

---

## Potential improvements

Several extensions could further improve the project.

### Modeling

- Tune the `LinearSVC` regularization parameter `C`.
- Compare word-level and character-level TF-IDF.
- Combine word-level and character-level n-grams.
- Use independent representations for `productName` and `longDescription`.
- Give higher weight to `productName`.
- Compare Logistic Regression.
- Compare Naive Bayes.
- Explore transformer embeddings.
- Fine-tune a French language model.

### Validation

- Add cross-validation.
- Add repeated stratified validation.
- Perform systematic error analysis.
- Analyze errors for `COFFRET`, `CREM JOUR`, and `SERUM`.

### Class imbalance

- Tune class weights.
- Analyze minority-class errors.
- Compare alternative imbalance strategies.

### MLOps

- Add MLflow experiment tracking.
- Add formal model versioning.
- Containerize the project with Docker.
- Add a FastAPI inference service.
- Add deployment workflows.
- Add data-drift monitoring.
- Add model-performance monitoring.

---

## Current limitations

### `COFFRET`

The `COFFRET` category has lower recall than most other classes.

### TF-IDF representation

TF-IDF captures lexical importance but has limited semantic representation compared with modern transformer-based approaches.

### Validation strategy

The current holdout validation split is reproducible and efficient, but cross-validation could provide a more robust estimate of generalization performance.

---

## Development philosophy

The project evolves from a simple Data Science experiment toward:

```text
Exploration
     │
     ▼
Notebook
     │
     ▼
Reusable Python package
     │
     ├── Tests
     ├── Documentation
     ├── Security checks
     ├── Code quality
     ├── CI
     └── Reproducible experiments
```

The objective is to demonstrate not only Machine Learning performance, but also the software-engineering practices required to maintain a real-world ML project.

---

## Author

**Mounsef Debache**

Data Science · Machine Learning · Artificial Intelligence · NLP

GitHub: [Mounsef27](https://github.com/Mounsef27)

---

<p align="center">
  <strong>Sephora Data Science Technical Test</strong>
</p>

<p align="center">
  Machine Learning · NLP · Software Engineering · MLOps
</p>
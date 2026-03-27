# Quick Start

## 1. Set Up Virtual Environment

Create and activate a virtual environment before installing dependencies.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Add Dataset

Place the following files in the specified directories:

* `client_*.csv` and `invoice_*.csv` → `data/raw/`
* `SamplesSubmission (2).csv` → `sample_submission/`

### Notes

* The provided **test dataset is unlabeled**.
* It will go through the processing pipeline but **is not used for model training**.
* The **train/test split used for modelling is created from the training dataset only**.

---

## 3. Run Data Pipeline

Run the scripts below to aggregate invoices and generate the modelling dataset.

### Option A — Full Feature Set (Default)

```bash
python src/data/aggregate_invoices.py
python src/data/preprocess.py
```

* Uses the **complete set of engineered features**
* Recommended as a **baseline feature configuration**


### Option B — Reduced Feature Set (Drop Low-Importance Features)

```bash
python src/data/aggregate_invoices.py
python src/data/preprocess.py --drop_low_importance
```

* Removes features identified as **low importance**



## 4. Run Experiments

All models are trained through the experiment runner.

### Default (No SMOTE)

```bash
python -m src.experiments.run_experiment \
  --config configs/<model>.yaml \
  --name <experiment_name>
```

Example:

```bash
python -m src.experiments.run_experiment \
  --config configs/knn.yaml \
  --name knn_baseline
```

### Enable SMOTE

To apply SMOTE during training, use the `--smote` flag:

```bash
python -m src.experiments.run_experiment \
  --config configs/<model>.yaml \
  --name <experiment_name> \
  --smote
```

Example:

```bash
python -m src.experiments.run_experiment \
  --config configs/logreg.yaml \
  --name logreg_smote \
  --smote
```

## Notes on Imbalanced Learning

* **SMOTE is disabled by default** to provide a clean baseline
* When enabled, SMOTE is applied **only to training folds** (no data leakage)
* Validation and test sets always retain the **original class distribution**

Additionally:

* Logistic Regression, SVM, and Decision Tree support `class_weight` (configured in YAML)
* KNN does **not** support class weighting → relies on SMOTE for imbalance handling


## Results

Results will be saved automatically in the `results/` directory.

Each experiment includes:

* Evaluation metrics
* Best hyperparameters
* ROC and PR curves
* Trained model artifact

---

# Project Structure

```
configs/            Model configs (hyperparameters + CV settings)

src/
  data/             Data processing scripts
  models/           Model definitions
  evaluation/       Metrics and plotting utilities
  experiments/      Main experiment runner
  utils/            Helper utilities

results/            Saved experiment outputs
```
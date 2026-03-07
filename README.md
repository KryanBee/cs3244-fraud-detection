# Quick Start

## 1. Set Up Virtual Environment

Create and activate a virtual environment before installing dependencies.

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Add Dataset

Place the following files in the specified directories:

* `client_*.csv` and `invoice_*.csv` → `data/raw/`
* `SamplesSubmission (2).csv` → `sample_submission/`

Note:

* The provided **test dataset is unlabeled**.
* It will go through the processing pipeline but **is not used for model training**.
* The **train/test split used for modelling is created from the training dataset only**.

---

## 3. Run Data Pipeline

Run the scripts below to aggregate invoices and generate the modelling dataset.

```
python src/data/aggregate_invoices.py
python src/data/preprocess.py
```

These scripts will prepare the feature dataset and create the training/testing splits used for experiments.

---

## 4. Run Experiments

All models are trained through the experiment runner.

```
python -m src.experiments.run_experiment --config configs/<model>.yaml --name <experiment_name>
```

Example:

```
python -m src.experiments.run_experiment --config configs/knn.yaml --name knn_baseline
```

Results will be saved automatically in the `results/` directory.

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

# Introduction
This project explores machine learning approaches for electricity fraud detection under severe class imbalance. The repository is designed to support running controlled experiments across different models and configurations. Only the final XGBoost models reported in the study are saved, as the full set of experiments is too large to store.

# Project Structure
```
configs/            Model configs (hyperparameters + CV settings)

data/
  raw/              Raw input data
  processed/        Processed datasets

documents/          Report and supporting materials

notebooks/          Exploratory analysis

results/            Saved experiment outputs

src/
  data/             Data processing and feature engineering
  models/           Model implementations
  evaluation/       Metrics and evaluation utilities
  experiments/      Experiment runner
  utils/            Helper functions
```

# Steps
## 1. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Add Dataset
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mrmorj/fraud-detection-in-electricity-and-gas-consumption) and place the files in the following directories. The test set is unlabeled and is processed but not used for model training.

* `client_*.csv` and `invoice_*.csv` → `data/raw/`
* `SamplesSubmission (2).csv` → `sample_submission/`

## 3. Run Data Pipeline
### Generate Features
```
python src/data/aggregate_invoices.py
python src/data/preprocess.py

# Optional: drop low-importance features
python src/data/preprocess.py --drop_low_importance
```

### Run Experiments

```
python -m src.experiments.run_experiment \
  --config configs/<model>.yaml \
  --name <experiment_name>

# Optional: enable SMOTE
python -m src.experiments.run_experiment \
  --config configs/<model>.yaml \
  --name <experiment_name> \
  --smote
```

### View Results
All experiment outputs are stored in the `results/` directory.
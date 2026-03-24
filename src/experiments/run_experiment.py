import argparse
import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline

from src.utils.config import load_config
from src.models.logreg_model import LogRegModel
from src.models.knn_model import KNNModel
from src.models.dt_model import DecisionTreeModel
from src.models.svm_model import SVMModel

from src.evaluation.metrics import compute_metrics, find_best_threshold
from src.evaluation.plots import plot_roc, plot_pr
from src.utils.experiment import create_experiment_dir, save_metrics, save_config


def load_data(cfg):
    # Load preprocessed train and test sets
    X_train = pd.read_csv(cfg["data"]["x_train"])
    y_train = pd.read_csv(cfg["data"]["y_train"]).values.ravel()

    X_test = pd.read_csv(cfg["data"]["x_test"])
    y_test = pd.read_csv(cfg["data"]["y_test"]).values.ravel()

    return X_train, y_train, X_test, y_test


def run(config_path, name):
    cfg = load_config(config_path)
    exp_dir = create_experiment_dir(name)
    X_train, y_train, X_test, y_test = load_data(cfg)

    # Instantiate model based on config
    if cfg["model"] == "logreg":
        model = LogRegModel()
    elif cfg["model"] == "knn":
        model = KNNModel()
    elif cfg["model"] == "dt":
        model = DecisionTreeModel()
    elif cfg["model"] == "svm":
        model = SVMModel()
    else:
        raise ValueError("Unknown model type")

    # Create pipeline with BorderlineSMOTE + model to handle class imbalance within CV
    pipeline = Pipeline([
        ('smote', BorderlineSMOTE(random_state=42)),
        ('classifier', model.model)
    ])

    # Prefix parameter names with 'classifier__' to work with the pipeline
    param_grid = {f"classifier__{k}": v for k, v in cfg["params"].items()}

    # Hyperparameter tuning via grid search with cross-validation
    # BorderlineSMOTE is applied within each fold to avoid data leakage
    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cfg["search"]["cv"],
        scoring=cfg["search"]["scoring"],
        n_jobs=-1,   # Change to -1 to use ALL your CPU cores
        verbose=3,   # Change to 3 so you see progress in the terminal
        return_train_score=True
    )
    grid.fit(X_train, y_train)

    # Get best model performance
    best_index = grid.best_index_
    cv_mean = grid.cv_results_["mean_test_score"][best_index]
    cv_std = grid.cv_results_["std_test_score"][best_index]
    best_model = grid.best_estimator_

    # Find optimal threshold on training data, then evaluate on test
    train_probs = best_model.predict_proba(X_train)[:, 1]
    best_threshold = find_best_threshold(y_train, train_probs)
    test_probs = best_model.predict_proba(X_test)[:, 1]
    results = compute_metrics(y_test, test_probs, best_threshold)

    # Include CV stats in results
    results["cv_mean_pr_auc"] = cv_mean
    results["cv_std_pr_auc"] = cv_std
    results["best_params"] = grid.best_params_

    # Save outputs
    save_metrics(results, exp_dir)
    save_config(cfg, exp_dir)
    joblib.dump(best_model, f"{exp_dir}/model.joblib")
    plot_roc(y_test, test_probs, f"{exp_dir}/roc_curve.png")
    plot_pr(y_test, test_probs, f"{exp_dir}/pr_curve.png")

    # Display results
    print("\nBest Hyperparameters:")
    print(grid.best_params_)
    print("\nCross-Validation PR-AUC:")
    print(f"Mean: {cv_mean:.4f}")
    print(f"Std : {cv_std:.4f}")
    print("\nBest Threshold:")
    print(f"{best_threshold:.4f}")
    print("\nTest Set Results:")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--name", type=str, default="experiment")

    args = parser.parse_args()

    run(args.config, args.name)

# python -m src.experiments.run_experiment --config configs/logreg.yaml --name logreg_
# python -m src.experiments.run_experiment --config configs/knn.yaml --name knn_
# python -m src.experiments.run_experiment --config configs/dt.yaml --name dt_
# python -m src.experiments.run_experiment --config configs/svm.yaml --name svm_
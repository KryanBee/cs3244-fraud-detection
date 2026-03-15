import argparse
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(
    input_path: Path = Path("data/processed/aggregate_train.csv"),
    output_dir: Path = Path("data/input"),
    label_col: str = "target",
    test_size: float = 0.2,
    seed: int = 3244,
    drop_cols: list = None,
):
    """
    Preprocesses aggregated features for model training.

    Steps:
    1. Load aggregated client features
    2. Remove irrelevant columns (ID fields)
    3. One-hot encode categorical variables
    4. Stratified train-test split (preserves class distribution)
    5. Standardize features using training set statistics
    6. Save processed splits to CSV

    The standardization is fit exclusively on training data to avoid
    test-set leakage when scaling is applied to both splits.
    """

    if drop_cols is None:
        drop_cols = ["client_id", "creation_date"]

    df = pd.read_csv(input_path)

    # Drop non-predictive columns
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Separate labels and features
    y = df[label_col].astype(int)
    X = df.drop(columns=[label_col])

    # Explicit categorical columns
    cat_cols = ["disrict", "client_catg", "region"]

    # One-hot encode categorical variables
    X = pd.get_dummies(X, columns=[c for c in cat_cols if c in X.columns])

    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Reset indices to avoid inherited indexing
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Fit scaler on training data only to avoid test leakage
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrames with original column names
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Export processed datasets
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)

    print("Saved processed datasets to:", output_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default="data/processed/aggregate_train.csv",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default="data/input",
    )

    parser.add_argument(
        "--label_col",
        type=str,
        default="target",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=3244,
    )

    parser.add_argument(
        "--drop_cols",
        nargs="+",
        default=["client_id", "creation_date"],
    )

    args = parser.parse_args()

    preprocess(
        args.input,
        args.output,
        args.label_col,
        args.test_size,
        args.seed,
        args.drop_cols,
    )

# python src/data/preprocess.py


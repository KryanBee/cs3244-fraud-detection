import pandas as pd
import numpy as np


def aggregate_invoices(invoice_path, client_path):
    """
    Very naive aggregation:
    - numeric columns -> mean
    - categorical columns -> mode
    - date columns -> dropped
    """

    invoices = pd.read_csv(invoice_path)
    clients = pd.read_csv(client_path)

    # Drop date columns
    date_cols = ["invoice_date"]
    invoices = invoices.drop(columns=date_cols, errors="ignore")

    # Separate numeric and categorical columns
    numeric_cols = invoices.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = invoices.select_dtypes(exclude=np.number).columns.tolist()

    # Remove client_id from aggregation lists
    numeric_cols = [c for c in numeric_cols if c != "client_id"]
    categorical_cols = [c for c in categorical_cols if c != "client_id"]

    # Aggregate numeric features using mean
    numeric_agg = invoices.groupby("client_id")[numeric_cols].mean()

    # Aggregate categorical features using mode
    def mode_func(x):
        if len(x.mode()) > 0:
            return x.mode()[0]
        return np.nan

    categorical_agg = invoices.groupby("client_id")[categorical_cols].agg(mode_func)

    # Combine aggregated features
    agg = pd.concat([numeric_agg, categorical_agg], axis=1).reset_index()

    # Merge aggregated transaction features with client metadata
    df = clients.merge(agg, on="client_id", how="left")

    return df


if __name__ == "__main__":

    print("Aggregating training data...")
    train = aggregate_invoices(
        "data/raw/invoice_train.csv",
        "data/raw/client_train.csv"
    )

    print("Aggregating test data...")
    test = aggregate_invoices(
        "data/raw/invoice_test.csv",
        "data/raw/client_test.csv"
    )

    # Save outputs
    train.to_csv("data/processed/aggregate_train.csv", index=False)
    test.to_csv("data/processed/aggregate_test.csv", index=False)

    print("Aggregation complete.")

# python src/data/aggregate_invoices.py
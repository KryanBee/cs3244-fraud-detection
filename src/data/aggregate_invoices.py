import pandas as pd
import numpy as np


def aggregate_invoices(invoice_path, client_path):
    """
    Enhanced aggregation:
    - numeric columns -> mean, std, max, min, sum
    - categorical columns -> mode, unique count
    - date columns -> converted to features (e.g., number of months)
    """

    invoices = pd.read_csv(invoice_path)
    clients = pd.read_csv(client_path)

    # Convert date to datetime
    invoices["invoice_date"] = pd.to_datetime(invoices["invoice_date"], dayfirst=True)
    invoices["invoice_year"] = invoices["invoice_date"].dt.year

    # Feature Engineering: Total consumption
    invoices["total_consumption"] = (invoices["consommation_level_1"] + 
                                     invoices["consommation_level_2"] + 
                                     invoices["consommation_level_3"] + 
                                     invoices["consommation_level_4"])

    # Basic numeric features
    numeric_cols = ["consommation_level_1", "consommation_level_2", 
                    "consommation_level_3", "consommation_level_4",
                    "total_consumption"]
    
    # Aggregations for numeric columns
    agg_funcs = {
        col: ["mean", "std", "max", "min", "sum"] for col in numeric_cols
    }
    
    # Add count of invoices and unique counts for some categories
    agg_funcs["invoice_date"] = ["count"]
    agg_funcs["invoice_year"] = ["nunique"]
    agg_funcs["tarif_type"] = ["nunique"]
    agg_funcs["counter_statue"] = ["nunique"]
    agg_funcs["counter_code"] = ["nunique"]

    # Perform aggregation
    agg = invoices.groupby("client_id").agg(agg_funcs)
    
    # Flatten multi-index columns
    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    # Feature Engineering: Counter Type specific aggregations
    # (Electricity vs Gas)
    for c_type in ["ELEC", "GAZ"]:
        type_df = invoices[invoices["counter_type"] == c_type]
        type_agg = type_df.groupby("client_id")["consommation_level_1"].agg(["sum", "mean"]).reset_index()
        type_agg.columns = ["client_id", f"{c_type}_sum", f"{c_type}_mean"]
        agg = agg.merge(type_agg, on="client_id", how="left").fillna(0)

    # Merge aggregated transaction features with client metadata
    df = clients.merge(agg, on="client_id", how="left")

    # Feature Engineering: Convert creation_date to numeric (days since a reference date)
    df["creation_date"] = pd.to_datetime(df["creation_date"], dayfirst=True)
    ref_date = df["creation_date"].max()
    df["days_since_creation"] = (ref_date - df["creation_date"]).dt.days
    df = df.drop(columns=["creation_date"])

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
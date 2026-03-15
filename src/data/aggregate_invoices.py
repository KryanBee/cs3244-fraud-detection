import pandas as pd
import numpy as np


def aggregate_invoices(invoice_path, client_path):

    invoices = pd.read_csv(invoice_path)
    clients = pd.read_csv(client_path)

    # =====================================================
    # 1. CLIENT DATA PREPROCESSING
    # =====================================================

    clients["creation_date"] = pd.to_datetime(clients["creation_date"], dayfirst=True)

    ref_date = pd.Timestamp("2019-12-31")

    # Account age features
    clients["days_since_creation"] = (
        ref_date - clients["creation_date"]
    ).dt.days

    clients["months_active"] = clients["days_since_creation"] / 30

    # =====================================================
    # 2. INVOICE DATE FEATURES
    # =====================================================

    invoices["invoice_date"] = pd.to_datetime(invoices["invoice_date"], dayfirst=True)

    invoices["invoice_year"] = invoices["invoice_date"].dt.year
    invoices["invoice_month"] = invoices["invoice_date"].dt.month
    invoices["is_weekend"] = (invoices["invoice_date"].dt.dayofweek >= 5).astype(int)

    # =====================================================
    # 3. CLEAN DIRTY FEATURES
    # =====================================================

    invoices["counter_statue"] = invoices["counter_statue"].replace({
        'A': 0, 618: 5, 269375: 5, 420: 5, 46: 5, 769: 5
    })

    # =====================================================
    # 4. INVOICE LEVEL ENGINEERED FEATURES
    # =====================================================

    invoices["total_consumption"] = (
        invoices["consommation_level_1"]
        + invoices["consommation_level_2"]
        + invoices["consommation_level_3"]
        + invoices["consommation_level_4"]
    )

    invoices["delta_index"] = invoices["new_index"] - invoices["old_index"]

    invoices = invoices.sort_values(["client_id", "invoice_date"])

    # time between invoices
    invoices["delta_time"] = (
        invoices.groupby("client_id")["invoice_date"]
        .diff()
        .dt.days
    )

    # consumption change
    invoices["consumption_change"] = (
        invoices.groupby("client_id")["total_consumption"]
        .diff()
    )

    # =====================================================
    # 5. AGGREGATION FEATURES
    # =====================================================

    numeric_cols = [
        "consommation_level_1",
        "consommation_level_2",
        "consommation_level_3",
        "consommation_level_4",
        "total_consumption",
        "delta_index",
        "delta_time",
        "consumption_change",
        "reading_remarque",
        "counter_coefficient",
        "months_number"
    ]

    agg_funcs = {
        col: ["mean", "std", "max", "min", "sum"]
        for col in numeric_cols
    }

    # transaction behaviour
    agg_funcs["invoice_date"] = ["count"]
    agg_funcs["invoice_year"] = ["nunique"]

    # diversity features (fraudsters often change meter settings)
    agg_funcs["tarif_type"] = ["nunique"]
    agg_funcs["counter_statue"] = ["nunique"]
    agg_funcs["counter_code"] = ["nunique"]
    agg_funcs["counter_number"] = ["nunique"]

    # temporal behaviour
    agg_funcs["invoice_month"] = ["mean", "std", "max", "min"]
    agg_funcs["is_weekend"] = ["mean"]

    # =====================================================
    # 6. CLIENT LEVEL AGGREGATION
    # =====================================================

    agg = invoices.groupby("client_id").agg(agg_funcs)

    agg.columns = ["_".join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    agg.rename(columns={"invoice_date_count": "transactions_count"}, inplace=True)

    # =====================================================
    # 7. ELECTRICITY vs GAS FEATURES
    # =====================================================

    for c_type in ["ELEC", "GAZ"]:

        type_df = invoices[invoices["counter_type"] == c_type]

        type_agg = (
            type_df.groupby("client_id")["total_consumption"]
            .agg(["sum", "mean", "std"])
            .reset_index()
        )

        type_agg.columns = [
            "client_id",
            f"{c_type}_consumption_sum",
            f"{c_type}_consumption_mean",
            f"{c_type}_consumption_std",
        ]

        agg = agg.merge(type_agg, on="client_id", how="left")

    agg.fillna(0, inplace=True)

    # =====================================================
    # 8. VOLATILITY FEATURES
    # =====================================================

    range_cols = [
        "consommation_level_1",
        "consommation_level_2",
        "consommation_level_3",
        "consommation_level_4",
        "total_consumption",
        "delta_index"
    ]

    for col in range_cols:
        agg[f"{col}_range"] = agg[f"{col}_max"] - agg[f"{col}_min"]

    # max / mean anomaly ratios
    for col in range_cols:

        agg[f"{col}_max_mean_ratio"] = (
            agg[f"{col}_max"] /
            (agg[f"{col}_mean"] + 1e-6)
        )

    # =====================================================
    # 9. MERGE CLIENT FEATURES
    # =====================================================

    df = clients.merge(agg, on="client_id", how="left")

    # invoice frequency
    df["invoice_frequency"] = (
        df["transactions_count"] /
        (df["months_active"] + 1e-6)
    )

    df.drop(columns=["creation_date"], inplace=True)

    return df


if __name__ == "__main__":

    print("Aggregating training data...")

    train = aggregate_invoices(
        "data/raw/invoice_train.csv",
        "data/raw/client_train.csv",
    )

    print("Aggregating test data...")

    test = aggregate_invoices(
        "data/raw/invoice_test.csv",
        "data/raw/client_test.csv",
    )

    # Save processed datasets
    train.to_csv("data/processed/aggregate_train.csv", index=False)
    test.to_csv("data/processed/aggregate_test.csv", index=False)

    print("Aggregation complete.")

# python src/data/aggregate_invoices.py
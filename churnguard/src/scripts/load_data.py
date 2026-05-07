"""Chargement et préparation du jeu de données Telco."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Charge le jeu de données Telco et retourne (X, y) après prétraitement de base."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    print(f"[info] Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    # Handle numeric conversion (TotalCharges may have spaces)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    initial_rows = len(df)
    df = df.dropna()
    print(f"[info] Dropped {initial_rows - len(df)} rows with missing values")

    # Drop identifier
    df = df.drop(columns=["customerID"])

    # Target: Churn (Yes/No -> 1/0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    print(f"[ok] Dataset ready: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[info] Class distribution: {y.value_counts().to_dict()}")

    return X, y


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Load and prepare Telco dataset")
    parser.add_argument("--csv-path", default="data/telco_churn.csv")
    args = parser.parse_args()

    load_data(args.csv_path)

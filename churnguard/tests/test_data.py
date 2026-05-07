"""Tests unitaires pour le module load_data."""
from __future__ import annotations

import pandas as pd

from src.scripts.load_data import load_data

CSV_PATH = "data/telco_churn.csv"

EXPECTED_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def test_load_data_returns_dataframe():
    """load_data retourne bien un DataFrame et une Serie."""
    X, y = load_data(CSV_PATH)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_load_data_has_expected_columns():
    """Le DataFrame contient les 19 colonnes features attendues."""
    X, _ = load_data(CSV_PATH)
    for col in EXPECTED_FEATURE_COLUMNS:
        assert col in X.columns, f"Colonne manquante : {col}"


def test_load_data_drops_identifiers():
    """customerID et Churn ne doivent pas etre dans X."""
    X, _ = load_data(CSV_PATH)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns


def test_load_data_shapes_consistent():
    """X et y ont le meme nombre de lignes, non vide."""
    X, y = load_data(CSV_PATH)
    assert len(X) == len(y)
    assert len(X) > 0


def test_load_data_handles_missing_total_charges():
    """TotalCharges (qui contient des espaces) est converti en float sans NaN."""
    X, _ = load_data(CSV_PATH)
    assert X["TotalCharges"].dtype in (float, "float64")
    assert not X["TotalCharges"].isna().any()


def test_load_data_target_binary():
    """La cible y ne contient que des valeurs 0 et 1."""
    _, y = load_data(CSV_PATH)
    assert set(y.unique()).issubset({0, 1})

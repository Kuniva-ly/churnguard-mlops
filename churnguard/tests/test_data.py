"""Tests unitaires pour le module load_data."""
from __future__ import annotations

import pandas as pd
import pytest

from src.scripts.load_data import load_data

EXPECTED_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


@pytest.fixture
def telco_csv(tmp_path) -> str:
    """CSV minimal au format Telco écrit dans un fichier temporaire."""
    rows = [
        {
            "customerID": f"CUS-{i:04d}",
            "gender": "Female" if i % 2 == 0 else "Male",
            "SeniorCitizen": i % 2,
            "Partner": "Yes" if i % 3 == 0 else "No",
            "Dependents": "No",
            "tenure": i % 72,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 50.0 + i,
            "TotalCharges": str(600.0 + i * 10) if i % 10 != 0 else " ",
            "Churn": "Yes" if i % 4 == 0 else "No",
        }
        for i in range(1, 51)
    ]
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "telco_churn.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_data_returns_dataframe(telco_csv):
    """load_data retourne bien un DataFrame et une Serie."""
    X, y = load_data(telco_csv)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_load_data_has_expected_columns(telco_csv):
    """Le DataFrame contient les 19 colonnes features attendues."""
    X, _ = load_data(telco_csv)
    for col in EXPECTED_FEATURE_COLUMNS:
        assert col in X.columns, f"Colonne manquante : {col}"


def test_load_data_drops_identifiers(telco_csv):
    """customerID et Churn ne doivent pas etre dans X."""
    X, _ = load_data(telco_csv)
    assert "customerID" not in X.columns
    assert "Churn" not in X.columns


def test_load_data_shapes_consistent(telco_csv):
    """X et y ont le meme nombre de lignes, non vide."""
    X, y = load_data(telco_csv)
    assert len(X) == len(y)
    assert len(X) > 0


def test_load_data_handles_missing_total_charges(telco_csv):
    """TotalCharges (qui contient des espaces) est converti en float sans NaN."""
    X, _ = load_data(telco_csv)
    assert X["TotalCharges"].dtype in (float, "float64")
    assert not X["TotalCharges"].isna().any()


def test_load_data_target_binary(telco_csv):
    """La cible y ne contient que des valeurs 0 et 1."""
    _, y = load_data(telco_csv)
    assert set(y.unique()).issubset({0, 1})

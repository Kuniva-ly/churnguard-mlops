"""Tests unitaires pour le module load_data et download_data."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock, patch

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


# --- Tests download_data ---

def test_sha256_of(tmp_path):
    """sha256_of retourne le bon digest SHA-256."""
    import src.scripts.download_data as dd
    content = b"hello churnguard"
    f = tmp_path / "file.bin"
    f.write_bytes(content)
    assert dd.sha256_of(f) == hashlib.sha256(content).hexdigest()


def test_download_skips_valid_file(tmp_path, monkeypatch):
    """download() ne re-télécharge pas si le fichier est déjà valide."""
    import src.scripts.download_data as dd
    dest = tmp_path / "telco_churn.csv"
    dest.write_bytes(b"fake")
    monkeypatch.setattr(dd, "DEST", dest)
    monkeypatch.setattr(dd, "sha256_of", lambda p: dd.EXPECTED_SHA256)
    result = dd.download()
    assert result == dest


def test_download_redownloads_bad_local_checksum(tmp_path, monkeypatch):
    """download() re-télécharge si le checksum local ne correspond pas."""
    import src.scripts.download_data as dd
    dest = tmp_path / "telco_churn.csv"
    dest.write_bytes(b"corrupted")
    monkeypatch.setattr(dd, "DEST", dest)

    mock_resp = MagicMock()
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.read.return_value = b"wrong"

    with patch("urllib.request.urlopen", return_value=mock_resp):
        with pytest.raises(SystemExit) as exc:
            dd.download()
    assert exc.value.code == 2


def test_download_exits_on_network_error(tmp_path, monkeypatch):
    """download() quitte avec code 1 si le réseau échoue."""
    import src.scripts.download_data as dd
    dest = tmp_path / "telco_churn.csv"
    monkeypatch.setattr(dd, "DEST", dest)

    with patch("urllib.request.urlopen", side_effect=OSError("network error")):
        with pytest.raises(SystemExit) as exc:
            dd.download()
    assert exc.value.code == 1

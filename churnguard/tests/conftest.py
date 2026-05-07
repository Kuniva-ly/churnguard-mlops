"""Fixtures partagées entre tous les modules de tests."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest




@pytest.fixture(scope="session")
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Dataset Telco minimal (200 lignes) généré de façon déterministe."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(0, 72, n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"], n
        ),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
        "TotalCharges": rng.uniform(0, 8000, n).round(2),
    })
    y = pd.Series(rng.integers(0, 2, n).astype(int), name="Churn")
    return X, y




VALID_PAYLOAD: dict = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2,
}


@pytest.fixture
def mock_bundle() -> MagicMock:
    """Bundle mocké : predict_proba retourne [[0.3, 0.7]]."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return MagicMock(model=model, version="test-v1")

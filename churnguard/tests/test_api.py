"""Tests unitaires de l API ChurnGuard.

On mocke le bundle modele pour isoler l API de l infrastructure MLflow/pkl.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import VALID_PAYLOAD

# Credentials de test
AUTH = ("admin", "changeme")

# Payload batch (POST /predict/batch)
BATCH_PAYLOAD = {"data": [VALID_PAYLOAD, VALID_PAYLOAD]}


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    """Injecte les variables d environnement pour les tests."""
    monkeypatch.setenv("API_USERNAME", AUTH[0])
    monkeypatch.setenv("API_PASSWORD", "changeme")
    monkeypatch.setenv("API_VERSION", "1.0.0")
    monkeypatch.setenv("MODEL_NAME", "churn_model_v1")
    monkeypatch.setenv("CHURN_THRESHOLD", "0.5")


@pytest.fixture
def client(mock_bundle):
    """TestClient avec le bundle mocke injecte."""
    with patch("src.api.model_loader._bundle", mock_bundle):
        from main import app
        yield TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def client_no_model():
    """TestClient sans modele charge (mode degrade)."""
    with patch("src.api.model_loader._bundle", None):
        from main import app
        yield TestClient(app, raise_server_exceptions=True)


# Tests /health

def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_health_degraded(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


def test_health_no_auth_required(client):
    response = client.get("/health")
    assert response.status_code == 200


# Tests /predict

def test_predict_returns_200(client):
    response = client.post("/predict", json=VALID_PAYLOAD, auth=AUTH)
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json=VALID_PAYLOAD, auth=AUTH)
    body = response.json()
    assert "churn" in body
    assert "probability" in body
    assert isinstance(body["churn"], bool)
    assert isinstance(body["probability"], float)


def test_predict_probability_range(client):
    response = client.post("/predict", json=VALID_PAYLOAD, auth=AUTH)
    proba = response.json()["probability"]
    assert 0.0 <= proba <= 1.0


def test_predict_accepts_flat_json(client):
    response = client.post("/predict", json=VALID_PAYLOAD, auth=AUTH)
    assert response.status_code == 200


def test_predict_multiple_observations(client, mock_bundle):
    """POST /predict/batch retourne autant de predictions que de clients envoyes."""
    mock_bundle.model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
    response = client.post("/predict/batch", json=BATCH_PAYLOAD, auth=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    assert "churn" in body["predictions"][0]


def test_predict_401_without_auth(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 401


def test_predict_401_wrong_password(client):
    response = client.post("/predict", json=VALID_PAYLOAD, auth=("admin", "wrong"))
    assert response.status_code == 401


def test_predict_503_no_model(client_no_model):
    response = client_no_model.post("/predict", json=VALID_PAYLOAD, auth=AUTH)
    assert response.status_code == 503


# Tests /version

def test_version_requires_auth(client):
    response = client.get("/version")
    assert response.status_code == 401


def test_version_returns_info(client):
    response = client.get("/version", auth=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert "api_version" in body
    assert "model_name" in body


# Tests model_loader

def test_load_bundle_from_local_pickle(tmp_path, monkeypatch):
    """load_bundle charge un pickle local et retourne un ModelBundle valide."""
    import pickle
    import src.api.model_loader as ml_module
    from sklearn.linear_model import LogisticRegression

    original_bundle = ml_module._bundle
    try:
        model = LogisticRegression()
        pkl_path = tmp_path / "model.pkl"
        pkl_path.write_bytes(pickle.dumps(model))

        ml_module._bundle = None
        monkeypatch.setenv("MODEL_URI", str(pkl_path))
        monkeypatch.setenv("MODEL_VERSION", "v-test")
        monkeypatch.delenv("CHURNGUARD_ENV", raising=False)

        bundle = ml_module.load_bundle()

        assert bundle is not None
        assert bundle.version == "v-test"
        assert isinstance(bundle.model, LogisticRegression)
    finally:
        ml_module._bundle = original_bundle

"""Tests unitaires pour les modules train et evaluate."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.scripts.evaluate import compute_metrics
from src.scripts.load_data import load_data
from src.scripts.train import train_model

CSV_PATH = "data/telco_churn.csv"
EXPECTED_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


@pytest.fixture(scope="module")
def small_dataset():
    """Charge 300 lignes pour accelrer les tests d'entrainement."""
    X, y = load_data(CSV_PATH)
    return X.iloc[:300].reset_index(drop=True), y.iloc[:300].reset_index(drop=True)


def test_train_model_returns_fitted_pipeline(small_dataset):
    """train_model retourne un Pipeline sklearn entraine."""
    X, y = small_dataset
    model = train_model(X, y, "rf", {"n_estimators": 5, "random_state": 42})
    assert isinstance(model, Pipeline)


def test_train_model_can_predict(small_dataset):
    """Le modele entraine peut faire des predictions sur X."""
    X, y = small_dataset
    model = train_model(X, y, "rf", {"n_estimators": 5, "random_state": 42})
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_compute_metrics_returns_expected_keys(small_dataset):
    """compute_metrics retourne un dict avec les 5 cles attendues."""
    X, y = small_dataset
    model = train_model(X, y, "rf", {"n_estimators": 5, "random_state": 42})
    metrics = compute_metrics(model, X, y)
    assert set(metrics.keys()) == EXPECTED_METRIC_KEYS


def test_compute_metrics_in_valid_range(small_dataset):
    """Toutes les metriques sont entre 0 et 1."""
    X, y = small_dataset
    model = train_model(X, y, "rf", {"n_estimators": 5, "random_state": 42})
    metrics = compute_metrics(model, X, y)
    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"Metrique hors borne : {name}={value}"


def test_train_model_logistic_regression(small_dataset):
    """train_model fonctionne avec LogisticRegression."""
    X, y = small_dataset
    model = train_model(X, y, "lr", {"max_iter": 100, "random_state": 42})
    assert isinstance(model, Pipeline)


def test_train_model_gradient_boosting(small_dataset):
    """train_model fonctionne avec GradientBoostingClassifier."""
    X, y = small_dataset
    model = train_model(X, y, "gbt", {"n_estimators": 5, "random_state": 42})
    assert isinstance(model, Pipeline)


def test_preprocessor_expands_categorical_features(synthetic_dataset):
    """Le ColumnTransformer encode les catégorielles (OHE) et standardise les numériques."""
    X, y = synthetic_dataset
    model = train_model(X, y, "lr", {"max_iter": 100, "random_state": 42})
    preprocessor = model.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] > X.shape[1]  # OHE élargit les colonnes catégorielles
    assert not np.isnan(X_transformed).any()  # aucune valeur manquante après transformation

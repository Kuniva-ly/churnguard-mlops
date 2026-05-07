"""Évaluation du modèle entraîné."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.scripts.load_data import load_data


def compute_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calcule et retourne les 5 métriques d'évaluation du modèle."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def evaluate(csv_path: str | Path, model_path: str | Path) -> None:
    """Charge le modèle et l'évalue sur le jeu de test."""
    X, y = load_data(csv_path)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    with model_file.open("rb") as f:
        model = pickle.load(f)

    metrics = compute_metrics(model, X_test, y_test)

    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k.capitalize():<12}: {v:.4f}")
    print("=" * 60)

    y_pred = model.predict(X_test)
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Evaluate churn model")
    parser.add_argument("--csv-path", default="data/telco_churn.csv")
    parser.add_argument("--model-path", default="models/best_model.pkl")
    args = parser.parse_args()

    evaluate(args.csv_path, args.model_path)

"""Enregistre best_model.pkl dans le Registre de Modèles MLflow."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow import MlflowClient

load_dotenv()


def register(model_path: str, model_name: str, tracking_uri: str) -> str:
    """Enregistre un pickle local dans le MLflow Model Registry avec l'alias production."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path.resolve()}")

    with path.open("rb") as f:
        model = pickle.load(f)

    mlflow.set_experiment("churnguard-registration")

    with mlflow.start_run(run_name="register_pkl") as run:
        mlflow.log_param("source", str(path.resolve()))
        mlflow.log_param("model_type", type(model).__name__)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )
        run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{model_name}'")
    latest = sorted(versions, key=lambda v: int(v.version))[-1]

    client.set_registered_model_alias(
        name=model_name,
        alias="production",
        version=latest.version,
    )

    uri = f"models:/{model_name}@production"
    print(f"[ok] Modèle enregistré : {model_name} v{latest.version}")
    print("[ok] Alias             : production")
    print(f"[ok] Run ID            : {run_id}")
    print(f"[ok] MODEL_URI         : {uri}")
    return uri


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Register best_model.pkl into MLflow")
    parser.add_argument(
        "--model-path",
        default=os.environ.get("CHURNGUARD_MODEL_PATH", "models/best_model.pkl"),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "churn_model_v1"),
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    )
    args = parser.parse_args()

    register(args.model_path, args.model_name, args.tracking_uri)

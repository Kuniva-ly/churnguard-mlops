"""Entraînement du modèle de churn."""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.scripts.evaluate import compute_metrics
from src.scripts.load_data import load_data

# Catalogue des modèles disponibles
MODELS: dict[str, tuple[type, dict[str, Any]]] = {
    "lr": (LogisticRegression, {"max_iter": 1000, "random_state": 42}),
    "rf": (RandomForestClassifier, {"n_estimators": 100, "random_state": 42, "n_jobs": -1}),
    "gbt": (GradientBoostingClassifier, {"n_estimators": 100, "random_state": 42}),
}

RUN_NAMES = {
    "lr": "logistic_regression",
    "rf": "random_forest",
    "gbt": "gradient_boosting",
}


def _build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Construit le préprocesseur ColumnTransformer."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features),
        ]
    )


def train_model(
    X: pd.DataFrame, y: pd.Series, model_name: str, params: dict[str, Any]
) -> Pipeline:
    """Entraîne un pipeline sklearn et retourne le modèle ajusté."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    classifier_cls = {
        "lr": LogisticRegression,
        "rf": RandomForestClassifier,
        "gbt": GradientBoostingClassifier,
    }[model_name]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor(numeric_features, categorical_features)),
            ("classifier", classifier_cls(**params)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def train_with_mlflow(
    csv_path: str | Path,
    model_key: str = "rf",
    output_path: str | Path = "models/best_model.pkl",
    test_size: float = 0.2,
    register: bool = False,
) -> Pipeline:
    """Entraîne un modèle, logue dans MLflow et optionnellement l'enregistre dans le Registry."""
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature

    X, y = load_data(csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    cls, params = MODELS[model_key]
    mlflow.set_experiment("churnguard")

    with mlflow.start_run(run_name=RUN_NAMES[model_key]) as run:
        mlflow.log_params(params)
        mlflow.log_param("model_type", cls.__name__)
        mlflow.log_param("test_size", test_size)

        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            ).decode().strip()
            mlflow.set_tag("git_commit", commit)
        except Exception:
            pass
        mlflow.set_tag("python_version", sys.version.split()[0])

        model = train_model(X_train, y_train, model_key, params)
        metrics = compute_metrics(model, X_test, y_test)
        mlflow.log_metrics(metrics)

        signature = infer_signature(
            X_train.iloc[:5], model.predict_proba(X_train.iloc[:5])[:, 1]
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[:1],
            registered_model_name="churnguard" if register else None,
        )

        print(f"[ok] {RUN_NAMES[model_key]}: {metrics}")
        if register:
            print(f"[ok] Modele enregistre sous 'churnguard' (run={run.info.run_id})")

    return model


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Train churn model")
    parser.add_argument("--csv-path", default="data/telco_churn.csv")
    parser.add_argument("--output-path", default="models/best_model.pkl")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--model", choices=["lr", "rf", "gbt", "all"], default="rf",
        help="Modele a entrainer. 'all' entraine les 3."
    )
    parser.add_argument(
        "--register", action="store_true",
        help="Enregistre dans MLflow Model Registry sous le nom 'churnguard'."
    )
    parser.add_argument(
        "--best", action="store_true",
        help="Avec --model all : enregistre le modele avec le meilleur ROC-AUC."
    )
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)

    keys = ["lr", "rf", "gbt"] if args.model == "all" else [args.model]

    if args.best and len(keys) > 1:
        run_ids: dict[str, str] = {}
        roc_aucs: dict[str, float] = {}
        trained_models: dict[str, Any] = {}

        for key in keys:
            trained_models[key] = train_with_mlflow(
                csv_path=args.csv_path,
                model_key=key,
                output_path=args.output_path,
                test_size=args.test_size,
                register=False,
            )
            exp = mlflow.get_experiment_by_name("churnguard")
            if exp is None:
                raise RuntimeError("Experiment 'churnguard' not found in MLflow")
            runs: pd.DataFrame = mlflow.search_runs(
                experiment_ids=[exp.experiment_id],
                filter_string=f"tags.mlflow.runName = '{RUN_NAMES[key]}'",
                order_by=["start_time DESC"],
                max_results=1,
            )
            if not runs.empty:
                run_ids[key] = runs.iloc[0]["run_id"]
                roc_aucs[key] = runs.iloc[0]["metrics.roc_auc"]

        best_key = max(roc_aucs, key=lambda k: roc_aucs[k])
        best_roc = roc_aucs[best_key]
        print(f"\n[ok] Meilleur modele : {RUN_NAMES[best_key]} (ROC-AUC={best_roc:.4f})")

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(trained_models[best_key], f)
        print(f"[ok] Modele sauvegarde : {output_path}")

        client = MlflowClient()
        mv = mlflow.register_model(
            model_uri=f"runs:/{run_ids[best_key]}/model",
            name="churnguard",
        )
        client.set_registered_model_alias("churnguard", "production", mv.version)
        print(f"[ok] Modele enregistre : churnguard v{mv.version} @production")
        print("[ok] MODEL_URI : models:/churnguard@production")
    else:
        for key in keys:
            train_with_mlflow(
                csv_path=args.csv_path,
                model_key=key,
                output_path=args.output_path,
                test_size=args.test_size,
                register=args.register,
            )

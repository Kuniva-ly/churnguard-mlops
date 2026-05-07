"""Chargeur de modèle singleton pour l'API ChurnGuard.

Charge best_model.pkl une seule fois au démarrage et le met en cache en mémoire.
Sans danger pour les requêtes concurrentes — l'artefact pickle est en lecture
seule après chargement.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_bundle: ModelBundle | None = None


@dataclass
class ModelBundle:
    """Conteneur pour le modèle chargé et ses métadonnées."""

    version: str
    model_dir: Path
    model: Any


def load_bundle(model_path: str | Path | None = None) -> ModelBundle:
    """Charge le modèle depuis *model_path* et le met en cache comme bundle global.

    Utilise en priorité MODEL_URI (variable d'environnement), puis models/best_model.pkl.
    Formats supportés :
    - Pickle local : models/best_model.pkl
    - URI MLflow   : mlflow-models:/churn_model_v1/Production
    """
    global _bundle

    uri = str(model_path) if model_path else os.environ.get("MODEL_URI", "models/best_model.pkl")

    is_mlflow_uri = uri.startswith(("mlflow-models:/", "runs:/", "models:/"))

    if is_mlflow_uri:
        try:
            import mlflow.sklearn
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
            model = mlflow.sklearn.load_model(uri)
        except ImportError:
            raise RuntimeError("mlflow not installed. Run: pip install mlflow")
        version_label = os.environ.get("MODEL_VERSION", uri.rstrip("/").split("/")[-1])
        model_dir = Path(".")
    else:
        if os.environ.get("CHURNGUARD_ENV", "dev") == "production":
            raise RuntimeError(
                "Pickle loading is disabled in production. Use a models:/ MLflow URI."
            )
        path = Path(uri)
        if not path.exists():
            raise RuntimeError(f"Model file not found: {path.resolve()}")
        with path.open("rb") as f:
            model = pickle.load(f)
        version_label = os.environ.get("MODEL_VERSION", path.stem)
        model_dir = path.parent

    bundle = ModelBundle(
        version=version_label,
        model_dir=model_dir,
        model=model,
    )
    _bundle = bundle
    logger.info("Model bundle ready — version=%s, uri=%s", bundle.version, uri)
    return bundle


def get_bundle() -> ModelBundle | None:
    """Retourne le bundle actuellement chargé (None si pas encore chargé)."""
    return _bundle

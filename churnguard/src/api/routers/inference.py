"""Routeur d'inférence — prédictions de churn."""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from src.api.model_loader import get_bundle
from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CustomerFeatures,
    PredictSingleResponse,
)
from src.api.security import require_auth

logger = logging.getLogger(__name__)

router = APIRouter(tags=["inference"])


def _get_threshold() -> float:
    """Seuil de décision churn configurable via CHURN_THRESHOLD (défaut : 0.5)."""
    try:
        return float(os.environ.get("CHURN_THRESHOLD", "0.5"))
    except ValueError:
        return 0.5


def _require_bundle():
    """Lève HTTP 503 si le modèle n'est pas chargé."""
    bundle = get_bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health for details.",
        )
    return bundle


@router.post("/predict", response_model=PredictSingleResponse)
def predict_endpoint(
    request: CustomerFeatures,
    _: None = Depends(require_auth),
) -> PredictSingleResponse:
    """Retourne la probabilité de churn pour un client — authentification Basic requise."""
    bundle = _require_bundle()
    threshold = _get_threshold()

    t0 = time.perf_counter()
    try:
        df = pd.DataFrame([request.model_dump()])
        proba = float(bundle.model.predict_proba(df)[0, 1])
    except Exception as exc:
        logger.exception("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    logger.info("predict | 1 row | %.1f ms", (time.perf_counter() - t0) * 1000)
    return PredictSingleResponse(churn=proba > threshold, probability=proba)


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(
    request: BatchPredictRequest,
    _: None = Depends(require_auth),
) -> BatchPredictResponse:
    """Retourne les probabilités de churn pour un lot de clients (max 100) — auth requise."""
    bundle = _require_bundle()
    threshold = _get_threshold()

    t0 = time.perf_counter()
    try:
        df = pd.DataFrame([c.model_dump() for c in request.data])
        probas = await run_in_threadpool(lambda: bundle.model.predict_proba(df)[:, 1].tolist())
    except Exception as exc:
        logger.exception("Batch inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    logger.info("predict/batch | %d rows | %.1f ms", len(probas), (time.perf_counter() - t0) * 1000)
    predictions = [PredictSingleResponse(churn=p > threshold, probability=float(p)) for p in probas]
    return BatchPredictResponse(predictions=predictions, model_version=bundle.version)


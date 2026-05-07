"""Endpoints de santé et de version."""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.model_loader import get_bundle
from src.api.schemas import HealthResponse, VersionResponse
from src.api.security import require_auth

router = APIRouter(tags=["ops"])


@router.get("/", include_in_schema=False)
def root() -> JSONResponse:
    return JSONResponse({
        "message": "Bienvenue sur l'API ChurnGuard ",
        "docs": "/docs",
        "health": "/health",
        "version": os.environ.get("API_VERSION", "1.0.0"),
    })


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Sonde de disponibilité — publique, aucune authentification requise."""
    bundle = get_bundle()
    return HealthResponse(
        status="ok" if bundle is not None else "degraded",
        model_loaded=bundle is not None,
        model_version=bundle.version if bundle else None,
    )


@router.get("/version", response_model=VersionResponse)
def version(_: None = Depends(require_auth)) -> VersionResponse:
    """Informations de version de l'API et du modèle — authentification Basic requise."""
    bundle = get_bundle()
    return VersionResponse(
        api_version=os.environ.get("API_VERSION", "1.0.0"),
        model_version=bundle.version if bundle else None,
        model_name=os.environ.get("MODEL_NAME", "churn_model_v1"),
    )

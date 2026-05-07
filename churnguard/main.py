"""API d'inférence ChurnGuard.

Endpoints
---------
GET  /health     Sonde de disponibilité + statut du modèle
GET  /version    Informations de version de l'API et du modèle
POST /predict    Inférence de probabilité de churn

Configuration (variables d'environnement)
------------------------------------------
MODEL_URI        Chemin vers best_model.pkl ou URI mlflow-models:/
MODEL_VERSION    Label de version (défaut : best_model)
API_USERNAME     Nom d'utilisateur Basic auth (défaut : admin)
API_PASSWORD     Mot de passe Basic auth (défaut : changeme)

Exécution locale
----------------
    uvicorn src.main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_loader import load_bundle
from src.api.routers import health, inference

API_VERSION = os.environ.get("API_VERSION", "1.0.0")

# Structured JSON logging
try:
    try:
        from pythonjsonlogger.json import JsonFormatter as _JsonFormatter
    except ImportError:
        from pythonjsonlogger.jsonlogger import JsonFormatter as _JsonFormatter  # type: ignore[no-redef]
    handler = logging.StreamHandler()
    handler.setFormatter(
        _JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logging.root.setLevel(os.environ.get("API_LOG_LEVEL", "INFO").upper())
    logging.root.handlers = [handler]
except ImportError:
    logging.basicConfig(
        level=os.environ.get("API_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

logger = logging.getLogger(__name__)


# Lifecycle

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ChurnGuard model ...")
    try:
        load_bundle()
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        # Start in degraded mode — /health will report the problem
    yield
    logger.info("Shutting down.")


# Application

app = FastAPI(
    title="ChurnGuard Inference API",
    description=(
        "Bienvenue sur l'API ChurnGuard — votre alliée pour prédire le churn des clients TelcoFr !\n\n"
        "Cette API prédit la probabilité de churn des clients TelcoFr.\n\n"
        "Consultez la documentation interactive ci-dessous pour explorer les endpoints disponibles.\n\n"
        "Les endpoints `/predict` et `/version` nécessitent une authentification **Basic Auth**."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "").split(",") or ["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Gestionnaire d'exceptions global

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Routers

app.include_router(health.router)
app.include_router(inference.router)

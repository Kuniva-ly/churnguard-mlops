"""Schémas Pydantic de requête / réponse pour l'API ChurnGuard."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Caractéristiques d'un client Telco — 19 features attendues par le modèle."""

    gender: Literal["Male", "Female"]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1)]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: Annotated[int, Field(ge=0)]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: Annotated[float, Field(ge=0.0)]
    TotalCharges: Annotated[float, Field(ge=0.0)]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
                    "StreamingTV": "Yes",
                    "StreamingMovies": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                    "MonthlyCharges": 70.35,
                    "TotalCharges": 844.2,
                }
            ]
        }
    }


class BatchPredictRequest(BaseModel):
    """Payload pour POST /predict/batch — entre 1 et 100 clients."""

    data: Annotated[list[CustomerFeatures], Field(min_length=1, max_length=100)]


class PredictSingleResponse(BaseModel):
    """Réponse pour un client : probabilité de churn et décision binaire."""

    churn: bool
    probability: float


class BatchPredictResponse(BaseModel):
    """Réponse batch : liste de prédictions et version du modèle."""

    model_config = {"protected_namespaces": ()}
    predictions: list[PredictSingleResponse]
    model_version: str


class HealthResponse(BaseModel):
    """Statut de santé de l'API."""

    model_config = {"protected_namespaces": ()}
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None = None


class VersionResponse(BaseModel):
    """Informations de version de l'API et du modèle."""

    model_config = {"protected_namespaces": ()}
    api_version: str
    model_version: str | None
    model_name: str

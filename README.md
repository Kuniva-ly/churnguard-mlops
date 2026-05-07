# ChurnGuard MLOps

> Système de prédiction de churn client pour opérateur Telco — pipeline MLOps complet avec MLflow, FastAPI, Docker et Kubernetes (k3s).

![CI](https://github.com/Kuniva-ly/churnguard-mlops/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/Kuniva-ly/churnguard-mlops/actions/workflows/release.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Architecture

```
┌─────────────┐    entraîne     ┌─────────────┐    charge     ┌─────────────┐
│   Trainer   │ ──────────────▶ │   MLflow    │ ◀──────────── │     API     │
│  (one-shot) │                 │  Registry   │               │  (FastAPI)  │
└─────────────┘                 └─────────────┘               └─────────────┘
       │                              │                               │
  telco_churn.csv               :5000 (UI)                     :8000 (REST)
```

| Service | Port | Rôle |
|---|---|---|
| `mlflow` | 5000 | Tracking server + Model Registry |
| `trainer` | — | Entraîne le modèle (LR / RF / GBT) et l'enregistre dans MLflow |
| `api` | 8000 | Sert les prédictions via REST (HTTP Basic Auth) |

---

## Démarrage rapide

### Prérequis

- Docker Desktop
- Dataset `data/telco_churn.csv` — [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Stack complète (Docker Compose)

```bash
cp churnguard/.env.example churnguard/.env
cd churnguard
docker compose up --build
```

| URL | Description |
|---|---|
| http://localhost:5000 | MLflow UI |
| http://localhost:8000/docs | API Swagger |
| http://localhost:8000/health | Health check |

### Développement local

```bash
cd churnguard
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements/dev.txt
pytest -v
```

---

## Déploiement Kubernetes (k3s)

### Prérequis

- k3s installé sur Linux/WSL2
- Images Docker pushées sur Docker Hub

### Déployer

```bash
# 1. Copier le CSV sur le nœud
sudo mkdir -p /opt/churnguard/data
sudo cp churnguard/data/telco_churn.csv /opt/churnguard/data/

# 2. Builder et pusher les images
docker build -f churnguard/Dockerfile -t lyagoubimohamed/churnguard:latest churnguard/
docker build -f churnguard/Dockerfile.trainer -t lyagoubimohamed/churnguard-trainer:latest churnguard/
docker push lyagoubimohamed/churnguard:latest
docker push lyagoubimohamed/churnguard-trainer:latest

# 3. Déployer
bash k3s/deploy.sh
```

### Accéder à l'API

Ajouter dans `/etc/hosts` :
```
127.0.0.1 churnguard.local
```

- API : http://churnguard.local/docs
- Health : http://churnguard.local/health

---

## API

### Authentification

HTTP Basic Auth — configurable via variables d'environnement.

| Variable | Défaut |
|---|---|
| `API_USERNAME` | `admin` |
| `API_PASSWORD` | `changeme` |

### Endpoints

| Méthode | Route | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | Non | Statut de l'API et du modèle |
| `GET` | `/version` | Oui | Version de l'API et du modèle |
| `POST` | `/predict` | Oui | Prédiction pour un client |
| `POST` | `/predict/batch` | Oui | Prédictions batch (max 100) |

### Exemple

```bash
curl -X POST http://localhost:8000/predict \
  -u admin:changeme \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
    "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "Yes",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35, "TotalCharges": 844.2
  }'
```

```json
{"churn": true, "probability": 0.82}
```

---

## Structure du projet

```
churnguard-mlops/
├── churnguard/
│   ├── main.py                    # point d'entrée FastAPI
│   ├── Dockerfile                 # image API (multi-stage)
│   ├── Dockerfile.trainer         # image entraînement
│   ├── docker-compose.yml
│   ├── requirements/
│   │   ├── base.txt               # pandas, numpy, sklearn, dotenv
│   │   ├── api.txt                # fastapi, uvicorn, mlflow-skinny
│   │   ├── ml.txt                 # mlflow complet
│   │   └── dev.txt                # pytest, ruff, mypy
│   ├── src/
│   │   ├── api/
│   │   │   ├── model_loader.py    # chargement modèle MLflow
│   │   │   ├── schemas.py         # schémas Pydantic (19 features)
│   │   │   ├── security.py        # HTTP Basic Auth
│   │   │   └── routers/
│   │   │       ├── health.py      # GET /health, GET /version
│   │   │       └── inference.py   # POST /predict, POST /predict/batch
│   │   └── scripts/
│   │       ├── train.py           # entraînement + MLflow tracking
│   │       ├── evaluate.py        # métriques d'évaluation
│   │       ├── register_model.py  # enregistrement dans le registry
│   │       └── load_data.py       # chargement et préparation des données
│   └── tests/                     # 27 tests, couverture > 70%
├── k3s/                           # manifests Kubernetes
│   ├── deploy.sh                  # script de déploiement automatisé
│   ├── namespace.yaml
│   ├── mlflow/                    # Deployment + Service + PVC
│   ├── trainer/                   # Job one-shot
│   └── api/                       # Deployment + Service + Ingress + Secret
└── .github/workflows/
    ├── ci.yml                     # Lint + Tests + Docker build
    └── release.yml                # Build + push GHCR + GitHub Release
```

---

## MLflow

Accéder à l'UI : http://localhost:5000

Charger le modèle en production :

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/churnguard@production")
```

Modèles disponibles : `lr` (Logistic Regression), `rf` (Random Forest), `gbt` (Gradient Boosting).

---

## CI/CD

| Workflow | Déclencheur | Actions |
|---|---|---|
| `ci.yml` | Push / PR sur `main` | Ruff + mypy + pytest (70% coverage) + Docker build + Trivy scan + Slack |
| `release.yml` | Tag `v*.*.*` | Build + push GHCR + GitHub Release |

### Créer une release

```bash
git tag v1.0.0
git push origin v1.0.0
```

---

## Tests

```bash
cd churnguard
pytest -v --cov=src --cov-report=term-missing
```

27 tests — couverture minimum **70%** imposée en CI.

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | URI du serveur MLflow |
| `MODEL_URI` | `models:/churnguard@production` | URI du modèle dans le registry |
| `MODEL_NAME` | `churnguard` | Nom du modèle |
| `API_USERNAME` | `admin` | Identifiant Basic Auth |
| `API_PASSWORD` | `changeme` | Mot de passe Basic Auth |
| `CHURN_THRESHOLD` | `0.5` | Seuil de décision churn |
| `API_LOG_LEVEL` | `INFO` | Niveau de log |

---

## Données

**Telco Customer Churn** — IBM Sample Data, 7 043 clients, 21 colonnes.

- Source : [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Le CSV n'est pas commité (`.gitignore`)

---

## Licence

Code : MIT — Données : IBM Sample Data (usage éducatif).

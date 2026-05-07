# ChurnGuard MLOps

Prédiction de churn client pour TelcoFr — stack MLOps complète avec MLflow, FastAPI et Docker.

![CI](https://github.com/Kuniva-ly/churnguard_mlops/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/Kuniva-ly/churnguard_mlops/actions/workflows/release.yml/badge.svg)

---

## Architecture

```
┌─────────────┐    entraîne     ┌─────────────┐    charge     ┌─────────────┐
│   Trainer   │ ─────────────▶  │   MLflow    │ ◀────────────│     API     │
│  (one-shot) │                 │  Registry   │               │  (FastAPI)  │
└─────────────┘                 └─────────────┘               └─────────────┘
       │                              │                               │
  data/telco_churn.csv           :5000 (UI)                     :8000 (REST)
```

**Services Docker Compose :**
| Service | Port | Rôle |
|---|---|---|
| `mlflow` | 5000 | Tracking server + Model Registry |
| `trainer` | — | Entraîne le modèle et l'enregistre dans MLflow |
| `api` | 8000 | Sert les prédictions via REST |

---

## Démarrage rapide

### Prérequis
- Docker Desktop
- Le dataset `data/telco_churn.csv` ([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn))

### Lancer la stack complète

```bash
cp .env.example .env
docker compose up --build
```

La stack démarre en moins de 2 minutes. L'ordre est garanti : MLflow → Trainer → API.

- MLflow UI : http://localhost:5000
- API docs  : http://localhost:8000/docs

### Utiliser l'image publiée (sans build)

```bash
docker pull ghcr.io/<github-username>/churnguard:latest
```

### Développement local

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Structure du projet

```
churnguard/
├── main.py                     # point d'entrée FastAPI
├── requirements/
│   ├── base.txt                # pandas, numpy, dotenv, json-logger
│   ├── api.txt                 # fastapi, uvicorn, scikit-learn, mlflow
│   ├── ml.txt                  # scikit-learn, mlflow
│   └── dev.txt                 # pytest, ruff, mypy, black, httpx
├── Dockerfile                  # image multi-stage (API)
├── Dockerfile.trainer          # image entraînement
├── docker-compose.yml
├── data/
│   └── telco_churn.csv         # dataset Telco (non commité)
├── models/                     # best_model.pkl (non commité)
├── notebook/
│   └── exploration.ipynb       # exploration initiale
├── src/
│   ├── api/
│   │   ├── model_loader.py     # chargement du modèle MLflow
│   │   ├── schemas.py          # schémas Pydantic (19 features)
│   │   ├── security.py         # HTTP Basic Auth
│   │   └── routers/
│   │       ├── health.py       # GET /health, GET /version
│   │       └── inference.py    # POST /predict, POST /predict/batch
│   └── scripts/
│       ├── train.py            # entraînement + MLflow tracking
│       ├── evaluate.py         # évaluation du modèle
│       ├── register_model.py   # enregistrement dans le registry
│       └── load_data.py        # chargement et préparation des données
└── tests/
    ├── test_api.py             # 13 tests API
    ├── test_data.py            # 6 tests données
    └── test_train.py           # 6 tests entraînement
```

---

## API

### Authentification

HTTP Basic Auth — identifiants configurables via variables d'environnement :

| Variable | Défaut |
|---|---|
| `API_USERNAME` | `admin` |
| `API_PASSWORD` | `changeme` |

### Endpoints

| Méthode | Route | Auth | Description |
|---|---|---|---|
| `GET` | `/health` | Non | Statut de l'API et du modèle chargé |
| `GET` | `/version` | Oui | Version de l'API et du modèle |
| `POST` | `/predict` | Oui | Prédiction pour un client |
| `POST` | `/predict/batch` | Oui | Prédictions pour un lot (max 100) |

### Exemple de requête

```bash
curl -X POST http://localhost:8000/predict \
  -u admin:changeme \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 906.0
  }'
```

Réponse :
```json
{"churn": true, "probability": 0.82}
```

---

## MLflow

Accéder à l'UI : http://localhost:5000

Le modèle est enregistré sous le nom `churnguard` avec l'alias `production`.  
Pour le charger manuellement :

```python
import mlflow.sklearn
model = mlflow.sklearn.load_model("models:/churnguard@production")
```

---

## CI/CD

| Workflow | Déclencheur | Actions |
|---|---|---|
| `ci.yml` | Push / PR sur `main` | Lint, type check, tests, build Docker |
| `release.yml` | Push de tag `v*.*.*` | Build + push sur GHCR, GitHub Release avec changelog |

### Créer une release

```bash
git tag v1.0.0
git push origin v1.0.0
# → image publiée sur ghcr.io/<github-username>/churnguard:1.0.0
# → GitHub Release créée avec changelog Conventional Commits
```

---

## Tests

```bash
pytest -v
```

25 tests — couverture minimum 70% imposée en CI.

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | URI du serveur MLflow |
| `MODEL_URI` | `models:/churnguard@production` | URI du modèle dans le registry |
| `MODEL_NAME` | `churnguard` | Nom du modèle dans le registry |
| `MODEL_VERSION` | `best_model` | Label de version affiché |
| `API_USERNAME` | `admin` | Identifiant Basic Auth |
| `API_PASSWORD` | `changeme` | Mot de passe Basic Auth |
| `CHURN_THRESHOLD` | `0.5` | Seuil de décision churn |
| `API_LOG_LEVEL` | `INFO` | Niveau de log |

Copier `.env.example` en `.env` pour la configuration locale.

---

## Données

**Telco Customer Churn** — IBM Sample Data, 7 043 lignes, 21 colonnes.

- [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

Le CSV n'est pas commité dans le dépôt (voir `.gitignore`).

---

## Licence

Code : MIT.  
Données : IBM Sample Data — usage éducatif.

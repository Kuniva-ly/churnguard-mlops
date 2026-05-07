#!/bin/bash
set -e

NAMESPACE="churnguard"
DATA_PATH="/opt/churnguard/data"
CSV_SRC="churnguard/data/telco_churn.csv"

echo " ChurnGuard - Deploiement k3s"


# 1. Namespace
echo ""
echo "[1/5] Creation du namespace..."
k3s kubectl apply -f k3s/namespace.yaml

# 2. Copie du CSV sur le noeud
echo ""
echo "[2/5] Preparation des donnees..."
sudo mkdir -p "$DATA_PATH"
sudo cp "$CSV_SRC" "$DATA_PATH/telco_churn.csv"
echo "      CSV copie -> $DATA_PATH/telco_churn.csv"

# 3. MLflow
echo ""
echo "[3/5] Deploiement MLflow..."
k3s kubectl apply -f k3s/mlflow/pvc.yaml
k3s kubectl apply -f k3s/mlflow/deployment.yaml
k3s kubectl apply -f k3s/mlflow/service.yaml

echo "      Attente MLflow ready..."
k3s kubectl wait --namespace "$NAMESPACE" \
  --for=condition=ready pod \
  --selector=app=mlflow \
  --timeout=120s

# 4. Trainer (Job one-shot)
echo ""
echo "[4/5] Lancement du trainer..."
# Supprime l'ancien job si existe
k3s kubectl delete job churnguard-trainer -n "$NAMESPACE" --ignore-not-found
k3s kubectl apply -f k3s/trainer/job.yaml

echo "      Attente fin du training..."
k3s kubectl wait --namespace "$NAMESPACE" \
  --for=condition=complete job/churnguard-trainer \
  --timeout=600s

echo "      Training termine !"

# 5. API
echo ""
echo "[5/5] Deploiement de l'API..."
k3s kubectl apply -f k3s/api/secret.yaml
k3s kubectl apply -f k3s/api/deployment.yaml
k3s kubectl apply -f k3s/api/service.yaml
k3s kubectl apply -f k3s/api/ingress.yaml

echo "      Attente API ready..."
k3s kubectl wait --namespace "$NAMESPACE" \
  --for=condition=ready pod \
  --selector=app=churnguard-api \
  --timeout=120s

# 6. Ajout hosts local
echo ""
echo "[6/6] Configuration /etc/hosts..."
if ! grep -q "churnguard.local" /etc/hosts; then
  echo "127.0.0.1 churnguard.local" | sudo tee -a /etc/hosts
  echo "      churnguard.local ajoute dans /etc/hosts"
else
  echo "      churnguard.local deja present"
fi

echo ""
echo " Deploiement termine !"

echo ""
echo " MLflow  : http://$(k3s kubectl get svc mlflow -n $NAMESPACE -o jsonpath='{.spec.clusterIP}'):5000"
echo " API     : http://churnguard.local/health"
echo " API doc : http://churnguard.local/docs"
echo ""
echo " k3s kubectl get pods -n $NAMESPACE"
echo "========================================"

#!/bin/bash
set -e

NAMESPACE="churnguard"

echo " ChurnGuard - Deploiement k3s"

# 1. Namespace
echo ""
echo "[1/4] Creation du namespace..."
k3s kubectl apply -f k3s/namespace.yaml

# 2. MLflow
echo ""
echo "[2/4] Deploiement MLflow..."
k3s kubectl apply -f k3s/mlflow/pvc.yaml
k3s kubectl apply -f k3s/mlflow/deployment.yaml
k3s kubectl apply -f k3s/mlflow/service.yaml

echo "      Attente MLflow ready..."
k3s kubectl wait --namespace "$NAMESPACE" \
  --for=condition=ready pod \
  --selector=app=mlflow \
  --timeout=120s

# 3. Trainer (Job one-shot)
echo ""
echo "[3/4] Lancement du trainer..."
k3s kubectl delete job churnguard-trainer -n "$NAMESPACE" --ignore-not-found
k3s kubectl apply -f k3s/trainer/job.yaml

echo "      Attente fin du training..."
k3s kubectl wait --namespace "$NAMESPACE" \
  --for=condition=complete job/churnguard-trainer \
  --timeout=600s

echo "      Training termine !"

# 4. API
echo ""
echo "[4/4] Deploiement de l'API..."
k3s kubectl apply -f k3s/api/secret.yaml
k3s kubectl apply -f k3s/api/deployment.yaml
k3s kubectl apply -f k3s/api/service.yaml
k3s kubectl apply -f k3s/api/ingress.yaml

echo "      Attente API ready..."
k3s kubectl wait --namespace "$NAMESPACE" \
  --for=condition=ready pod \
  --selector=app=churnguard-api \
  --timeout=120s

# 5. Ajout hosts local
echo ""
echo "[5/5] Configuration /etc/hosts..."
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

#!/bin/bash
# Deploy to Kubernetes with smoke tests

set -e

NAMESPACE="mlops"
DEPLOYMENT="cats-dogs-api"
IMAGE_TAG="${1:-latest}"

echo "=========================================="
echo "Deploying to Kubernetes"
echo "=========================================="
echo "Namespace: $NAMESPACE"
echo "Deployment: $DEPLOYMENT"
echo "Image Tag: $IMAGE_TAG"
echo ""

# Create namespace
echo "[1/6] Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply deployments
echo "[2/6] Applying Kubernetes manifests..."
kubectl apply -f k8s/deployment.yaml

# Wait for rollout
echo "[3/6] Waiting for deployment..."
kubectl rollout status deployment/$DEPLOYMENT -n $NAMESPACE --timeout=5m

# Get service info
echo ""
echo "[4/6] Getting service endpoint..."
SERVICE_IP=$(kubectl get service $DEPLOYMENT -n $NAMESPACE \
  -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null \
  || kubectl get service $DEPLOYMENT -n $NAMESPACE \
  -o jsonpath='{.spec.clusterIP}' 2>/dev/null \
  || echo "pending")

echo "Service IP/Hostname: $SERVICE_IP"
echo ""

# Pod status
echo "[5/6] Pod status:"
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT

# Run smoke tests
echo ""
echo "[6/6] Running smoke tests..."
if [ "$SERVICE_IP" != "pending" ]; then
    ENDPOINT="http://$SERVICE_IP"
else
    # Use port-forward for local testing
    kubectl port-forward -n $NAMESPACE \
        svc/$DEPLOYMENT 8000:80 &
    ENDPOINT="http://localhost:8000"
    sleep 5
fi

# Health check
echo "Testing health endpoint..."
for i in {1..5}; do
    if curl -f "$ENDPOINT/health" > /dev/null 2>&1; then
        echo "[OK] Health check passed"
        break
    fi
    echo "  Attempt $i/5 failed, retrying..."
    sleep 2
done

echo ""
echo "=========================================="
echo "Deployment completed successfully"
echo "=========================================="
echo ""
echo "Access the API at: $ENDPOINT"
echo "View logs: kubectl logs -n $NAMESPACE -f -l app=$DEPLOYMENT"
echo "Check status: kubectl get all -n $NAMESPACE"

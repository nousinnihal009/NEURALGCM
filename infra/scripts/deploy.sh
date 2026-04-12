#!/bin/bash
# Full deploy: update k8s manifests with new image tag, apply, verify.
# Usage: ./deploy.sh [image_tag]
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
CLUSTER="neuralgcm-weather-cluster"
REGISTRY_URL="us-central1-docker.pkg.dev/${PROJECT_ID}/neuralgcm-weather"
TAG=${1:-latest}

echo "════════════════════════════════════════════════════════════"
echo "  Deploying tag: $TAG"
echo "  Cluster: $CLUSTER"
echo "  Registry: $REGISTRY_URL"
echo "════════════════════════════════════════════════════════════"

# Get cluster credentials
echo "→ Fetching cluster credentials..."
gcloud container clusters get-credentials "$CLUSTER" \
  --region "$REGION" --project "$PROJECT_ID"

# Apply namespace first
echo "→ Applying namespace..."
kubectl apply -f infra/k8s/namespace.yaml

# Get infrastructure values for configmap substitution
echo "→ Fetching infrastructure values..."
DB_IP=$(terraform -chdir=infra/terraform output -raw db_private_ip 2>/dev/null || echo "CLOUD_SQL_PRIVATE_IP")
REDIS_IP=$(terraform -chdir=infra/terraform output -raw redis_host 2>/dev/null || echo "REDIS_HOST")

# Update image tags and config values (using sed on temp copies)
echo "→ Preparing manifests..."
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT
cp -r infra/k8s/ "$TMPDIR/"

# Substitute image references
sed -i "s|REGISTRY_URL|${REGISTRY_URL}|g" "$TMPDIR/k8s/api/deployment.yaml"
sed -i "s|IMAGE_TAG|${TAG}|g"             "$TMPDIR/k8s/api/deployment.yaml"
sed -i "s|REGISTRY_URL|${REGISTRY_URL}|g" "$TMPDIR/k8s/worker/deployment.yaml"
sed -i "s|IMAGE_TAG|${TAG}|g"             "$TMPDIR/k8s/worker/deployment.yaml"

# Substitute config values
sed -i "s|CLOUD_SQL_PRIVATE_IP|${DB_IP}|g"  "$TMPDIR/k8s/api/configmap.yaml"
sed -i "s|REDIS_HOST|${REDIS_IP}|g"         "$TMPDIR/k8s/api/configmap.yaml"
sed -i "s|PROJECT_ID|${PROJECT_ID}|g"       "$TMPDIR/k8s/api/configmap.yaml"
sed -i "s|CLOUD_SQL_PRIVATE_IP|${DB_IP}|g"  "$TMPDIR/k8s/worker/configmap.yaml"
sed -i "s|REDIS_HOST|${REDIS_IP}|g"         "$TMPDIR/k8s/worker/configmap.yaml"
sed -i "s|PROJECT_ID|${PROJECT_ID}|g"       "$TMPDIR/k8s/worker/configmap.yaml"

# Apply manifests
echo "→ Applying API manifests..."
kubectl apply -f "$TMPDIR/k8s/api/"

echo "→ Applying worker manifests..."
kubectl apply -f "$TMPDIR/k8s/worker/"

# Wait for rollout (zero-downtime rolling update)
echo ""
echo "→ Waiting for API rollout..."
kubectl rollout status deployment/neuralgcm-api \
  -n neuralgcm --timeout=5m

echo "→ Waiting for worker rollout..."
kubectl rollout status deployment/neuralgcm-worker \
  -n neuralgcm --timeout=10m

# Get external IP
echo ""
API_IP=$(kubectl get svc neuralgcm-api-lb -n neuralgcm \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")

echo "════════════════════════════════════════════════════════════"
echo "  Deployment complete!"
echo "  Tag: $TAG"
echo "  API LoadBalancer IP: $API_IP"
if [ "$API_IP" != "pending" ]; then
  echo "  API URL: http://$API_IP/api/v1/docs"
fi
echo "════════════════════════════════════════════════════════════"

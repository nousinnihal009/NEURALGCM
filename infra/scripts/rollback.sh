#!/bin/bash
# Roll back to previous deployment.
# Usage: ./rollback.sh [revision_number]
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
CLUSTER="neuralgcm-weather-cluster"
REVISION=${1:-""}

echo "════════════════════════════════════════════════════════════"
echo "  Rolling back deployments..."
echo "════════════════════════════════════════════════════════════"

gcloud container clusters get-credentials "$CLUSTER" \
  --region "$REGION" --project "$PROJECT_ID"

if [ -n "$REVISION" ]; then
  echo "→ Rolling back to revision: $REVISION"
  kubectl rollout undo deployment/neuralgcm-api \
    -n neuralgcm --to-revision="$REVISION"
  kubectl rollout undo deployment/neuralgcm-worker \
    -n neuralgcm --to-revision="$REVISION"
else
  echo "→ Rolling back to previous revision"
  kubectl rollout undo deployment/neuralgcm-api    -n neuralgcm
  kubectl rollout undo deployment/neuralgcm-worker -n neuralgcm
fi

echo "→ Waiting for rollout..."
kubectl rollout status deployment/neuralgcm-api    -n neuralgcm --timeout=5m
kubectl rollout status deployment/neuralgcm-worker -n neuralgcm --timeout=10m

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Rollback complete ✓"
echo "════════════════════════════════════════════════════════════"

#!/bin/bash
# Build and push all Docker images to Artifact Registry.
# Usage: ./build_push.sh [image_tag]
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
REGISTRY="${REGION}-docker.pkg.dev"
REPO="neuralgcm-weather"
TAG=${1:-$(git rev-parse --short HEAD)}
REGISTRY_URL="${REGISTRY}/${PROJECT_ID}/${REPO}"

echo "════════════════════════════════════════════════════════════"
echo "  Building images with tag: $TAG"
echo "  Registry: $REGISTRY_URL"
echo "════════════════════════════════════════════════════════════"

# Configure Docker auth
gcloud auth configure-docker "$REGISTRY" --quiet

# Build and push API
echo ""
echo "► [1/3] Building API image..."
docker build \
  -f infra/dockerfiles/Dockerfile.api.prod \
  -t "${REGISTRY_URL}/api:${TAG}" \
  -t "${REGISTRY_URL}/api:latest" \
  .
docker push "${REGISTRY_URL}/api:${TAG}"
docker push "${REGISTRY_URL}/api:latest"
echo "  ✓ API image pushed"

# Build and push GPU worker
echo ""
echo "► [2/3] Building GPU worker image..."
docker build \
  -f infra/dockerfiles/Dockerfile.worker.gpu \
  -t "${REGISTRY_URL}/worker:${TAG}" \
  -t "${REGISTRY_URL}/worker:latest" \
  .
docker push "${REGISTRY_URL}/worker:${TAG}"
docker push "${REGISTRY_URL}/worker:latest"
echo "  ✓ GPU worker image pushed"

# Build and push frontend
echo ""
echo "► [3/3] Building frontend image..."
docker build \
  -f infra/dockerfiles/Dockerfile.frontend \
  -t "${REGISTRY_URL}/frontend:${TAG}" \
  -t "${REGISTRY_URL}/frontend:latest" \
  .
docker push "${REGISTRY_URL}/frontend:${TAG}"
docker push "${REGISTRY_URL}/frontend:latest"
echo "  ✓ Frontend image pushed"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  All images pushed successfully. Tag: $TAG"
echo "  Deploy with: ./infra/scripts/deploy.sh $TAG"
echo "════════════════════════════════════════════════════════════"

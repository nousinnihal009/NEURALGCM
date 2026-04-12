#!/bin/bash
# Run ONCE before terraform init.
# Creates the GCS bucket for Terraform state and enables billing.
set -euo pipefail

PROJECT_ID=${1:?"Usage: ./bootstrap.sh PROJECT_ID"}
REGION="us-central1"
BUCKET_NAME="${PROJECT_ID}-neuralgcm-terraform-state"

echo "════════════════════════════════════════════════════════════"
echo "  Bootstrapping GCP project: $PROJECT_ID"
echo "════════════════════════════════════════════════════════════"

# Ensure gcloud is authenticated
gcloud config set project "$PROJECT_ID"

# Enable billing API (required for all other APIs)
echo "→ Enabling core APIs..."
gcloud services enable cloudbilling.googleapis.com \
  cloudresourcemanager.googleapis.com \
  storage.googleapis.com

# Create Terraform state bucket
echo "→ Creating Terraform state bucket: gs://${BUCKET_NAME}"
gsutil mb -p "$PROJECT_ID" -l "$REGION" \
  "gs://${BUCKET_NAME}" 2>/dev/null || echo "  Bucket may already exist — continuing"

# Enable versioning on state bucket (protect state from accidents)
echo "→ Enabling versioning and uniform access..."
gsutil versioning set on "gs://${BUCKET_NAME}"
gsutil ubla set on "gs://${BUCKET_NAME}"

# Set up Workload Identity Federation for GitHub Actions (optional)
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Bootstrap complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Copy terraform.tfvars.example → terraform.tfvars"
echo "     cp infra/terraform/terraform.tfvars.example infra/terraform/terraform.tfvars"
echo ""
echo "  2. Edit terraform.tfvars with your actual values"
echo ""
echo "  3. Update backend bucket name in main.tf if needed:"
echo "     Current: neuralgcm-terraform-state"
echo "     Created: ${BUCKET_NAME}"
echo ""
echo "  4. Run Terraform:"
echo "     cd infra/terraform"
echo "     terraform init"
echo "     terraform plan -out=tfplan"
echo "     terraform apply tfplan"

#!/bin/bash
# Deploy Weber electrodynamics simulation to Google Cloud Run with GPU
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - Docker or Cloud Build access
#   - Project: gnosis-459403

set -e

PROJECT_ID="gnosis-459403"
REGION="us-central1"
SERVICE_NAME="weber-sim"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== Weber Electrodynamics — Cloud Run GPU Deployment ==="
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Service:  ${SERVICE_NAME}"
echo "  Image:    ${IMAGE}"
echo

# Build with Cloud Build (has access to GPU-capable build machines)
echo ">>> Submitting build to Cloud Build..."
gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE}" \
  --timeout=1800 \
  --machine-type=e2-highcpu-8 \
  .

echo
echo ">>> Deploying to Cloud Run with GPU..."
gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE}" \
  --execution-environment gen2 \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --memory 32Gi \
  --cpu 8 \
  --timeout 3600 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 1 \
  --port 8080 \
  --allow-unauthenticated \
  --set-env-vars "CUDA_VISIBLE_DEVICES=0"

echo
echo ">>> Deployment complete."
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --format "value(status.url)")
echo "  URL: ${SERVICE_URL}"
echo
echo "Test:"
echo "  curl ${SERVICE_URL}/health"
echo "  curl ${SERVICE_URL}/reference"
echo "  curl -X POST ${SERVICE_URL}/run -H 'Content-Type: application/json' -d '{\"binary\":\"brake-recoil\"}'"

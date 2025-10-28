#!/bin/bash
# Script to build the Docker image for TS-DIA training
# Usage: ./build.sh [IMAGE_NAME] [--force]

set -e

# Parse arguments
IMAGE_NAME=${1:-kaldi-ts-dia}
FORCE_BUILD=false

# Check for --force flag
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE_BUILD=true
    fi
done

DOCKERFILE_PATH="$(dirname "$0")/dockerfile.kaldi.cpu"
CONTEXT_PATH="$(dirname "$0")/.."

echo "======================================"
echo "Building TS-DIA Training Docker Image"
echo "======================================"
echo "Image name: ${IMAGE_NAME}"
echo "Dockerfile: ${DOCKERFILE_PATH}"
echo "Context: ${CONTEXT_PATH}"
echo ""

# Check if image already exists
if docker images ${IMAGE_NAME}:latest --format "{{.Repository}}:{{.Tag}}" | grep -q "${IMAGE_NAME}:latest"; then
    if [ "$FORCE_BUILD" = false ]; then
        echo "✓ Image '${IMAGE_NAME}:latest' already exists!"
        echo ""
        echo "Skipping build. To rebuild, use:"
        echo "  ./docker/kaldi_build.sh ${IMAGE_NAME} --force"
        echo ""
        echo "To run the container, use:"
        echo "  ./docker/kaldi_run_cpu.sh ${IMAGE_NAME}"
        echo ""
        exit 0
    else
        echo "⚠ Image exists but --force flag provided. Rebuilding..."
        echo ""
    fi
else
    echo "Image not found. Building from scratch..."
    echo ""
fi

# Build the Docker image
docker build \
    -t ${IMAGE_NAME}:latest \
    -f ${DOCKERFILE_PATH} \
    ${CONTEXT_PATH}

echo ""
echo "======================================"
echo "Build completed successfully!"
echo "======================================"
echo "Image: ${IMAGE_NAME}:latest"
echo ""
echo "To run the container, use:"
echo "  ./docker/kaldi_run_cpu.sh ${IMAGE_NAME}"
echo ""


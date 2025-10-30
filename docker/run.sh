#!/bin/bash
# Script to run the TS-DIA training container with flexible GPU configuration
# Usage: 
#   ./run.sh [IMAGE_NAME] [CONTAINER_NAME] [GPU_IDS]
#
# Examples:
#   ./run.sh                                    # Use all GPUs (e.g., 8 GPUs)
#   ./run.sh ts-dia-training                    # Specify image, use all GPUs
#   ./run.sh ts-dia-training my-training 0      # Use GPU 0 only
#   ./run.sh ts-dia-training my-training 0,1    # Use GPU 0 and 1
#   ./run.sh ts-dia-training my-training all    # Use all 8 GPUs explicitly

set -e

# Load volume configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/volume_config.sh"

# Parse arguments
IMAGE_NAME=${1:-ts-dia-training}
CONTAINER_NAME=${2:-ts-dia-training-$(date +%Y%m%d-%H%M%S)}
GPU_IDS=${3:-all}

# Set GPU device based on input
if [ "$GPU_IDS" = "all" ]; then
    GPU_FLAG="--gpus all"
else
    GPU_FLAG="--gpus \"device=${GPU_IDS}\""
fi

echo "======================================"
echo "Running TS-DIA Training Container"
echo "======================================"
echo "Image: ${IMAGE_NAME}:latest"
echo "Container: ${CONTAINER_NAME}"
echo "GPU configuration: ${GPU_IDS}"
echo ""
echo "Volume Mounts:"
echo "  Fast Storage: ${FAST_STORAGE} -> /storage/fast"
echo "  Slow Storage: ${SLOW_STORAGE} -> /storage/slow"
echo "  Temp Storage: ${TEMP_STORAGE} -> /storage/temp"
echo ""

# Run the container with entire workspace and storage volumes mounted
eval docker run -it --rm \
    ${GPU_FLAG} \
    --user $(id -u):$(id -g) \
    -e HOME=/tmp \
    -e USER=$(whoami) \
    -w /workspace \
    --ipc=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name ${CONTAINER_NAME} \
    -v "$(pwd):/workspace" \
    ${VOLUME_MOUNTS} \
    -p 6006:6006 \
    -p 8888:8888 \
    ${IMAGE_NAME}:latest

echo ""
echo "Container exited."


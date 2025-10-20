#!/bin/bash
# Script to run training directly in a container (non-interactive)
# Usage: 
#   ./run_training.sh [CONFIG_FILE] [IMAGE_NAME] [CONTAINER_NAME] [GPU_IDS]
#
# Examples:
#   ./run_training.sh configs/train/training_example.yml
#   ./run_training.sh configs/train/full_experiment.yml ts-dia-training my-exp 0,1
#   ./run_training.sh configs/train/encoder_model.yml ts-dia-training exp1 all  # All 8 GPUs

set -e

# Load volume configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/volume_config.sh"

# Parse arguments
CONFIG_FILE=${1:-configs/train/training_example.yml}
IMAGE_NAME=${2:-ts-dia-training}
CONTAINER_NAME=${3:-ts-dia-training-$(date +%Y%m%d-%H%M%S)}
GPU_IDS=${4:-all}

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Set GPU device based on input
if [ "$GPU_IDS" = "all" ]; then
    GPU_FLAG="--gpus all"
else
    GPU_FLAG="--gpus \"device=${GPU_IDS}\""
fi

echo "======================================"
echo "Running TS-DIA Training"
echo "======================================"
echo "Image: ${IMAGE_NAME}:latest"
echo "Container: ${CONTAINER_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "GPU configuration: ${GPU_IDS}"
echo ""

# Run training in container with entire workspace and storage volumes mounted
eval docker run --rm \
    ${GPU_FLAG} \
    --ipc=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name ${CONTAINER_NAME} \
    -v "$(pwd):/workspace" \
    ${VOLUME_MOUNTS} \
    -p 6006:6006 \
    ${IMAGE_NAME}:latest \
    python train.py --config ${CONFIG_FILE}

echo ""
echo "Training completed."


#!/bin/bash
# Script to run distributed training using Accelerate
# Usage: 
#   ./run_distributed.sh [CONFIG_FILE] [NUM_GPUS] [IMAGE_NAME] [CONTAINER_NAME] [ACCELERATE_CONFIG]
#
# Examples:
#   ./run_distributed.sh configs/train/training_example.yml 4  # Use 4 out of 8 GPUs
#   ./run_distributed.sh configs/train/full_experiment.yml 8 ts-dia-training my-ddp-exp  # Use all 8 GPUs
#   ./run_distributed.sh configs/train/encoder_model.yml 4 ts-dia-training my-exp configs/accelerate_config.yaml

set -e

# Load volume configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/volume_config.sh"

# Parse arguments
CONFIG_FILE=${1:-configs/train/training_example.yml}
NUM_GPUS=${2:-$(nvidia-smi --list-gpus 2>/dev/null | wc -l)}
IMAGE_NAME=${3:-ts-dia-training}
CONTAINER_NAME=${4:-ts-dia-ddp-$(date +%Y%m%d-%H%M%S)}
ACCELERATE_CONFIG=${5:-}

# Validate config file
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "======================================"
echo "Running Distributed TS-DIA Training"
echo "======================================"
echo "Image: ${IMAGE_NAME}:latest"
echo "Container: ${CONTAINER_NAME}"
echo "Config: ${CONFIG_FILE}"
echo "Number of GPUs: ${NUM_GPUS}"
if [ -n "$ACCELERATE_CONFIG" ]; then
    echo "Accelerate Config: ${ACCELERATE_CONFIG}"
fi
echo ""

# Build accelerate command
if [ -n "$ACCELERATE_CONFIG" ]; then
    # Validate accelerate config file exists
    if [ ! -f "$ACCELERATE_CONFIG" ]; then
        echo "Error: Accelerate config file not found: $ACCELERATE_CONFIG"
        exit 1
    fi
    ACCELERATE_CMD="accelerate launch --config_file ${ACCELERATE_CONFIG} train.py --config ${CONFIG_FILE}"
else
    ACCELERATE_CMD="accelerate launch --num_processes=${NUM_GPUS} train.py --config ${CONFIG_FILE}"
fi

# Run distributed training with Accelerate and entire workspace and storage volumes mounted
docker run --rm \
    --gpus all \
    --user $(id -u):$(id -g) \
    -e HOME=/tmp \
    -e USER=nvidia \
    -w /workspace \
    --ipc=host \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --name ${CONTAINER_NAME} \
    -v "$(pwd):/workspace" \
    ${VOLUME_MOUNTS} \
    -p 6006:6006 \
    ${IMAGE_NAME}:latest \
    bash -c "${ACCELERATE_CMD}"

echo ""
echo "Distributed training completed."


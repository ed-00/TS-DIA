#!/bin/bash
# Script to run CPU-only container for data preparation and preprocessing
# Usage: 
#   ./run_cpu.sh [IMAGE_NAME] [CONTAINER_NAME] [COMMAND]
#
# Examples:
#   ./run_cpu.sh                                    # Interactive shell
#   ./run_cpu.sh ts-dia-training my-prep            # Named interactive session
#   ./run_cpu.sh ts-dia-training prep "python data_manager/data_manager.py --help"

set -e

# Load volume configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/volume_config.sh"

# Parse arguments
IMAGE_NAME=${1:-ts-dia-training}
CONTAINER_NAME=${2:-ts-dia-cpu-$(date +%Y%m%d-%H%M%S)}
COMMAND=${3:-/bin/bash}

echo "======================================"
echo "Running TS-DIA CPU-Only Container"
echo "======================================"
echo "Image: ${IMAGE_NAME}:latest"
echo "Container: ${CONTAINER_NAME}"
echo "Mode: CPU-only (no GPU access)"
echo ""

# Run the container without GPU access
if [ "$COMMAND" = "/bin/bash" ]; then
    # Interactive mode
    docker run -it --rm \
        --cpus="$(nproc)" \
        --user $(id -u):$(id -g) \
        -e HOME=/tmp \
        -e USER=$(whoami) \
        -w /workspace \
        --ipc=host \
        --shm-size=16g \
        --name ${CONTAINER_NAME} \
        -v "$(pwd):/workspace" \
        ${VOLUME_MOUNTS} \
        ${IMAGE_NAME}:latest \
        ${COMMAND}
else
    # Command mode
    echo "Command: ${COMMAND}"
    docker run --rm \
        --cpus="$(nproc)" \
        --user $(id -u):$(id -g) \
        -e HOME=/tmp \
        -e USER=$(whoami) \
        -w /workspace \
        --ipc=host \
        --shm-size=16g \
        --name ${CONTAINER_NAME} \
        -v "$(pwd):/workspace" \
        ${VOLUME_MOUNTS} \
        ${IMAGE_NAME}:latest \
        bash -c "${COMMAND}"
fi

echo ""
echo "Container exited."


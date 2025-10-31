#!/bin/bash
# Script to run CPU-only container for data preparation and preprocessing
# Usage: 
#   ./run_cpu.sh [IMAGE_NAME] [CONTAINER_NAME] [COMMAND]
#
# Examples:
#   ./kaldi_run_cpu.sh                                    # Interactive shell
#   ./kaldi_run_cpu.sh kaldi-ts-dia my-prep            # Named interactive session
#   ./kaldi_run_cpu.sh kaldi-ts-dia prep "python data_manager/data_manager.py --help"

set -e

# Load volume configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/volume_config.sh"

# Parse arguments
IMAGE_NAME=${1:-kaldi-ts-dia}
CONTAINER_NAME=${2:-kaldi-ts-dia-cpu-$(date +%Y%m%d-%H%M%S)}
COMMAND=${3:-/bin/bash}

# Export host user info so container can resolve UID/GID to names and avoid "I have no name!"
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(id -un)
PASSWD_MOUNTS="-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"

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
        --user ${HOST_UID}:${HOST_GID} \
        -e HOME=/tmp \
        -e USER=${HOST_USER} \
        ${PASSWD_MOUNTS} \
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
        --user ${HOST_UID}:${HOST_GID} \
        -e HOME=/tmp \
        -e USER=${HOST_USER} \
        ${PASSWD_MOUNTS} \
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


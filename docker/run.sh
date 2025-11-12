#!/bin/bash
# Script to run the TS-DIA training container with flexible GPU configuration and memory limits
# Usage:
#   ./run.sh [IMAGE_NAME] [CONTAINER_NAME] [GPU_IDS]
#
# Examples:
#   ./run.sh                                    # Use all GPUs (e.g., 8 GPUs)
#   ./run.sh ts-dia-training                    # Specify image, use all GPUs
#   ./run.sh ts-dia-training my-training 0      # Use GPU 0 only
#   ./run.sh ts-dia-training my-training 0,1    # Use GPU 0 and 1
#   ./run.sh ts-dia-training my-training all    # Use all GPUs explicitly

set -euo pipefail

# Load volume configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/volume_config.sh"

# Parse arguments
IMAGE_NAME=${1:-ts-dia-training}
CONTAINER_NAME=${2:-ts-dia-training-$(date +%Y%m%d-%H%M%S)}
GPU_IDS=${3:-all}

# Export host user info so container can resolve UID/GID to names and avoid "I have no name!"
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(id -un)
PASSWD_MOUNTS="-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"

# Compute memory cap = 80% of host RAM (read from /proc/meminfo, MiB)
HOST_MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
MEM_LIMIT_MB=$(( HOST_MEM_MB * 80 / 100 ))
MEM_LIMIT="${MEM_LIMIT_MB}m"              # e.g., "24576m"
SWAP_LIMIT="${SWAP_LIMIT:-${MEM_LIMIT}}"  # equal to MEM_LIMIT to avoid swap thrash

# Optional: cap PIDs to avoid fork storms
PIDS_LIMIT="${PIDS_LIMIT:-4096}"

# Set GPU device based on input (no eval needed)
if [ "$GPU_IDS" = "all" ]; then
  GPU_FLAG=(--gpus all)
else
  GPU_FLAG=(--gpus "device=${GPU_IDS}")
fi

echo "======================================"
echo "Running TS-DIA Training Container"
echo "======================================"
echo "Image: ${IMAGE_NAME}:latest"
echo "Container: ${CONTAINER_NAME}"
echo "GPU configuration: ${GPU_IDS}"
echo ""
echo "Host memory: ${HOST_MEM_MB} MiB  -> Limit (80%): ${MEM_LIMIT}"
echo ""
echo "Volume Mounts:"
echo "  Fast Storage: ${FAST_STORAGE} -> /storage/fast"
echo "  Slow Storage: ${SLOW_STORAGE} -> /storage/slow"
echo "  Temp Storage: ${TEMP_STORAGE} -> /storage/temp"
echo ""

# Run the container with workspace and storage volumes mounted
docker run -it --rm \
  "${GPU_FLAG[@]}" \
  --user "${HOST_UID}:${HOST_GID}" \
  -e HOME=/tmp \
  -e USER="${HOST_USER}" \
  -e HOST_UID="${HOST_UID}" \
  -e HOST_GID="${HOST_GID}" \
  -w /workspace \
  --ipc=host \
  --shm-size=16g \
  --memory "${MEM_LIMIT}" \
  --memory-swap "${SWAP_LIMIT}" \
  --pids-limit "${PIDS_LIMIT}" \
  --name "${CONTAINER_NAME}" \
  -v "$(pwd):/workspace" \
  ${PASSWD_MOUNTS} \
  ${VOLUME_MOUNTS} \
  -p 6006:6006 \
  -p 8888:8888 \
  "${IMAGE_NAME}:latest"

echo ""
echo "Container exited."

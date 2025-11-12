#!/bin/bash
# Script to run CPU-only container for data preparation and preprocessing with memory limits
# Usage:
#   ./run_cpu.sh [IMAGE_NAME] [CONTAINER_NAME] [COMMAND]
#
# Examples:
#   ./run_cpu.sh                                    # Interactive shell
#   ./run_cpu.sh ts-dia-training my-prep            # Named interactive session
#   ./run_cpu.sh ts-dia-training prep "python data_manager/data_manager.py --help"

set -euo pipefail

# Load volume configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/volume_config.sh"

# Parse arguments
IMAGE_NAME=${1:-ts-dia-training}
CONTAINER_NAME=${2:-ts-dia-cpu-$(date +%Y%m%d-%H%M%S)}
COMMAND=${3:-/bin/bash}

# Export host user info so container can resolve UID/GID to names and avoid "I have no name!"
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(id -un)
PASSWD_MOUNTS="-v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro"

# Compute memory cap = 80% of host RAM (MiB), and set swap equal to mem by default
HOST_MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
MEM_LIMIT_MB=$(( HOST_MEM_MB * 80 / 100 ))
MEM_LIMIT="${MEM_LIMIT_MB}m"              # e.g., "24576m"
SWAP_LIMIT="${SWAP_LIMIT:-${MEM_LIMIT}}"  # equal to MEM_LIMIT to avoid swap bursts
PIDS_LIMIT="${PIDS_LIMIT:-4096}"          # optional guard against fork storms

echo "======================================"
echo "Running TS-DIA CPU-Only Container"
echo "======================================"
echo "Image: ${IMAGE_NAME}:latest"
echo "Container: ${CONTAINER_NAME}"
echo "Mode: CPU-only (no GPU access)"
echo "Host memory: ${HOST_MEM_MB} MiB  -> Limit (80%): ${MEM_LIMIT}"
echo ""

# Common docker args
common_args=(
  --cpus="$(nproc)"
  --user "${HOST_UID}:${HOST_GID}"
  -e HOME=/tmp
  -e USER="${HOST_USER}"
  -w /workspace
  --ipc=host
  --shm-size=16g
  --memory "${MEM_LIMIT}"
  --memory-swap "${SWAP_LIMIT}"
  --pids-limit "${PIDS_LIMIT}"
  --name "${CONTAINER_NAME}"
  -v "$(pwd):/workspace"
)

if [ "$COMMAND" = "/bin/bash" ]; then
  # Interactive mode
  docker run -it --rm \
    "${common_args[@]}" \
    ${PASSWD_MOUNTS} \
    ${VOLUME_MOUNTS} \
    "${IMAGE_NAME}:latest" \
    ${COMMAND}
else
  # Command mode
  echo "Command: ${COMMAND}"
  docker run --rm \
    "${common_args[@]}" \
    ${PASSWD_MOUNTS} \
    ${VOLUME_MOUNTS} \
    "${IMAGE_NAME}:latest" \
    bash -c "${COMMAND}"
fi

echo ""
echo "Container exited."

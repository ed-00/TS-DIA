#!/bin/bash
# Volume configuration loader
# This script loads volume paths from volumes.env or uses defaults

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load volumes.env if it exists
if [ -f "$SCRIPT_DIR/volumes.env" ]; then
    source "$SCRIPT_DIR/volumes.env"
fi

# Set defaults if not configured
FAST_STORAGE=${FAST_STORAGE:-"/sbt-fast"}
SLOW_STORAGE=${SLOW_STORAGE:-"/sbt-slow"}
TEMP_STORAGE=${TEMP_STORAGE:-"/sbt-temp"}

# Build volume mount arguments for Docker
VOLUME_MOUNTS="-v ${FAST_STORAGE}:/storage/fast -v ${SLOW_STORAGE}:/storage/slow -v ${TEMP_STORAGE}:/storage/temp"

# Export for use in other scripts
export FAST_STORAGE
export SLOW_STORAGE
export TEMP_STORAGE
export VOLUME_MOUNTS


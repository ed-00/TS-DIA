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
FAST_STORAGE=${FAST_STORAGE:-"$PROJECT_ROOT/sbt-fast"}
SLOW_STORAGE=${SLOW_STORAGE:-"$PROJECT_ROOT/sbt-slow"}
TEMP_STORAGE=${TEMP_STORAGE:-"$PROJECT_ROOT/sbt-temp"}

# Create directories if they don't exist (for defaults)
if [[ "$FAST_STORAGE" == "$PROJECT_ROOT/storage/fast" ]]; then
    mkdir -p "$FAST_STORAGE"
fi
if [[ "$SLOW_STORAGE" == "$PROJECT_ROOT/storage/slow" ]]; then
    mkdir -p "$SLOW_STORAGE"
fi
if [[ "$TEMP_STORAGE" == "$PROJECT_ROOT/storage/temp" ]]; then
    mkdir -p "$TEMP_STORAGE"
fi

# Build volume mount arguments for Docker
VOLUME_MOUNTS="-v ${FAST_STORAGE}:/storage/fast -v ${SLOW_STORAGE}:/storage/slow -v ${TEMP_STORAGE}:/storage/temp"

# Export for use in other scripts
export FAST_STORAGE
export SLOW_STORAGE
export TEMP_STORAGE
export VOLUME_MOUNTS


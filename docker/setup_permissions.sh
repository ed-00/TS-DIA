#!/bin/bash
# Script to set executable permissions on all Docker scripts
# Run this once after cloning the repository

echo "Setting executable permissions on Docker scripts..."

chmod +x docker/build.sh
chmod +x docker/run.sh
chmod +x docker/run_training.sh
chmod +x docker/run_distributed.sh
chmod +x docker/run_cpu.sh
chmod +x docker/volume_config.sh

echo "Done! All scripts are now executable."
echo ""
echo "You can now run:"
echo "  ./docker/build.sh"
echo "  ./docker/run.sh"
echo "  etc."


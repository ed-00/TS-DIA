#!/bin/bash
# Installation script for TS-DIA dependencies
# This script installs system-level dependencies required for training pipeline

set -e  # Exit on error

echo "=========================================="
echo "Installing TS-DIA System Dependencies"
echo "=========================================="

# Update package list
echo "Updating package list..."
apt-get update -y

# Install ffmpeg (includes ffprobe) for video/audio processing
echo "Installing ffmpeg for video/audio processing..."
apt-get install -y ffmpeg

# Install wget for dataset downloads
echo "Installing wget for dataset downloads..."
apt-get install -y wget

# Install gdown for Google Drive downloads (for AVA-AVD annotations)
echo "Installing gdown for Google Drive downloads..."
pip install --upgrade gdown

# Install other useful utilities
echo "Installing additional utilities..."
apt-get install -y curl git zip unzip

# Verify installations
echo ""
echo "=========================================="
echo "Verifying Installations"
echo "=========================================="

echo -n "ffmpeg: "
ffmpeg -version | head -n 1

echo -n "ffprobe: "
ffprobe -version | head -n 1

echo -n "wget: "
wget --version | head -n 1

echo -n "gdown: "
gdown --version || echo "gdown installed (no version flag)"

echo ""
echo "=========================================="
echo "All dependencies installed successfully!"
echo "=========================================="


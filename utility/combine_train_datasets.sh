#!/bin/bash

# Script to combine all training datasets with 100,000 mixtures
# Creates combined files for supervisions and recordings separately

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MANIFESTS_DIR="/workspace/outputs/manifests"
OUTPUT_DIR="/workspace/outputs/manifests/combos"
COMBINE_SCRIPT="/workspace/utility/combine_data.py"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if file exists
check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

# Create output directory
print_info "Creating output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Check if combine script exists
if [ ! -f "$COMBINE_SCRIPT" ]; then
    print_error "Combine script not found: $COMBINE_SCRIPT"
    exit 1
fi

print_info "Starting dataset combination process..."

# Define arrays for train supervision files (100,000 mixtures only)
TRAIN_SUPERVISION_FILES=(
    "$MANIFESTS_DIR/simu_1spk/simu_1spk_supervisions_train_b2_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_2spk/simu_2spk_supervisions_train_b2_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_3spk/simu_3spk_supervisions_train_b5_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_4spk/simu_4spk_supervisions_train_b9_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_5spk/simu_5spk_supervisions_train_b13_mix100000.jsonl.gz"
)

# Define arrays for train recording files (100,000 mixtures only)
TRAIN_RECORDING_FILES=(
    "$MANIFESTS_DIR/simu_1spk/simu_1spk_recordings_train_b2_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_2spk/simu_2spk_recordings_train_b2_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_3spk/simu_3spk_recordings_train_b5_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_4spk/simu_4spk_recordings_train_b9_mix100000.jsonl.gz"
    "$MANIFESTS_DIR/simu_5spk/simu_5spk_recordings_train_b13_mix100000.jsonl.gz"
)

# Function to validate all files exist
validate_files() {
    local files=("$@")
    local missing_files=0
    
    for file in "${files[@]}"; do
        if ! check_file "$file"; then
            missing_files=$((missing_files + 1))
        fi
    done
    
    return $missing_files
}

# Validate supervision files
print_info "Validating train supervision files..."
if ! validate_files "${TRAIN_SUPERVISION_FILES[@]}"; then
    print_error "Some train supervision files are missing. Aborting."
    exit 1
fi
print_success "All train supervision files found"

# Validate recording files
print_info "Validating train recording files..."
if ! validate_files "${TRAIN_RECORDING_FILES[@]}"; then
    print_error "Some train recording files are missing. Aborting."
    exit 1
fi
print_success "All train recording files found"

# Combine train datasets using paired manifests to maintain alignment
print_info "Combining train datasets with paired manifests..."
print_info "This ensures recordings and supervisions stay properly aligned"
print_info "Output file: simu_1_to_5spk_train_combined.jsonl.gz"

python3 "$COMBINE_SCRIPT" \
    --paired-manifests \
    --recording-files "${TRAIN_RECORDING_FILES[@]}" \
    --supervision-files "${TRAIN_SUPERVISION_FILES[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --output-name "simu_1_to_5spk_train_combined.jsonl.gz" \
    --verbose \
    --random-seed 42

if [ $? -eq 0 ]; then
    print_success "Train datasets combined successfully with proper alignment"
else
    print_error "Failed to combine train datasets"
    exit 1
fi

# Display summary
print_success "Dataset combination completed!"
print_info "Combined file created with proper recording-supervision alignment:"
echo "  - $OUTPUT_DIR/simu_1_to_5spk_train_combined.jsonl.gz"

# Display file sizes
print_info "File size:"
if [ -f "$OUTPUT_DIR/simu_1_to_5spk_train_combined.jsonl.gz" ]; then
    echo "  - Combined dataset: $(du -h "$OUTPUT_DIR/simu_1_to_5spk_train_combined.jsonl.gz" | cut -f1)"
fi

print_success "All operations completed successfully!"
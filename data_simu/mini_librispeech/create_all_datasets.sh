#!/bin/bash

# Script to create all datasets according to the experimental setup
# Based on the table in mini-librispeech-data-sim.md

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to run dataset generation
generate_dataset() {
    local dataset_name="$1"
    local split="$2"
    local num_speakers="$3"
    local num_mixtures="$4"
    local beta="$5"
    
    print_status "Generating ${dataset_name} ${split} - Speakers: ${num_speakers}, Mixtures: ${num_mixtures}, β: ${beta}"
    
    # Create output directory if it doesn't exist
    output_dir="data/simu_${dataset_name,,}_${split,,}_ns${num_speakers}_beta${beta}_${num_mixtures}"
    
    # Check if dataset already exists
    if [ -d "$output_dir" ] && [ -f "$output_dir/.done" ]; then
        print_warning "Dataset ${dataset_name} ${split} already exists, skipping..."
        return 0
    fi
    
    # Run the preparation script with appropriate parameters
    if ./run_prepare_shared.sh \
        --num-speaker "$num_speakers" \
        --sil-scale "$beta" \
        --num-train "$num_mixtures" \
        --stage 0; then
        
        # Mark as done
        touch "$output_dir/.done"
        print_success "Completed ${dataset_name} ${split}"
    else
        print_error "Failed to generate ${dataset_name} ${split}"
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting dataset generation according to experimental setup"
    print_status "=========================================================="
    
    # Change to the script directory
    cd "$(dirname "$0")"
    
    # Check if run_prepare_shared.sh exists
    if [ ! -f "run_prepare_shared.sh" ]; then
        print_error "run_prepare_shared.sh not found in current directory"
        exit 1
    fi
    
    # Make sure the script is executable
    chmod +x run_prepare_shared.sh
    
    # Dataset generation according to the table
    
    # Sim1spk
    print_status "=== Generating Sim1spk datasets ==="
    generate_dataset "Sim1spk" "Train" 1 100000 2
    generate_dataset "Sim1spk" "Test" 1 100000 2
    
    # Sim2spk  
    print_status "=== Generating Sim2spk datasets ==="
    generate_dataset "Sim2spk" "Train" 2 100000 2
    generate_dataset "Sim2spk" "Test" 2 500 2
    generate_dataset "Sim2spk" "Test" 2 500 3
    generate_dataset "Sim2spk" "Test" 2 500 5
    
    # Sim3spk
    print_status "=== Generating Sim3spk datasets ==="
    generate_dataset "Sim3spk" "Train" 3 100000 5
    generate_dataset "Sim3spk" "Test" 3 500 5
    generate_dataset "Sim3spk" "Test" 3 500 7
    generate_dataset "Sim3spk" "Test" 3 500 11
    
    # Sim4spk
    print_status "=== Generating Sim4spk datasets ==="
    generate_dataset "Sim4spk" "Train" 4 100000 9
    generate_dataset "Sim4spk" "Test" 4 500 9
    
    # Sim5spk
    print_status "=== Generating Sim5spk datasets ==="
    generate_dataset "Sim5spk" "Train" 5 100000 13
    generate_dataset "Sim5spk" "Test" 5 500 13
    
    print_success "=========================================================="
    print_success "All datasets generated successfully!"
    
    # Summary
    echo ""
    print_status "Generated datasets summary:"
    echo "- Sim1spk: 2 datasets (Train: 100K, Test: 100K mixtures)"
    echo "- Sim2spk: 4 datasets (Train: 100K, Test: 3×500 mixtures)"  
    echo "- Sim3spk: 4 datasets (Train: 100K, Test: 3×500 mixtures)"
    echo "- Sim4spk: 2 datasets (Train: 100K, Test: 500 mixtures)"
    echo "- Sim5spk: 2 datasets (Train: 100K, Test: 500 mixtures)"
    echo "Total: 14 datasets"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Generate all datasets according to the experimental setup table."
    echo ""
    echo "Options:"
    echo "  --help, -h     Show this help message"
    echo "  --dry-run      Show what would be generated without actually running"
    echo "  --resume       Resume from where it left off (skip existing datasets)"
    echo ""
    echo "Datasets to be generated:"
    echo "  Sim1spk: Train(1spk,100K,β=2), Test(1spk,100K,β=2)"
    echo "  Sim2spk: Train(2spk,100K,β=2), Test(2spk,500,β=2/3/5)"
    echo "  Sim3spk: Train(3spk,100K,β=5), Test(3spk,500,β=5/7/11)"
    echo "  Sim4spk: Train(4spk,100K,β=9), Test(4spk,500,β=9)"
    echo "  Sim5spk: Train(5spk,100K,β=13), Test(5spk,500,β=13)"
}

# Parse command line arguments
DRY_RUN=false
RESUME=true  # Default to resume mode

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            show_help
            exit 0
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    print_status "DRY RUN MODE - No datasets will be generated"
    echo ""
    echo "Would generate the following datasets:"
    echo "1.  Sim1spk Train: 1 speaker, 100,000 mixtures, β=2"
    echo "2.  Sim1spk Test:  1 speaker, 100,000 mixtures, β=2"
    echo "3.  Sim2spk Train: 2 speakers, 100,000 mixtures, β=2"
    echo "4.  Sim2spk Test:  2 speakers, 500 mixtures, β=2"
    echo "5.  Sim2spk Test:  2 speakers, 500 mixtures, β=3"
    echo "6.  Sim2spk Test:  2 speakers, 500 mixtures, β=5"
    echo "7.  Sim3spk Train: 3 speakers, 100,000 mixtures, β=5"
    echo "8.  Sim3spk Test:  3 speakers, 500 mixtures, β=5"
    echo "9.  Sim3spk Test:  3 speakers, 500 mixtures, β=7"
    echo "10. Sim3spk Test:  3 speakers, 500 mixtures, β=11"
    echo "11. Sim4spk Train: 4 speakers, 100,000 mixtures, β=9"
    echo "12. Sim4spk Test:  4 speakers, 500 mixtures, β=9"
    echo "13. Sim5spk Train: 5 speakers, 100,000 mixtures, β=13"
    echo "14. Sim5spk Test:  5 speakers, 500 mixtures, β=13"
    exit 0
fi

# Run main function
main
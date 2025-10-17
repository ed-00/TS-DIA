#!/bin/bash
# Test Suite Execution Script for TS-DIA
# Runs all 7 test configurations and verifies checkpoint structure

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log file
LOG_DIR="./test_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_LOG="$LOG_DIR/test_summary_${TIMESTAMP}.log"

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  TS-DIA Test Suite Execution${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo "Log directory: $LOG_DIR"
echo "Summary log: $SUMMARY_LOG"
echo ""

# Initialize summary
echo "TS-DIA Test Suite Summary - $(date)" > "$SUMMARY_LOG"
echo "======================================" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"

# Test configurations
declare -a tests=(
    "configs/test/test_ava_base_8khz.yml"
    "configs/test/test_ava_linear_attention.yml"
    "configs/test/test_ava_optimizer_variants.yml"
    "configs/test/test_ava_loss_variants.yml"
    "configs/test/test_ava_precomputed_mfcc.yml"
    "configs/test/test_ava_small_model.yml"
    "configs/test/test_ava_checkpoint_resume.yml"
)

declare -a test_names=(
    "Test 1: Baseline 8kHz on-the-fly features"
    "Test 2: Linear attention (Performer)"
    "Test 3: Optimizer variants (AdamW, cosine)"
    "Test 4: Loss variants (auxiliary losses)"
    "Test 5: Precomputed MFCC with bucketing"
    "Test 6: Small model with dynamic bucketing"
    "Test 7: Checkpoint resumption"
)

# Track results
PASSED=0
FAILED=0
declare -a failed_tests

# Function to verify checkpoint structure
verify_checkpoint_structure() {
    local checkpoint_dir=$1
    local test_name=$2
    
    echo -e "\n${YELLOW}Verifying checkpoint structure for: $test_name${NC}"
    
    if [ ! -d "$checkpoint_dir" ]; then
        echo -e "${RED}✗ Checkpoint directory not found: $checkpoint_dir${NC}"
        return 1
    fi
    
    # Check for numbered checkpoints (should NOT exist after fix)
    if [ -d "$checkpoint_dir/checkpoints" ]; then
        local numbered_count=$(find "$checkpoint_dir/checkpoints" -maxdepth 1 -type d -name "checkpoint_*" 2>/dev/null | wc -l)
        if [ "$numbered_count" -gt 0 ]; then
            echo -e "${RED}✗ Found $numbered_count numbered checkpoints (should be 0 after fix)${NC}"
            echo "  Numbered checkpoints found in: $checkpoint_dir/checkpoints"
            return 1
        else
            echo -e "${GREEN}✓ No numbered checkpoints found (expected after fix)${NC}"
        fi
    fi
    
    # Check for named checkpoints (should exist)
    local named_count=$(find "$checkpoint_dir" -maxdepth 1 -type d -name "checkpoint-epoch*-step*" 2>/dev/null | wc -l)
    if [ "$named_count" -eq 0 ]; then
        echo -e "${RED}✗ No named checkpoints found${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Found $named_count named checkpoint(s)${NC}"
    
    # Verify checkpoint contents
    for ckpt_dir in "$checkpoint_dir"/checkpoint-epoch*-step*; do
        if [ -d "$ckpt_dir" ]; then
            echo "  Checking: $(basename $ckpt_dir)"
            
            # Check required files
            local required_files=("model.safetensors" "optimizer.bin" "scheduler.bin" "extra_state.pt")
            for file in "${required_files[@]}"; do
                if [ -f "$ckpt_dir/$file" ]; then
                    echo -e "    ${GREEN}✓${NC} $file"
                else
                    echo -e "    ${RED}✗${NC} $file (missing)"
                fi
            done
            
            # Check for sampler state in extra_state.pt
            if [ -f "$ckpt_dir/extra_state.pt" ]; then
                echo -e "    ${GREEN}✓${NC} extra_state.pt exists (contains sampler state)"
            fi
        fi
    done
    
    return 0
}

# Function to list directory structure recursively
list_checkpoint_structure() {
    local checkpoint_dir=$1
    local test_name=$2
    
    echo -e "\n${BLUE}Recursive structure of: $checkpoint_dir${NC}"
    if [ -d "$checkpoint_dir" ]; then
        find "$checkpoint_dir" -type f -o -type d | head -50
    else
        echo "Directory does not exist"
    fi
}

# Run tests
for i in "${!tests[@]}"; do
    test_config="${tests[$i]}"
    test_name="${test_names[$i]}"
    test_num=$((i + 1))
    
    echo -e "\n${BLUE}======================================${NC}"
    echo -e "${BLUE}Running: $test_name${NC}"
    echo -e "${BLUE}Config: $test_config${NC}"
    echo -e "${BLUE}======================================${NC}"
    
    log_file="$LOG_DIR/test_${test_num}_$(date +%Y%m%d_%H%M%S).log"
    
    # Extract checkpoint directory from config and clean it (new experiment management requires this)
    checkpoint_dir=$(grep -A 1 "checkpoint:" "$test_config" | grep "save_dir:" | awk '{print $2}')
    if [ -n "$checkpoint_dir" ] && [ -d "$checkpoint_dir" ]; then
        echo "Cleaning old checkpoint directory: $checkpoint_dir"
        rm -rf "$checkpoint_dir"
    fi
    
    # Run the test
    if python train.py --config "$test_config" > "$log_file" 2>&1; then
        echo -e "${GREEN}✓ Test passed: $test_name${NC}"
        echo "[PASS] $test_name" >> "$SUMMARY_LOG"
        PASSED=$((PASSED + 1))
        
        # Extract checkpoint directory from config
        checkpoint_dir=$(grep -A 1 "checkpoint:" "$test_config" | grep "save_dir:" | awk '{print $2}')
        
        # Verify checkpoint structure
        if verify_checkpoint_structure "$checkpoint_dir" "$test_name"; then
            echo -e "${GREEN}✓ Checkpoint structure verified${NC}"
            echo "  [PASS] Checkpoint structure verified" >> "$SUMMARY_LOG"
        else
            echo -e "${YELLOW}⚠ Checkpoint structure verification failed${NC}"
            echo "  [WARN] Checkpoint structure verification failed" >> "$SUMMARY_LOG"
        fi
        
        # List checkpoint structure
        list_checkpoint_structure "$checkpoint_dir" "$test_name" >> "$SUMMARY_LOG"
        
    else
        echo -e "${RED}✗ Test failed: $test_name${NC}"
        echo "[FAIL] $test_name" >> "$SUMMARY_LOG"
        echo "  Log: $log_file" >> "$SUMMARY_LOG"
        FAILED=$((FAILED + 1))
        failed_tests+=("$test_name")
        
        # Show last 20 lines of error log
        echo -e "${RED}Last 20 lines of error log:${NC}"
        tail -n 20 "$log_file"
    fi
    
    echo "" >> "$SUMMARY_LOG"
done

# Verify precomputed features cache
echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Verifying Precomputed Features Cache${NC}"
echo -e "${BLUE}======================================${NC}"

if [ -d "./features/ava_avd" ]; then
    echo -e "${GREEN}✓ Features cache directory exists${NC}"
    echo "" >> "$SUMMARY_LOG"
    echo "Precomputed Features Cache:" >> "$SUMMARY_LOG"
    find ./features/ava_avd -type f | head -20 >> "$SUMMARY_LOG"
else
    echo -e "${YELLOW}⚠ Features cache directory not found${NC}"
    echo "" >> "$SUMMARY_LOG"
    echo "[WARN] Features cache directory not found" >> "$SUMMARY_LOG"
fi

# Final summary
echo -e "\n${BLUE}======================================${NC}"
echo -e "${BLUE}Test Suite Summary${NC}"
echo -e "${BLUE}======================================${NC}"
echo -e "Total tests: ${#tests[@]}"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

echo "" >> "$SUMMARY_LOG"
echo "======================================" >> "$SUMMARY_LOG"
echo "Final Results:" >> "$SUMMARY_LOG"
echo "  Total: ${#tests[@]}" >> "$SUMMARY_LOG"
echo "  Passed: $PASSED" >> "$SUMMARY_LOG"
echo "  Failed: $FAILED" >> "$SUMMARY_LOG"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${failed_tests[@]}"; do
        echo -e "  ${RED}✗${NC} $test"
    done
    echo "" >> "$SUMMARY_LOG"
    echo "Failed tests:" >> "$SUMMARY_LOG"
    for test in "${failed_tests[@]}"; do
        echo "  - $test" >> "$SUMMARY_LOG"
    done
    exit 1
else
    echo -e "\n${GREEN}All tests passed!${NC}"
    echo "" >> "$SUMMARY_LOG"
    echo "All tests passed!" >> "$SUMMARY_LOG"
fi

echo -e "\nFull summary saved to: $SUMMARY_LOG"


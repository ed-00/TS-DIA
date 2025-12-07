#!/bin/bash

# This script validates all development sets in the cache directory.

set -e # Exit immediately if a command exits with a non-zero status.

# Find all files matching the pattern for dev set cuts and validate them.
# The pattern looks for 'cuts_windowed.jsonl.gz' in any directory that starts with 'dev_'
# under any directory that starts with 'simu_' inside './cache/'.
find ./cache/simu_* -path '*/train_*/cuts_windowed.jsonl.gz' -print0 | while IFS= read -r -d $'\0' file; do
    echo "Validating $file"
    lhotse validate --dont-read-data "$file"
done

echo "All dev sets validated successfully."

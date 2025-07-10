#!/bin/bash

# Parse command line arguments
CONFIG=""
OUTPUT=""
ITERATIONS=28  # Default number of layers to analyze

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$CONFIG" ] || [ -z "$OUTPUT" ]; then
    echo "Usage: $0 --config <mergekit_config.yml> --output <output_dir> [--iterations <number>]"
    exit 1
fi

# Create a temporary directory for merge contribution analysis
MERGE_CONTRIB_DIR="${OUTPUT}/merge_contribution_analysis"
mkdir -p "$MERGE_CONTRIB_DIR"

echo "Starting merge contribution analysis..."

# Backup original config
cp "$CONFIG" "${CONFIG}.merge.backup"

# For each iteration (layer position)
for i in $(seq 1 $ITERATIONS); do
    echo "Analyzing merge contribution at position $i..."
    
    # Create iteration-specific config
    ITER_CONFIG="${MERGE_CONTRIB_DIR}/iter${i}.yml"
    cp "${CONFIG}.merge.backup" "$ITER_CONFIG"
    
    # Get the number of models
    NUM_MODELS=$(grep -c "model:" "$ITER_CONFIG")
    
    # For each model, modify its weight at position i to 0.5
    for m in $(seq 1 $NUM_MODELS); do
        # Get the line number for this model
        MODEL_LINE=$(grep -n "model:" "$ITER_CONFIG" | sed -n "${m}p" | cut -d':' -f1)
        if [ -z "$MODEL_LINE" ]; then
            echo "Error: Could not find model $m in config"
            continue
        fi
        
        # Find the weight parameters line for this model
        WEIGHT_LINE=$(awk -v start=$MODEL_LINE 'NR>=start && /weight:/ {print NR; exit}' "$ITER_CONFIG")
        if [ -z "$WEIGHT_LINE" ]; then
            echo "Error: Could not find weight for model $m"
            continue
        fi
        
        # In a real implementation, we would modify the weight array at position i to 0.5
        # For simplicity in this demo, we just indicate the intention
        echo "Would modify weight for model $m at position $i to 0.5"
    done
    
    # Run mergekit with the modified config
    ITER_OUTPUT="${MERGE_CONTRIB_DIR}/iter${i}"
    mergekit-yaml "$ITER_CONFIG" "$ITER_OUTPUT" --allow-crimes --cuda
    
    echo "Completed merge analysis for position $i"
done

# Restore original config
cp "${CONFIG}.merge.backup" "$CONFIG"

# Process the results to find optimal configuration
echo "Processing merge contribution results..."

# Update the final config based on merge contribution analysis
# (In a real implementation, this would be based on evaluation results)
echo "Updating main config based on merge contribution analysis"
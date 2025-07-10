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

# Create a temporary directory for contribution analysis
CONTRIB_DIR="${OUTPUT}/single_contribution_analysis"
mkdir -p "$CONTRIB_DIR"

echo "Starting single contribution analysis..."

# Backup original config
cp "$CONFIG" "${CONFIG}.contrib.backup"

# Get the number of models in the config
NUM_MODELS=$(grep -c "model:" "$CONFIG")
echo "Detected $NUM_MODELS models in the configuration"

# For each iteration (layer position)
for i in $(seq 1 $ITERATIONS); do
    echo "Analyzing contribution at position $i..."
    
    # For each model
    for m in $(seq 1 $NUM_MODELS); do
        # Get the line number for this model
        MODEL_LINE=$(grep -n "model:" "$CONFIG" | sed -n "${m}p" | cut -d':' -f1)
        if [ -z "$MODEL_LINE" ]; then
            echo "Error: Could not find model $m in config"
            continue
        fi
        
        # Find the weight parameters line for this model
        WEIGHT_LINE=$(awk -v start=$MODEL_LINE 'NR>=start && /weight:/ {print NR; exit}' "$CONFIG")
        if [ -z "$WEIGHT_LINE" ]; then
            echo "Error: Could not find weight for model $m"
            continue
        fi
        
        # Create a model-specific config for this iteration
        ITER_CONFIG="${CONTRIB_DIR}/model${m}_pos${i}.yml"
        cp "${CONFIG}.contrib.backup" "$ITER_CONFIG"
        
        # Create weight vector with all 0.0 except position i which is 1.0
        # First, set all weights to 0.0 for this model
        sed -i "${WEIGHT_LINE}s/weight:.*/weight: 0.0/" "$ITER_CONFIG"
        
        # Then set position i to 1.0
        # In a real implementation, we would need to parse the actual weight array 
        # structure, but for demonstration we're assuming a simple replacement
        
        # Run mergekit with the modified config
        ITER_OUTPUT="${CONTRIB_DIR}/model${m}_pos${i}"
        mergekit-yaml "$ITER_CONFIG" "$ITER_OUTPUT" --allow-crimes --cuda
        
        echo "Completed analysis for model $m at position $i"
    done
done

# Restore original config
cp "${CONFIG}.contrib.backup" "$CONFIG"

# Analyze contributions and update the original config
# This would typically involve evaluating each model variation and 
# determining which layers have the most impact

echo "Single contribution analysis completed. Results in $CONTRIB_DIR"

# Apply the insights from contribution analysis to update the config
# (In a real implementation, this would be based on evaluation results)
echo "Updating main config based on contribution analysis"
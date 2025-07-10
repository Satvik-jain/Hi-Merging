#!/bin/bash

# Parse command line arguments
CONFIG=""
OUTPUT=""
ITERATIONS=10

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

# Create a temporary directory for model pruning experiments
TEMP_DIR="${OUTPUT}/model_wise_experiments"
mkdir -p "$TEMP_DIR"

# Calculate density and weight step based on iterations
DENSITY_STEP=$(echo "scale=2; 1.0/$ITERATIONS" | bc)
WEIGHT_STEP=$(echo "scale=2; 1.0/$ITERATIONS" | bc)

echo "Starting model-wise pruning and scaling experiments..."
echo "Using density step: $DENSITY_STEP, weight step: $WEIGHT_STEP"

# Start with lower densities (more pruning) and gradually increase
for i in $(seq 1 $ITERATIONS); do
    DENSITY=$(echo "scale=2; $i*$DENSITY_STEP" | bc)
    for j in $(seq 1 $ITERATIONS); do
        WEIGHT=$(echo "scale=2; $j*$WEIGHT_STEP" | bc)
        
        echo "Running experiment with density=$DENSITY, weight=$WEIGHT"
        
        # Create a copy of the config file for this experiment
        EXP_CONFIG="${TEMP_DIR}/config_d${DENSITY}_w${WEIGHT}.yml"
        cp "$CONFIG" "$EXP_CONFIG"
        
        # Update the density and weight values in all models
        for MODEL_IDX in $(grep -n "model:" "$EXP_CONFIG" | cut -d':' -f1); do
            # Find the parameters section below this model
            PARAMS_LINE=$((MODEL_IDX + 1))
            WEIGHT_LINE=$(awk -v start=$PARAMS_LINE 'NR>=start && /weight:/ {print NR; exit}' "$EXP_CONFIG")
            DENSITY_LINE=$(awk -v start=$PARAMS_LINE 'NR>=start && /density:/ {print NR; exit}' "$EXP_CONFIG")
            
            if [ -n "$WEIGHT_LINE" ] && [ -n "$DENSITY_LINE" ]; then
                # Update the weight and density values
                sed -i "${WEIGHT_LINE}s/weight:.*/weight: $WEIGHT/" "$EXP_CONFIG"
                sed -i "${DENSITY_LINE}s/density:.*/density: $DENSITY/" "$EXP_CONFIG"
            fi
        done
        
        # Run mergekit with the updated config
        EXP_OUTPUT="${TEMP_DIR}/merge_d${DENSITY}_w${WEIGHT}"
        mergekit-yaml "$EXP_CONFIG" "$EXP_OUTPUT" --allow-crimes --cuda
        
        # Optionally run evaluation here to find optimal values
        # For now, we'll just save the parameters with the merged model
        echo "{ \"density\": $DENSITY, \"weight\": $WEIGHT }" > "${EXP_OUTPUT}/params.json"
    done
done

echo "Model-wise experiments completed. Results in $TEMP_DIR"

# Find the best performing model based on evaluation (placeholder)
# In a real implementation, you would analyze evaluation results
BEST_CONFIG="${TEMP_DIR}/best_config.yml"
cp "$CONFIG" "$BEST_CONFIG"

# Apply the best parameters to the original config
echo "Updating main config with optimal parameters"
cp "$BEST_CONFIG" "$CONFIG"

echo "Model-wise pruning and scaling completed."
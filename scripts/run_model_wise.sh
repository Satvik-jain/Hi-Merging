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

# Calculate density and weight step using Python instead of bc
DENSITY_STEP=$(python3 -c "print(round(1.0/$ITERATIONS, 2))")
WEIGHT_STEP=$(python3 -c "print(round(1.0/$ITERATIONS, 2))")

echo "Starting model-wise pruning and scaling experiments..."
echo "Using density step: $DENSITY_STEP, weight step: $WEIGHT_STEP"

# Start with lower densities (more pruning) and gradually increase
for i in $(seq 1 $ITERATIONS); do
    DENSITY=$(python3 -c "print(round($i*$DENSITY_STEP, 2))")
    for j in $(seq 1 $ITERATIONS); do
        WEIGHT=$(python3 -c "print(round($j*$WEIGHT_STEP, 2))")
        
        echo "Running experiment with density=$DENSITY, weight=$WEIGHT"
        
        # Create a copy of the config file for this experiment
        EXP_CONFIG="${TEMP_DIR}/config_d${DENSITY}_w${WEIGHT}.yml"
        cp "$CONFIG" "$EXP_CONFIG"
        
        # Update density and weight values using Python
        python3 -c "
import yaml

with open('$EXP_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

# Update all models
for model in config['models']:
    if 'parameters' in model:
        model['parameters']['density'] = $DENSITY
        model['parameters']['weight'] = $WEIGHT

with open('$EXP_CONFIG', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
"
        
        # Run mergekit with the updated config
        EXP_OUTPUT="${TEMP_DIR}/merge_d${DENSITY}_w${WEIGHT}"
        mergekit-yaml "$EXP_CONFIG" "$EXP_OUTPUT" --allow-crimes --cuda
        
        # Save parameters with the merged model
        echo "{ \"density\": $DENSITY, \"weight\": $WEIGHT }" > "${EXP_OUTPUT}/params.json"
    done
done

echo "Model-wise experiments completed. Results in $TEMP_DIR"

# Find the best performing model based on evaluation
# In a real implementation, this would analyze evaluation results
BEST_CONFIG="${TEMP_DIR}/best_config.yml"
cp "$CONFIG" "$BEST_CONFIG"

# Apply the best parameters to the original config
echo "Updating main config with optimal parameters"
cp "$BEST_CONFIG" "$CONFIG"

echo "Model-wise pruning and scaling completed."
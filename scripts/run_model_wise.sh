#!/bin/bash
# run_model_wise.sh - Applies initial pruning and scaling to a model
#
# Usage: ./run_model_wise.sh <model_path> <base_model> <output_path> <scale> <density>

set -e  # Exit on error

MODEL_PATH="$1"
BASE_MODEL="$2"
OUTPUT_PATH="$3"
SCALE="$4"
DENSITY="$5"

# Check if all arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$BASE_MODEL" ] || [ -z "$OUTPUT_PATH" ] || [ -z "$SCALE" ] || [ -z "$DENSITY" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: ./run_model_wise.sh <model_path> <base_model> <output_path> <scale> <density>"
    exit 1
fi

echo "[Info] Running model-wise pruning and scaling with:"
echo "  - Model: $MODEL_PATH"
echo "  - Base model: $BASE_MODEL" 
echo "  - Output path: $OUTPUT_PATH"
echo "  - Scale: $SCALE"
echo "  - Density: $DENSITY"

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Create temporary YAML config file for mergekit
TEMP_CONFIG="$(dirname "$OUTPUT_PATH")/temp_model_wise_config.yml"
cat > "$TEMP_CONFIG" << EOL
merge_method: ties
base_model: $BASE_MODEL
models:
  - model: $MODEL_PATH
    parameters:
      weight: $SCALE
      density: $DENSITY
dtype: float16
tokenizer:
  source: base
EOL

# Run mergekit
echo "[Info] Running mergekit to apply model-wise pruning and scaling..."
CUDA_VISIBLE_DEVICES=0,1,2,3 mergekit-yaml "$TEMP_CONFIG" "$OUTPUT_PATH" --allow-crimes --cuda

# Clean up
rm "$TEMP_CONFIG"

echo "[Success] Model-wise pruning and scaling complete. Output saved to $OUTPUT_PATH"
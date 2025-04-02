#!/bin/bash

# Base Directory
BASE_DIR="/data/user/PycharmProjects"

# Number of positions in the density list
NUM_POSITIONS=28

# File Paths
TIES_YML="$BASE_DIR/mergekit/examples/ties.yml"
LLAMA_YML="$BASE_DIR/LLaMA-Factory/examples/train_lora/qwen2_lora_predict.yaml"

# Backup original files
cp "$TIES_YML" "${TIES_YML}.orig"
cp "$LLAMA_YML" "${LLAMA_YML}.orig"

# Get the line number of the cmedqa2-30k model
model_line=$(grep -n -m1 'cmexam' "$TIES_YML" | cut -d':' -f1)
if [ -z "$model_line" ]; then
  echo "Error: Could not find cmexam model in ties.yml"
  exit 1
fi

# Get the line number of the density list under cmedqa2-30k
density_line=$(awk -v mline="$model_line" 'NR > mline && /weight:/ {print NR; exit}' "$TIES_YML")
if [ -z "$density_line" ]; then
  echo "Error: Could not find density line under cmedqa2-30k in ties.yml"
  exit 1
fi

for ((i=1; i<=NUM_POSITIONS; i++)); do
  echo "Processing iteration $i..."

  # Restore original ties.yml at the start of each iteration
  cp "${TIES_YML}.orig" "$TIES_YML"

  # Modify the density list in ties.yml
  # Replace the i-th occurrence of 0.8 with 0.4
  sed -i "${density_line}s/\(0\.0\)/1.0/${i}" "$TIES_YML"

  # Modify llama3_lora_predict.yaml
  cp "${LLAMA_YML}.orig" "$LLAMA_YML"
  OUTPUT_DIR="saves/qwen2-7b/lora/predict/cmexam_iteration_$i"
  sed -i "s|^output_dir:.*|output_dir: $OUTPUT_DIR|" "$LLAMA_YML"

  # Run mergekit-yaml command
  cd "$BASE_DIR/mergekit"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mergekit-yaml ./examples/ties.yml ./output_model/qwen2_lora_sft/merge_en_zh --allow-crimes --cuda

  # Run llamafactory-cli train command
  cd "$BASE_DIR/LLaMA-Factory"
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train examples/train_lora/qwen2_lora_predict.yaml

  echo "Iteration $i completed."
done

# Restore original files after all iterations
cp "${TIES_YML}.orig" "$TIES_YML"
cp "${LLAMA_YML}.orig" "$LLAMA_YML"

echo "All iterations completed."

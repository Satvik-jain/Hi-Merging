#!/bin/bash

# ------------------------
# 1) Basic Settings
# ------------------------

# Path of the YAML file to process
TIES_YML="/data/user/PycharmProjects/mergekit/examples/ties.yml"

# YAML configuration for training
LLAMA_YML="/data/user/PycharmProjects/LLaMA-Factory/examples/train_lora/qwen2_lora_predict.yaml"

# Output directory prefix (suffix will be automatically added during iteration)
OUTPUT_DIR_PREFIX="saves/qwen2-7b/lora/predict/en_iteration_"

# Root directories for MergeKit and LLaMA-Factory projects
MERGEKIT_DIR="/data/user/PycharmProjects/mergekit"
LLAMAFACTORY_DIR="/data/user/PycharmProjects/LLaMA-Factory"

# Length of the weight array (number of iterations to loop through)
NUM_POSITIONS=28

# ------------------------
# 2) Backup Files
# ------------------------
echo "[Info] Backing up original configuration files..."
cp "$TIES_YML" "${TIES_YML}.orig"
cp "$LLAMA_YML" "${LLAMA_YML}.orig"

# ------------------------
# 3) Find the corresponding lines of weights for 'en' and 'cmexam'
#    (Assume stable yml structure, first locate the model line, then search for 'weight:')
# ------------------------

# Locate the line number for "qwen2_lora_sft/en"
EN_LINE=$(grep -n -m1 'qwen2_lora_sft/en' "$TIES_YML" | cut -d':' -f1)
if [ -z "$EN_LINE" ]; then
  echo "[Error] Could not find configuration for EN model (qwen2_lora_sft/en)"
  exit 1
fi

# Find the line number for 'weight:' below EN_LINE
EN_WEIGHT_LINE=$(awk -v mline="$EN_LINE" 'NR > mline && /weight:/ {print NR; exit}' "$TIES_YML")
if [ -z "$EN_WEIGHT_LINE" ]; then
  echo "[Error] Could not find weight line for EN model"
  exit 1
fi

# Locate the line number for "qwen2_lora_sft/cmexam"
CMEXAM_LINE=$(grep -n -m1 'qwen2_lora_sft/cmexam' "$TIES_YML" | cut -d':' -f1)
if [ -z "$CMEXAM_LINE" ]; then
  echo "[Error] Could not find configuration for CMEXAM model (qwen2_lora_sft/cmexam)"
  exit 1
fi

# Find the line number for 'weight:' below CMEXAM_LINE
CMEXAM_WEIGHT_LINE=$(awk -v mline="$CMEXAM_LINE" 'NR > mline && /weight:/ {print NR; exit}' "$TIES_YML")
if [ -z "$CMEXAM_WEIGHT_LINE" ]; then
  echo "[Error] Could not find weight line for CMEXAM model"
  exit 1
fi

# ------------------------
# 4) Loop to modify ties.yml, execute merging and training
# ------------------------
for ((i=1; i<=NUM_POSITIONS; i++)); do
  echo "------------------------------"
  echo "[Info] Starting iteration $i..."
  echo "------------------------------"

  # Restore original ties.yml for each iteration
  cp "${TIES_YML}.orig" "$TIES_YML"

  # ------------------------
  # 4.1) Modify ties.yml
  # 
  # For the i-th iteration:
  #  - Replace the i-th occurrence of "0.0" in EN weight array with "0.5"
  #  - Replace the i-th occurrence of "0.0" in CMEXAM weight array with "0.5"
  # 
  # Note the sed syntax "s/old/new/N" means replacing the N-th occurrence of 'old'
  # ------------------------
  sed -i "${EN_WEIGHT_LINE}s/0.0/0.5/${i}" "$TIES_YML"
  sed -i "${CMEXAM_WEIGHT_LINE}s/0.0/0.5/${i}" "$TIES_YML"

  # ------------------------
  # 4.2) Update output directory in qwen2_lora_predict.yaml
  # ------------------------
  cp "${LLAMA_YML}.orig" "$LLAMA_YML"
  THIS_OUTPUT_DIR="${OUTPUT_DIR_PREFIX}${i}"
  sed -i "s|^output_dir:.*|output_dir: ${THIS_OUTPUT_DIR}|" "$LLAMA_YML"

  # ------------------------
  # 4.3) Execute MergeKit
  # ------------------------
  echo "[Info] Executing MergeKit..."
  cd "$MERGEKIT_DIR" || exit 1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mergekit-yaml \
    ./examples/ties.yml \
    ./output_model/qwen2_lora_sft/merge_en_zh \
    --allow-crimes --cuda

  # ------------------------
  # 4.4) Execute LLaMA-Factory training
  # ------------------------
  echo "[Info] Executing LLaMA-Factory training..."
  cd "$LLAMAFACTORY_DIR" || exit 1
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train \
    examples/train_lora/qwen2_lora_predict.yaml

  echo "[Info] Iteration $i completed."
done

# ------------------------
# 5) Restore original files
# ------------------------
echo "[Info] All iterations completed. Restoring original configuration files."
cp "${TIES_YML}.orig" "$TIES_YML"
cp "${LLAMA_YML}.orig" "$LLAMA_YML"

echo "[Done] Script execution finished."

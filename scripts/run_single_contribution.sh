#!/bin/bash
# run_single_contribution.sh - Calculate single model contribution
#
# Usage: ./run_single_contribution.sh <model_path> <base_model> <output_json>

set -e  # Exit on error

MODEL_PATH="$1"
BASE_MODEL="$2"
OUTPUT_JSON="$3"

# Check if all arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$BASE_MODEL" ] || [ -z "$OUTPUT_JSON" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: ./run_single_contribution.sh <model_path> <base_model> <output_json>"
    exit 1
fi

echo "[Info] Calculating contribution for single model:"
echo "  - Model: $MODEL_PATH"
echo "  - Base model: $BASE_MODEL"
echo "  - Output JSON: $OUTPUT_JSON"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT_JSON")"

# Calculate contributions using Python
python3 -c "
import torch
import json
from transformers import AutoModelForCausalLM
import numpy as np

def analyze_contribution(model_path, base_model_path, output_json):
    print('Loading models...')
    # Load models with half precision to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map='auto',
        torch_dtype=torch.float16
    )
    
    contributions = {}
    
    # Analyze each layer
    print('Analyzing layer contributions...')
    for name, param in model.named_parameters():
        if not ('weight' in name or 'bias' in name):
            continue
            
        base_param = base.get_parameter(name)
        delta = param - base_param
        
        # Calculate alpha: magnitude of change (normalized)
        alpha = delta.abs().mean().item()
        
        # Calculate beta: impact on performance (approximated via sparsity)
        # In the Hi-Merging paper, this measures how much the delta affects model output
        # Here we use sparsity as a proxy measure
        sparsity = (delta.abs() < delta.abs().mean()).float().mean().item()
        beta = 1.0 - sparsity  # Higher beta = more impact
        
        contributions[name] = {
            'alpha': float(alpha),
            'beta': float(beta),
            'conflict_score': float(alpha * beta)
        }
    
    print('Saving contribution analysis...')
    with open(output_json, 'w') as f:
        json.dump(contributions, f, indent=2)

# Run analysis
analyze_contribution('$MODEL_PATH', '$BASE_MODEL', '$OUTPUT_JSON')
print('Contribution analysis complete!')
"

echo "[Success] Contribution analysis complete. Results saved to $OUTPUT_JSON"
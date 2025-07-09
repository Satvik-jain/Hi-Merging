#!/bin/bash
# run_merge_contribution.sh - Calculate contributions after merging
#
# Usage: ./run_merge_contribution.sh <merged_model> <base_model> <output_json>

set -e  # Exit on error

MERGED_MODEL="$1"
BASE_MODEL="$2"
OUTPUT_JSON="$3"

# Check if all arguments are provided
if [ -z "$MERGED_MODEL" ] || [ -z "$BASE_MODEL" ] || [ -z "$OUTPUT_JSON" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: ./run_merge_contribution.sh <merged_model> <base_model> <output_json>"
    exit 1
fi

echo "[Info] Calculating contribution for merged model:"
echo "  - Merged model: $MERGED_MODEL"
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

def analyze_merge_contribution(merged_model_path, base_model_path, output_json):
    print('Loading models...')
    # Load models with half precision to save memory
    merged = AutoModelForCausalLM.from_pretrained(
        merged_model_path, 
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
    print('Analyzing merged model contributions...')
    for name, param in merged.named_parameters():
        if not ('weight' in name or 'bias' in name):
            continue
            
        base_param = base.get_parameter(name)
        delta = param - base_param
        
        # Calculate alpha: magnitude of change (normalized)
        alpha = delta.abs().mean().item()
        
        # Calculate beta: impact on performance (approximated via sparsity)
        # In the Hi-Merging paper, this measures how much the delta affects model output
        sparsity = (delta.abs() < delta.abs().mean()).float().mean().item()
        beta = 1.0 - sparsity  # Higher beta = more impact
        
        # Calculate conflict score
        # According to Hi-Merging paper, alpha*beta can indicate conflict level
        # Positive values indicate positive contribution
        # Negative values would indicate conflicts (not implemented in this approximation)
        conflict_score = alpha * beta
        
        # Determine conflict type (for layer-specific adjustments)
        # In Hi-Merging: if conflict detected, decide whether to prune or scale
        conflict_type = 'prune' if alpha > beta else 'scale'
        
        contributions[name] = {
            'alpha': float(alpha),
            'beta': float(beta),
            'conflict_score': float(conflict_score),
            'conflict_type': conflict_type
        }
    
    # Sort layers by conflict score to find the most problematic ones
    sorted_layers = sorted(
        [(name, data['conflict_score']) for name, data in contributions.items()],
        key=lambda x: x[1]
    )
    
    # Store top conflicted layers for easier access
    contributions['_top_conflicts'] = [name for name, _ in sorted_layers[:10]]
    
    print('Saving merge contribution analysis...')
    with open(output_json, 'w') as f:
        json.dump(contributions, f, indent=2)

# Run analysis
analyze_merge_contribution('$MERGED_MODEL', '$BASE_MODEL', '$OUTPUT_JSON')
print('Merge contribution analysis complete!')
"

echo "[Success] Merge contribution analysis complete. Results saved to $OUTPUT_JSON"
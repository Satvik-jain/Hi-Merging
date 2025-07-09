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
import os
from transformers import AutoModelForCausalLM
import numpy as np

def analyze_merge_contribution(merged_model_path, base_model_path, output_json):
    print('Loading models...')
    
    # Configure model loading to handle HF hub models
    load_kwargs = {
        'torch_dtype': torch.float16,
        'device_map': 'auto',
        'trust_remote_code': True,
    }
    
    # Add token for HF Hub if needed (base model might be from Hub)
    if '/' in base_model_path and not os.path.isdir(base_model_path):
        token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        if token:
            print('Using HF token for base model download')
            load_kwargs['token'] = token
    
    try:
        # Load merged model
        print(f'Loading merged model: {merged_model_path}')
        merged = AutoModelForCausalLM.from_pretrained(merged_model_path, **load_kwargs)
        
        # Load base model
        print(f'Loading base model: {base_model_path}')
        base = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
        
        # Initialize contribution dictionary
        contributions = {}
        
        # Analyze each layer
        print('Analyzing layer contributions...')
        for name, param in merged.named_parameters():
            if not param.requires_grad:
                continue
                
            # Only process parameters that exist in both models
            if name not in dict(base.named_parameters()):
                continue
                
            base_param = base.get_parameter(name)
            delta = param.detach() - base_param.detach()
            
            # Calculate alpha: magnitude of change (normalized)
            alpha = delta.abs().mean().item()
            
            # Calculate beta: impact on performance (approximated via sparsity)
            sparsity = (delta.abs() < delta.abs().mean()).float().mean().item()
            beta = 1.0 - sparsity
            
            # Calculate conflict score
            conflict_score = alpha * beta
            conflict_type = 'prune' if alpha > beta else 'scale'
            
            contributions[name] = {
                'alpha': float(alpha),
                'beta': float(beta),
                'conflict_score': float(conflict_score),
                'conflict_type': conflict_type
            }
        
        # Store top conflicted layers for easier access
        sorted_layers = sorted(
            [(name, data['conflict_score']) for name, data in contributions.items()],
            key=lambda x: x[1]
        )
        contributions['_top_conflicts'] = [name for name, _ in sorted_layers[:10]]
        
        print('Saving merge contribution analysis...')
        with open(output_json, 'w') as f:
            json.dump(contributions, f, indent=2)
            
    except Exception as e:
        print(f'Error during merge contribution analysis: {e}')
        raise e

# Run analysis
analyze_merge_contribution('$MERGED_MODEL', '$BASE_MODEL', '$OUTPUT_JSON')
print('Merge contribution analysis complete!')
"

echo "[Success] Merge contribution analysis complete. Results saved to $OUTPUT_JSON"
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
import os
from transformers import AutoModelForCausalLM
import numpy as np

def analyze_contribution(model_path, base_model_path, output_json):
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
        # Load fine-tuned model
        print(f'Loading model: {model_path}')
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        # Load base model
        print(f'Loading base model: {base_model_path}')
        base = AutoModelForCausalLM.from_pretrained(base_model_path, **load_kwargs)
        
        # Initialize contribution dictionary
        contributions = {}
        
        # Analyze each layer
        print('Analyzing layer contributions...')
        for name, param in model.named_parameters():
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
            beta = 1.0 - sparsity  # Higher beta = more impact
            
            contributions[name] = {
                'alpha': float(alpha),
                'beta': float(beta),
                'conflict_score': float(alpha * beta)
            }
        
        print('Saving contribution analysis...')
        with open(output_json, 'w') as f:
            json.dump(contributions, f, indent=2)
            
    except Exception as e:
        print(f'Error during contribution analysis: {e}')
        raise e

# Run analysis
analyze_contribution('$MODEL_PATH', '$BASE_MODEL', '$OUTPUT_JSON')
print('Contribution analysis complete!')
"

echo "[Success] Contribution analysis complete. Results saved to $OUTPUT_JSON"
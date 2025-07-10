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

# Create a temporary Python file for better error handling
TMP_PY_FILE="/tmp/run_merge_contribution_$$.py"
cat > "$TMP_PY_FILE" << 'EOL'
import torch
import json
import os
import sys
from transformers import AutoModelForCausalLM
import traceback

def analyze_merge_contribution(merged_model_path, base_model_path, output_json):
    try:
        print(f'Loading models for merge contribution analysis')
        
        # Set environment variables for HuggingFace
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        
        # Get HF token from environment
        token = os.environ.get('HUGGING_FACE_HUB_TOKEN') or os.environ.get('HF_TOKEN')
        print(f'HF token available: {"Yes" if token else "No"}')
        
        # Create dummy contribution to ensure we can write output
        dummy_contribution = {"status": "loading models"}
        with open(output_json, 'w') as f:
            json.dump(dummy_contribution, f)
        
        # Load merged model
        print(f'Loading merged model: {merged_model_path}')
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "use_auth_token": token
        }
        merged = AutoModelForCausalLM.from_pretrained(merged_model_path, **model_kwargs)
        
        # Load base model - use same arguments
        print(f'Loading base model: {base_model_path}')
        base = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        
        # Calculate contributions
        print('Analyzing merged model contributions...')
        contributions = {}
        
        for name, param in merged.named_parameters():
            if not ('weight' in name or 'bias' in name):
                continue
                
            # Skip if parameter doesn't exist in base model
            if name not in dict(base.named_parameters()):
                print(f"Skipping {name} - not in base model")
                continue
                
            base_param = base.get_parameter(name)
            delta = param.detach() - base_param.detach()
            
            # Calculate alpha: magnitude of change
            alpha = delta.abs().mean().item()
            
            # Calculate beta: approximated by sparsity
            sparsity = (delta.abs() < delta.abs().mean()).float().mean().item()
            beta = 1.0 - sparsity
            
            # Calculate conflict score
            conflict_score = alpha * beta
            
            # Determine if this layer needs pruning or scaling
            conflict_type = 'prune' if alpha > beta else 'scale'
            
            contributions[name] = {
                'alpha': float(alpha),
                'beta': float(beta),
                'conflict_score': float(conflict_score),
                'conflict_type': conflict_type
            }
        
        # Sort layers by conflict score to find most problematic ones
        sorted_layers = sorted(
            [(name, contributions[name]['conflict_score']) for name in contributions],
            key=lambda x: x[1]
        )
        
        # Add top conflicts for easy reference
        contributions['_top_conflicts'] = [name for name, _ in sorted_layers[:10]]
        
        print(f'Saving merge contribution analysis to {output_json}...')
        with open(output_json, 'w') as f:
            json.dump(contributions, f, indent=2)
        
        print('Merge contribution analysis completed successfully')
        return 0
        
    except Exception as e:
        print(f"ERROR: Exception during merge contribution analysis: {e}")
        traceback.print_exc()
        
        # Write error to output file so it doesn't get lost
        with open(output_json, 'w') as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f, indent=2)
        
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <merged_model_path> <base_model_path> <output_json>")
        sys.exit(1)
    
    merged_model_path = sys.argv[1]
    base_model_path = sys.argv[2]
    output_json = sys.argv[3]
    
    exit_code = analyze_merge_contribution(merged_model_path, base_model_path, output_json)
    sys.exit(exit_code)
EOL

# Run the Python script with proper error handling
echo "[Info] Running merge contribution analysis..."
python3 "$TMP_PY_FILE" "$MERGED_MODEL" "$BASE_MODEL" "$OUTPUT_JSON"
RESULT=$?

# Clean up
rm "$TMP_PY_FILE"

if [ $RESULT -eq 0 ]; then
    echo "[Success] Merge contribution analysis completed. Results saved to $OUTPUT_JSON"
    exit 0
else
    echo "[Error] Merge contribution analysis failed. Check $OUTPUT_JSON for details."
    exit 1
fi
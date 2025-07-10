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

# Create a temporary Python file for better error handling
TMP_PY_FILE="/tmp/run_single_contribution_$$.py"
cat > "$TMP_PY_FILE" << 'EOL'
import torch
import json
import os
import sys
from transformers import AutoModelForCausalLM
import traceback

def analyze_contribution(model_path, base_model_path, output_json):
    try:
        print(f'Loading model from: {model_path}')
        
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
        
        # Load fine-tuned model
        print(f'Loading model: {model_path}')
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "use_auth_token": token
        }
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Load base model - use same arguments
        print(f'Loading base model: {base_model_path}')
        base = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
        
        # Calculate contributions
        print('Analyzing layer contributions...')
        contributions = {}
        
        for name, param in model.named_parameters():
            if not ('weight' in name or 'bias' in name):
                continue
                
            # Skip if parameter doesn't exist in base model
            if name not in dict(base.named_parameters()):
                print(f"Skipping {name} - not in base model")
                continue
                
            base_param = base.get_parameter(name)
            delta = param.detach() - base_param.detach()
            
            # Calculate alpha: magnitude of change (normalized)
            alpha = delta.abs().mean().item()
            
            # Calculate beta: impact on performance (approximated via sparsity)
            # In the Hi-Merging paper, beta measures the parameter's impact
            sparsity = (delta.abs() < delta.abs().mean()).float().mean().item()
            beta = 1.0 - sparsity  # Higher beta = more impact
            
            contributions[name] = {
                'alpha': float(alpha),
                'beta': float(beta),
                'conflict_score': float(alpha * beta)
            }
        
        print(f'Saving contribution analysis to {output_json}...')
        with open(output_json, 'w') as f:
            json.dump(contributions, f, indent=2)
        
        print('Contribution analysis completed successfully')
        return 0
        
    except Exception as e:
        print(f"ERROR: Exception during contribution analysis: {e}")
        traceback.print_exc()
        
        # Write error to output file so it doesn't get lost
        with open(output_json, 'w') as f:
            json.dump({"error": str(e), "traceback": traceback.format_exc()}, f, indent=2)
        
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_path> <base_model_path> <output_json>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    base_model_path = sys.argv[2]
    output_json = sys.argv[3]
    
    exit_code = analyze_contribution(model_path, base_model_path, output_json)
    sys.exit(exit_code)
EOL

# Run the Python script with proper error handling
echo "[Info] Running contribution analysis..."
python3 "$TMP_PY_FILE" "$MODEL_PATH" "$BASE_MODEL" "$OUTPUT_JSON"
RESULT=$?

# Clean up
rm "$TMP_PY_FILE"

if [ $RESULT -eq 0 ]; then
    echo "[Success] Contribution analysis completed. Results saved to $OUTPUT_JSON"
    exit 0
else
    echo "[Error] Contribution analysis failed. Check $OUTPUT_JSON for details."
    exit 1
fi
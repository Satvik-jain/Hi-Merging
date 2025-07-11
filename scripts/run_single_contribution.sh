#!/bin/bash

# Parse command line arguments
CONFIG=""
OUTPUT=""
ITERATIONS=28  # Default number of layers to analyze

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

# Create a temporary directory for contribution analysis
CONTRIB_DIR="${OUTPUT}/single_contribution_analysis"
mkdir -p "$CONTRIB_DIR"

echo "Starting single contribution analysis..."

# Backup original config
cp "$CONFIG" "${CONFIG}.contrib.backup"

# For each iteration (layer position)
for i in $(seq 1 $ITERATIONS); do
    echo "Analyzing contribution at position $i..."
    
    # Python script to modify configs and execute mergekit
    python3 -c "
import yaml
import subprocess
import os

# Load the original config
with open('${CONFIG}.contrib.backup', 'r') as f:
    config = yaml.safe_load(f)

# For each model, create a configuration where only this model contributes at position $i
for idx, model in enumerate(config['models']):
    # Create config name
    iter_config = '${CONTRIB_DIR}/model{}_pos{}.yml'.format(idx, $i)
    
    # Deep copy the config for this iteration
    iter_cfg = dict(config)
    iter_cfg['models'] = [dict(m) for m in config['models']]
    
    # Set all weights to 0.0
    for m in iter_cfg['models']:
        if 'parameters' in m:
            m['parameters'] = dict(m['parameters'])
            m['parameters']['weight'] = 0.0
    
    # For this model at position $i, set weight to 1.0
    if 'parameters' in iter_cfg['models'][idx]:
        iter_cfg['models'][idx]['parameters']['weight'] = 1.0
    
    # Write the config file
    with open(iter_config, 'w') as f:
        yaml.dump(iter_cfg, f, default_flow_style=False)
    
    # Run mergekit for this configuration
    output_dir = '${CONTRIB_DIR}/model{}_pos{}'.format(idx, $i)
    subprocess.run(['mergekit-yaml', iter_config, output_dir, '--allow-crimes', '--cuda'])
    
    print('Completed analysis for model {} at position {}'.format(idx, $i))
"
    
    echo "Completed analysis for position $i"
done

# Restore original config
cp "${CONFIG}.contrib.backup" "$CONFIG"

echo "Single contribution analysis completed. Results in $CONTRIB_DIR"
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

# Create a temporary directory for merge contribution analysis
MERGE_CONTRIB_DIR="${OUTPUT}/merge_contribution_analysis"
mkdir -p "$MERGE_CONTRIB_DIR"

echo "Starting merge contribution analysis..."

# Backup original config
cp "$CONFIG" "${CONFIG}.merge.backup"

# For each iteration (layer position)
for i in $(seq 1 $ITERATIONS); do
    echo "Analyzing merge contribution at position $i..."
    
    # Python script to modify configs and execute mergekit
    python3 -c "
import yaml
import subprocess
import os

# Load the original config
with open('${CONFIG}.merge.backup', 'r') as f:
    config = yaml.safe_load(f)

# Create iteration-specific config
iter_config = '${MERGE_CONTRIB_DIR}/iter{}.yml'.format($i)
iter_output = '${MERGE_CONTRIB_DIR}/iter{}'.format($i)

# Deep copy the config for this iteration
iter_cfg = dict(config)
iter_cfg['models'] = [dict(m) for m in config['models']]

# For each model, set weight at position $i to 0.5
for m in iter_cfg['models']:
    if 'parameters' in m:
        m['parameters'] = dict(m['parameters'])
        m['parameters']['weight'] = 0.5

# Write the config file
with open(iter_config, 'w') as f:
    yaml.dump(iter_cfg, f, default_flow_style=False)

# Run mergekit for this configuration
subprocess.run(['mergekit-yaml', iter_config, iter_output, '--allow-crimes', '--cuda'])

print('Completed merge analysis for position {}'.format($i))
"
    
    echo "Completed merge analysis for position $i"
done

# Restore original config
cp "${CONFIG}.merge.backup" "$CONFIG"

echo "Merge contribution analysis completed. Results in $MERGE_CONTRIB_DIR"
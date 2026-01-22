#!/bin/bash
# Wrapper script to train GPT2 small using HuggingFace Trainer
# This avoids the Flash Attention requirement of flame transformer

MODEL_ID=$1
shift  # Remove first argument, rest are passed to training script

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

# Load model config from registry
python3 -c "
import yaml
import sys
from pathlib import Path

registry_path = Path('scripts/model_registry.yaml')
with open(registry_path) as f:
    registry = yaml.safe_load(f)

if '$MODEL_ID' not in registry['models']:
    print(f'Error: Model {MODEL_ID} not found in registry', file=sys.stderr)
    sys.exit(1)

config = dict(registry['defaults'])
config.update(registry['models']['$MODEL_ID'])

# Build command
cmd = ['python3', 'models/train_gpt2_small.py']
cmd.append('--dataset')
if 'wikitext' in config.get('dataset', ''):
    cmd.append('wikitext-103')
elif 'openwebtext' in config.get('dataset', ''):
    cmd.append('openwebtext')
else:
    cmd.append(config.get('dataset', 'wikitext-103'))

cmd.extend(['--dropout', str(config.get('dropout', 0.0))])
cmd.extend(['--learning_rate', str(config.get('learning_rate', 1e-4))])
cmd.extend(['--weight_decay', str(config.get('weight_decay', 0.01))])
cmd.extend(['--warmup_steps', str(config.get('warmup_steps', 1000))])
cmd.extend(['--batch_size', str(config.get('batch_size', 64))])
cmd.extend(['--seq_len', str(config.get('seq_len', 512))])

if config.get('total_steps'):
    cmd.extend(['--total_steps', str(config['total_steps'])])
else:
    cmd.extend(['--epochs', '10'])

if config.get('skip_samples'):
    cmd.extend(['--skip_samples', str(config['skip_samples'])])

cmd.extend(['--save_steps', str(config.get('checkpoint_interval', 5000))])
cmd.extend(['--eval_steps', str(config.get('checkpoint_interval', 5000))])

if not config.get('enable_wandb', True):
    cmd.append('--nowandb')

print(' '.join(cmd))
" | bash

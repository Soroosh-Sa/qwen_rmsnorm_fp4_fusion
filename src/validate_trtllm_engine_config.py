#!/usr/bin/env python3
"""Validate that an engine directory still has an engine config, not a checkpoint config."""
import argparse, json, os, sys

parser = argparse.ArgumentParser()
parser.add_argument('--engine-dir', required=True)
args = parser.parse_args()
config_path = os.path.join(args.engine_dir, 'config.json')
if not os.path.isfile(config_path):
    print(f'ERROR: missing engine config.json: {config_path}', file=sys.stderr)
    sys.exit(2)
with open(config_path) as f:
    c = json.load(f)
keys = set(c.keys())
engine_like = any(k in keys for k in ['pretrained_config', 'build_config', 'engine_version', 'plugin_config'])
checkpoint_like = ('quantization' in keys and 'mapping' in keys and 'producer' in keys and 'rank' in keys)
print(f'config_path={config_path}')
print(f'top_level_keys={sorted(list(keys))[:50]}')
print(f'engine_like={engine_like}')
print(f'checkpoint_like={checkpoint_like}')
if checkpoint_like and not engine_like:
    print('ERROR: ENGINE_DIR/config.json looks like a TRT-LLM checkpoint config, not an engine config.', file=sys.stderr)
    print('This usually means an older script copied CHECKPOINT_DIR/config.json over the engine config.', file=sys.stderr)
    print('Rebuild with CLEAN_ENGINE_DIR=1 using the v13 scripts.', file=sys.stderr)
    sys.exit(3)
if not engine_like:
    print('WARNING: config.json does not expose common engine keys; continuing, but inspect manually.', file=sys.stderr)
print('OK: engine config.json does not look overwritten by checkpoint config.')

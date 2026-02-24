"""
Ablation Runner — runs all 4 model variants in sequence and collects results.
Variants:
  1. flat_bert         (separate script, run first)
  2. han_no_attn       USE_ATTENTION=False, USE_FOCAL=True
  3. han_cross_entropy USE_ATTENTION=True,  USE_FOCAL=False
  4. han_c_full        USE_ATTENTION=True,  USE_FOCAL=True  ← already trained as hanc_compliance.pt

Usage:
  python3 run_ablation.py            # runs han_no_attn and han_cross_entropy
  python3 run_ablation.py --all      # also runs flat_bert first
"""
import subprocess
import sys
import os
import importlib
import argparse

TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TRAIN_DIR)
sys.path.insert(0, ROOT_DIR)

parser = argparse.ArgumentParser()
parser.add_argument('--all', action='store_true', help='Also run flat_bert baseline')
args = parser.parse_args()


def run_variant(name: str, use_attention: bool, use_focal: bool):
    """Inject overrides into train_hanc.py by patching environment vars and re-running."""
    env = os.environ.copy()
    env['ABLATION_USE_ATTENTION'] = str(int(use_attention))
    env['ABLATION_USE_FOCAL']     = str(int(use_focal))
    env['ABLATION_NAME']          = name
    print(f"\n{'='*60}")
    print(f"Running variant: {name}  (attention={use_attention}, focal={use_focal})")
    print('='*60)
    result = subprocess.run(
        [sys.executable, os.path.join(TRAIN_DIR, 'train_hanc.py')],
        env=env, cwd=TRAIN_DIR
    )
    if result.returncode != 0:
        print(f"[WARNING] {name} exited with code {result.returncode}")


if args.all:
    print("Running Flat BERT baseline first...")
    subprocess.run([sys.executable, os.path.join(TRAIN_DIR, 'flat_bert.py')], cwd=TRAIN_DIR)

# Variant 2: HAN-NoAttn
run_variant('han_no_attn',       use_attention=False, use_focal=True)

# Variant 3: HAN-CrossEntropy
run_variant('han_cross_entropy', use_attention=True,  use_focal=False)

print("\n✅ Ablation variants complete.")
print("HAN-C Full (han_c_full) weights already in models/weights/hanc_compliance.pt")
print("Now run: python3 results_table.py")

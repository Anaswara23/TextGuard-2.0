"""
Standalone test-set evaluator — loads saved weights WITHOUT retraining.
Use this any time you want to re-evaluate a model that is already trained.

Usage:
  python3 eval_model.py                          # evaluates hanc_compliance.pt (default)
  python3 eval_model.py --variant han_no_attn
  python3 eval_model.py --variant han_cross_entropy
  python3 eval_model.py --variant han_c_full
"""
import argparse, os, sys
import torch
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import DistilBertTokenizer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.han_c   import HANC
from models.dataset import IDBDocumentDataset

DEVICE    = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
TASK      = 'compliance'
RESULTS   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
WEIGHTS   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights')
DATA      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'chunks_with_splits.csv')

VARIANT_MAP = {
    'han_c_full':        ('hanc_compliance.pt',    True),
    'han_no_attn':       ('han_no_attn.pt',         False),
    'han_cross_entropy': ('han_cross_entropy.pt',   True),
}

parser = argparse.ArgumentParser()
parser.add_argument('--variant', default='han_c_full', choices=list(VARIANT_MAP.keys()))
args = parser.parse_args()

weight_file, use_attention = VARIANT_MAP[args.variant]
weights_path = os.path.join(WEIGHTS, weight_file)

if not os.path.exists(weights_path):
    print(f"ERROR: {weights_path} not found. Run training first.")
    sys.exit(1)

tokenizer   = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
df          = pd.read_csv(DATA)
test_dataset = IDBDocumentDataset(df[df['split'] == 'test'], tokenizer, task=TASK)

model = HANC(use_attention=use_attention).to(DEVICE)
model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
model.eval()
print(f"Loaded {weight_file} → device={DEVICE}, attention={use_attention}")

all_preds, all_labels, doc_ids = [], [], []
with torch.no_grad():
    for batch in test_dataset:
        ids  = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        logits, _ = model(ids, mask)
        all_preds.append(logits.argmax(dim=-1).item())
        all_labels.append(batch['label'].item())
        doc_ids.append(batch['doc_id'])

label_names = ['Compliant', 'Non-Compliant']
print("\n=== Test Set Results ===")
print(classification_report(all_labels, all_preds, target_names=label_names, labels=[0, 1], zero_division=0))
print("Confusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))
macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
print(f"\nDoc-level Macro-F1: {macro_f1:.4f}")

# Save for results_table.py / t-test
results_df = pd.DataFrame({'doc_id': doc_ids, 'pred': all_preds, 'true': all_labels})
results_df['correct'] = (results_df['pred'] == results_df['true']).astype(int)
out_path = os.path.join(RESULTS, f'{args.variant}_doc_results.csv')
results_df.to_csv(out_path, index=False)
print(f"Per-doc results saved to {out_path}")

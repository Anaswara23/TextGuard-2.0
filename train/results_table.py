"""
Results Table + Paired t-test
Loads all ablation model weights, runs test-set evaluation for each,
prints a Markdown results table, and runs a Wilcoxon signed-rank test
(HAN-C Full vs Flat BERT) on per-document F1 scores.

Usage:
  python3 results_table.py
"""
import torch
import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (f1_score, precision_score, recall_score,
                              classification_report, confusion_matrix)
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import DistilBertTokenizer
from models.han_c  import HANC
from models.dataset import IDBDocumentDataset

DEVICE    = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
TASK      = 'compliance'
RESULTS   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
WEIGHTS   = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'weights')

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'chunks_with_splits.csv')
df        = pd.read_csv(data_path)
test_df   = df[df['split'] == 'test']
test_dataset = IDBDocumentDataset(test_df, tokenizer, task=TASK)
label_names  = ['Compliant', 'Non-Compliant']

def eval_hanc(weight_file: str, use_attention: bool) -> dict:
    path = os.path.join(WEIGHTS, weight_file)
    if not os.path.exists(path):
        print(f"  [SKIP] {weight_file} not found.")
        return None
    model = HANC(use_attention=use_attention).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_dataset:
            ids  = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            logits, _ = model(ids, mask)
            preds.append(logits.argmax(dim=-1).item())
            labels.append(batch['label'].item())
    return compute_metrics(preds, labels)

def eval_flat_bert() -> dict:
    """Load pre-saved per-doc CSV from flat_bert.py"""
    path = os.path.join(RESULTS, 'flat_bert_doc_results.csv')
    if not os.path.exists(path):
        print("  [SKIP] flat_bert_doc_results.csv not found — run flat_bert.py first")
        return None
    rdf    = pd.read_csv(path)
    preds  = rdf['pred'].tolist()
    labels = rdf['true'].tolist()
    return compute_metrics(preds, labels)

def compute_metrics(preds, labels) -> dict:
    macro_f1  = f1_score(labels, preds, average='macro',  zero_division=0)
    micro_f1  = f1_score(labels, preds, average='micro',  zero_division=0)
    nc_prec   = precision_score(labels, preds, pos_label=1, zero_division=0)
    nc_recall = recall_score   (labels, preds, pos_label=1, zero_division=0)
    nc_f1     = f1_score       (labels, preds, pos_label=1, zero_division=0)
    acc       = float(np.mean(np.array(preds) == np.array(labels)))
    return {
        'macro_f1':  round(macro_f1  * 100, 1),
        'micro_f1':  round(micro_f1  * 100, 1),
        'nc_prec':   round(nc_prec   * 100, 1),
        'nc_recall': round(nc_recall * 100, 1),
        'nc_f1':     round(nc_f1     * 100, 1),
        'accuracy':  round(acc       * 100, 1),
        'preds':     preds,
        'labels':    labels,
    }

# ── Evaluate all 4 variants ──
variants = [
    ('Flat BERT (baseline)',   None,            None,    'flat'),
    ('HAN-NoAttn',             'han_no_attn.pt',False,   'han_no_attn'),
    ('HAN-CrossEntropy',       'han_cross_entropy.pt', True, 'han_ce'),
    ('HAN-C Full (proposed)',  'hanc_compliance.pt', True, 'han_c'),
]

rows = []
all_results = {}
for name, weight_file, use_attn, key in variants:
    print(f"\nEvaluating: {name}")
    if weight_file is None:
        res = eval_flat_bert()
    else:
        res = eval_hanc(weight_file, use_attn)

    if res is None:
        rows.append({'Model': name, 'Macro-F1': '—', 'Micro-F1': '—',
                     'NC Prec': '—', 'NC Recall': '—', 'NC F1': '—', 'Acc': '—'})
        continue

    all_results[key] = res
    rows.append({
        'Model':    name,
        'Macro-F1': f"{res['macro_f1']}%",
        'Micro-F1': f"{res['micro_f1']}%",
        'NC Prec':  f"{res['nc_prec']}%",
        'NC Recall':f"{res['nc_recall']}%",
        'NC F1':    f"{res['nc_f1']}%",
        'Acc':      f"{res['accuracy']}%",
    })
    print(classification_report(res['labels'], res['preds'], target_names=label_names))

# ── Print Markdown Table ──
print("\n" + "="*70)
print("## Results Table (Test Set)")
print("="*70)
results_df = pd.DataFrame(rows)
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(RESULTS, 'ablation_results.csv'), index=False)
print(f"\nSaved to results/ablation_results.csv")

# ── Paired Wilcoxon test: HAN-C Full vs Flat BERT ──
if 'flat' in all_results and 'han_c' in all_results:
    print("\n" + "="*70)
    print("## Paired Wilcoxon Signed-Rank Test: HAN-C Full vs Flat BERT")
    print("="*70)
    flat_labels  = all_results['flat']['labels']
    flat_preds   = all_results['flat']['preds']
    hanc_preds   = all_results['han_c']['preds']

    # Per-document binary correct/incorrect
    flat_correct = [int(p == l) for p, l in zip(flat_preds, flat_labels)]
    hanc_correct = [int(p == l) for p, l in zip(hanc_preds, all_results['han_c']['labels'])]

    if flat_correct == hanc_correct:
        print("Identical predictions — t-test not applicable (all differences are 0).")
    else:
        try:
            stat, p_val = wilcoxon(hanc_correct, flat_correct)
            print(f"Wilcoxon statistic: {stat:.4f}")
            print(f"p-value:            {p_val:.4f}")
            if p_val < 0.05:
                print("✅ HAN-C Full significantly outperforms Flat BERT (p < 0.05)")
            else:
                print(f"⚠️  Not significant (p={p_val:.4f}). Consider dataset size.")
        except Exception as e:
            print(f"Could not run Wilcoxon: {e}")
else:
    print("\n[INFO] Run both flat_bert.py and train_hanc.py before running t-test.")

# ── Confusion Matrix Comparison Plot ──
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, key, title in [
    (axes[0], 'flat',  'Flat BERT (Baseline)'),
    (axes[1], 'han_c', 'HAN-C Full (Proposed)'),
]:
    if key not in all_results:
        ax.set_title(f"{title}\n(not available)")
        continue
    cm = confusion_matrix(all_results[key]['labels'], all_results[key]['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, 'confusion_matrix_comparison.png'), dpi=150, bbox_inches='tight')
print("\nSaved confusion matrix comparison to results/confusion_matrix_comparison.png")

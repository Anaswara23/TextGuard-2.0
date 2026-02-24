"""
Flat BERT Baseline — Ablation Model 1
Chunk-level DistilBERT classifier with length-weighted majority vote for document label.
No hierarchy, no attention — pure baseline.
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Config ──
TASK       = 'compliance'
MODEL_NAME = 'distilbert-base-uncased'
LR         = 2e-5
EPOCHS     = 10
DEVICE     = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
SAVE_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'models', 'weights', 'flat_bert.pt')
RESULTS    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

print(f"Device: {DEVICE}")

# ── Chunk-level Dataset (each row = one chunk) ──
class ChunkDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        if TASK == 'compliance':
            label_map = {'Compliant': 0, 'Non-Compliant': 1}
            df = df[df['compliance_status'].isin(label_map)].copy()
            df['label_int'] = df['compliance_status'].map(label_map)
        else:
            label_map = {'no_risk': 0, 'risk': 1}
            df['label_int'] = df['risk_label'].map(label_map).fillna(0).astype(int)

        texts  = df['chunk_text'].fillna('').tolist()
        labels = df['label_int'].tolist()
        doc_ids = df['document_id'].tolist()
        lengths = df['chunk_text'].fillna('').str.len().tolist()

        enc = tokenizer(texts, padding='max_length', truncation=True,
                        max_length=max_len, return_tensors='pt')
        self.input_ids      = enc['input_ids']
        self.attention_mask = enc['attention_mask']
        self.labels         = torch.tensor(labels, dtype=torch.long)
        self.doc_ids        = doc_ids
        self.lengths        = lengths

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids':      self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'label':          self.labels[idx],
            'doc_id':         self.doc_ids[idx],
            'length':         self.lengths[idx],
        }

# ── Model ──
class FlatBERT(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)

# ── Data ──
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'chunks_with_splits.csv')
df        = pd.read_csv(data_path)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

train_ds  = ChunkDataset(df[df['split'] == 'train'], tokenizer)
val_ds    = ChunkDataset(df[df['split'] == 'val'],   tokenizer)
test_ds   = ChunkDataset(df[df['split'] == 'test'],  tokenizer)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False, num_workers=0)

model     = FlatBERT().to(DEVICE)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training ──
train_losses, val_losses, val_f1s = [], [], []
best_val_f1 = 0
patience_counter = 0
PATIENCE = 3

for epoch in range(EPOCHS):
    # Freeze BERT for first 2 epochs
    for param in model.bert.parameters():
        param.requires_grad = (epoch >= 2)

    model.train()
    epoch_loss = 0
    for batch in train_loader:
        ids   = batch['input_ids'].to(DEVICE)
        mask  = batch['attention_mask'].to(DEVICE)
        lbls  = batch['label'].to(DEVICE)
        optimizer.zero_grad()
        logits = model(ids, mask)
        loss   = loss_fn(logits, lbls)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validate at chunk level
    model.eval()
    val_loss = 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids   = batch['input_ids'].to(DEVICE)
            mask  = batch['attention_mask'].to(DEVICE)
            lbls  = batch['label'].to(DEVICE)
            logits = model(ids, mask)
            val_loss += loss_fn(logits, lbls).item()
            preds_all.extend(logits.argmax(dim=-1).cpu().tolist())
            labels_all.extend(lbls.cpu().tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    macro_f1 = f1_score(labels_all, preds_all, average='macro', zero_division=0)
    val_f1s.append(macro_f1)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Macro-F1 (chunk): {macro_f1:.4f}")

    if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model (F1={macro_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

# ── Test — Document-level via length-weighted vote ──
print("\n=== Test Set Evaluation (Document-level, length-weighted) ===")
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
model.eval()

doc_scores = {}   # doc_id -> {'probs': [], 'lengths': [], 'true': int}
with torch.no_grad():
    for batch in test_loader:
        ids   = batch['input_ids'].to(DEVICE)
        mask  = batch['attention_mask'].to(DEVICE)
        probs = torch.softmax(model(ids, mask), dim=-1)[:, 1].cpu().tolist()
        for i in range(len(probs)):
            did = batch['doc_id'][i]
            if did not in doc_scores:
                doc_scores[did] = {'probs': [], 'lengths': [], 'true': batch['label'][i].item()}
            doc_scores[did]['probs'].append(probs[i])
            doc_scores[did]['lengths'].append(batch['length'][i] if isinstance(batch['length'][i], int)
                                              else batch['length'][i].item())

doc_preds, doc_labels = [], []
for did, info in doc_scores.items():
    weights = np.array(info['lengths'], dtype=float)
    weights /= weights.sum()
    score = float(np.dot(weights, info['probs']))
    doc_preds.append(1 if score > 0.5 else 0)
    doc_labels.append(info['true'])

label_names = ['Compliant', 'Non-Compliant'] if TASK == 'compliance' else ['no_risk', 'risk']
print(classification_report(doc_labels, doc_preds, target_names=label_names, labels=[0, 1], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(doc_labels, doc_preds, labels=[0, 1]))

macro_f1_test = f1_score(doc_labels, doc_preds, average='macro', zero_division=0)

# Save per-doc results for t-test later
results_df = pd.DataFrame({'doc_id': list(doc_scores.keys()),
                            'pred': doc_preds, 'true': doc_labels})
results_df['correct'] = (results_df['pred'] == results_df['true']).astype(int)
results_df.to_csv(os.path.join(RESULTS, 'flat_bert_doc_results.csv'), index=False)
print(f"\nDoc-level Macro-F1: {macro_f1_test:.4f}")
print(f"Saved per-doc results to results/flat_bert_doc_results.csv")

# ── Training curves ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses,   label='Val Loss')
ax1.set_title('Flat BERT — Loss Curves')
ax1.legend()
ax2.plot(val_f1s, label='Val Macro-F1 (chunk)', color='green')
ax2.set_title('Flat BERT — Validation Macro-F1')
ax2.legend()
plt.savefig(os.path.join(RESULTS, 'training_curves_flat_bert.png'), dpi=150, bbox_inches='tight')
print("Saved training curves to results/training_curves_flat_bert.png")

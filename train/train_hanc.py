import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.han_c import HANC
from models.focal_loss import FocalLoss
from models.dataset import IDBSectionDataset

# ── Config (can be overridden by run_ablation.py via env vars) ──
TASK          = 'risk'
MODEL_NAME    = 'distilbert-base-uncased'
USE_ATTENTION = bool(int(os.environ.get('ABLATION_USE_ATTENTION', '1')))
USE_FOCAL     = bool(int(os.environ.get('ABLATION_USE_FOCAL',     '1')))
ABLATION_NAME = os.environ.get('ABLATION_NAME', 'han_c_full')
LR            = 2e-5
EPOCHS        = 13
FREEZE_EPOCHS = 3
PATIENCE      = 4
DEVICE        = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
_weight_name  = {'han_c_full': 'hanc_risk.pt',
                 'han_no_attn': 'han_no_attn_risk.pt',
                 'han_cross_entropy': 'han_cross_entropy_risk.pt'}.get(ABLATION_NAME, f'{ABLATION_NAME}.pt')
SAVE_PATH     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'models', 'weights', _weight_name)
RESULTS       = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

print(f"Device: {DEVICE}")
print(f"Task: {TASK} | Attention: {USE_ATTENTION} | Focal: {USE_FOCAL}")

# ── Data ──
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'chunks_risk_only_slim_with_bbox_v1.csv')
df = pd.read_csv(data_path)
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

train_df = df[df['split'] == 'train']
val_df   = df[df['split'] == 'val']
test_df  = df[df['split'] == 'test']

print(f"Train docs: {train_df['document_id'].nunique()} | "
      f"Val docs: {val_df['document_id'].nunique()} | "
      f"Test docs: {test_df['document_id'].nunique()}")

train_dataset = IDBSectionDataset(train_df, tokenizer)
val_dataset   = IDBSectionDataset(val_df,   tokenizer)
test_dataset  = IDBSectionDataset(test_df,  tokenizer)

print(f"Train sections: {len(train_dataset)} | Val sections: {len(val_dataset)} | Test sections: {len(test_dataset)}")
for name, ds in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
    labels = [ds[i]['label'].item() for i in range(len(ds))]
    print(f"  {name} label dist: 0={labels.count(0)} no_risk, 1={labels.count(1)} risk")

def identity_collate(batch):
    return batch[0]

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,  collate_fn=identity_collate, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False, collate_fn=identity_collate, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False, collate_fn=identity_collate, num_workers=0)

# ── Model ──
model = HANC(bert_model_name=MODEL_NAME, use_attention=USE_ATTENTION).to(DEVICE)
loss_fn = FocalLoss(alpha=0.25, gamma=2.0) if USE_FOCAL else nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training Loop ──
train_losses, val_losses, val_f1s = [], [], []
best_val_f1 = 0
patience_counter = 0

for epoch in range(EPOCHS):
    if epoch < FREEZE_EPOCHS:
        for param in model.bert.parameters():
            param.requires_grad = False
    else:
        for param in model.bert.parameters():
            param.requires_grad = True
    
    # Train
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        label     = batch['label'].unsqueeze(0).to(DEVICE)
        
        optimizer.zero_grad()
        logits, _ = model(input_ids, attn_mask)
        loss = loss_fn(logits, label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validate
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attn_mask = batch['attention_mask'].to(DEVICE)
            label     = batch['label'].unsqueeze(0).to(DEVICE)
            
            logits, _ = model(input_ids, attn_mask)
            loss = loss_fn(logits, label)
            val_loss += loss.item()
            
            pred = logits.argmax(dim=-1).item()
            all_preds.append(pred)
            all_labels.append(batch['label'].item())
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_f1s.append(macro_f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")
    
    if macro_f1 > best_val_f1:
        best_val_f1 = macro_f1
        patience_counter = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Saved best model (F1={macro_f1:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            break

# ── Test Evaluation ──
print("\n=== Test Set Evaluation ===")
model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attn_mask = batch['attention_mask'].to(DEVICE)
        logits, _ = model(input_ids, attn_mask)
        pred = logits.argmax(dim=-1).item()
        all_preds.append(pred)
        all_labels.append(batch['label'].item())

label_names = ['Compliant', 'Non-Compliant'] if TASK == 'compliance' else ['no_risk', 'risk']
print(classification_report(all_labels, all_preds, target_names=label_names, labels=[0, 1], zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))

# Save per-section results for evaluation (results_table.py reads these)
doc_ids = [test_dataset[i]['doc_id'] for i in range(len(test_dataset))]
sec_ids = [test_dataset[i]['section_id'] for i in range(len(test_dataset))]
results_df = pd.DataFrame({'doc_id': doc_ids, 'section_id': sec_ids, 'pred': all_preds, 'true': all_labels})
results_df['correct'] = (results_df['pred'] == results_df['true']).astype(int)
results_df.to_csv(os.path.join(RESULTS, f'{ABLATION_NAME}_doc_results.csv'), index=False)
print(f"Saved per-section results to results/{ABLATION_NAME}_doc_results.csv")

# ── Save training curves ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(val_losses, label='Val Loss')
ax1.set_title(f'{ABLATION_NAME} — Loss Curves')
ax1.legend()
ax2.plot(val_f1s, label='Val Macro-F1', color='green')
ax2.set_title(f'{ABLATION_NAME} — Validation Macro-F1')
ax2.legend()
plt.savefig(os.path.join(RESULTS, f'training_curves_{ABLATION_NAME}.png'), dpi=150, bbox_inches='tight')
print(f"Saved training curves to results/training_curves_{ABLATION_NAME}.png")
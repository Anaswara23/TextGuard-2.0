import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import DistilBertTokenizer
from models.han_c import HANC
from models.dataset import IDBDocumentDataset

model = HANC()
weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models', 'weights', 'hanc_compliance.pt')
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'data', 'chunks_with_splits.csv')
df = pd.read_csv(data_path)
test_df = df[df['split'] == 'test']
dataset = IDBDocumentDataset(test_df, tokenizer, task='compliance')

results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

def plot_attention_heatmap(batch, attn_weights, prediction, true_label, doc_id):
    weights = attn_weights.squeeze().detach().numpy()
    if weights.ndim == 0:
        weights = np.array([weights])
    chunks = batch['chunks']
    labels = [f"Chunk {i}: {c[:60]}..." for i, c in enumerate(chunks)]
    
    fig, ax = plt.subplots(figsize=(10, max(4, len(chunks) * 0.4)))
    colors = ['red' if w > np.percentile(weights, 75) else
              'orange' if w > np.percentile(weights, 50) else 'lightblue'
              for w in weights]
    ax.barh(range(len(chunks)), weights, color=colors)
    ax.set_yticks(range(len(chunks)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('Attention Weight')
    ax.set_title(f'{doc_id}\nPredicted: {prediction} | True: {true_label}')
    plt.tight_layout()
    save_path = os.path.join(results_path, f'heatmap_{doc_id}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap for {doc_id}")

label_names = ['Compliant', 'Non-Compliant']
for i, batch in enumerate(dataset):
    if i >= 5:
        break
    with torch.no_grad():
        logits, attn_weights = model(batch['input_ids'], batch['attention_mask'])
    pred = label_names[logits.argmax().item()]
    true = label_names[batch['label'].item()]
    plot_attention_heatmap(batch, attn_weights, pred, true, batch['doc_id'])
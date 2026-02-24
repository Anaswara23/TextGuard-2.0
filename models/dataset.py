import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import pandas as pd

class IDBDocumentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128, task='compliance'):
        self.docs = []
        
        if task == 'compliance':
            label_col = 'compliance_status'
            label_map = {'Compliant': 0, 'Non-Compliant': 1}
        else:
            label_col = 'risk_label'
            label_map = {'no_risk': 0, 'risk': 1}
        
        for doc_id, group in df.groupby('document_id'):
            group = group.sort_values(['section_id', 'chunk_index_in_section'])
            chunks = group['chunk_text'].fillna('').tolist()
            
            labels = group[label_col].map(label_map).dropna()
            # Doc is positive if ANY chunk is positive (any-NC / any-risk)
            doc_label = int(labels.max())
            
            chunk_labels = group[label_col].map(label_map).fillna(0).tolist()
            
            encodings = tokenizer(
                chunks,
                padding='max_length',
                truncation=True,
                max_length=max_len,
                return_tensors='pt'
            )
            
            self.docs.append({
                'input_ids': encodings['input_ids'],
                'attention_mask': encodings['attention_mask'],
                'label': torch.tensor(doc_label, dtype=torch.long),
                'chunk_labels': torch.tensor(chunk_labels, dtype=torch.long),
                'doc_id': doc_id,
                'chunks': chunks,
                'section_titles': group['section_title'].tolist()
            })
    
    def __len__(self):
        return len(self.docs)
    
    def __getitem__(self, idx):
        return self.docs[idx]
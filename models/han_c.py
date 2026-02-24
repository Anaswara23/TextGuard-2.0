import torch
import torch.nn as nn
from transformers import DistilBertModel, BertModel

class HANC(nn.Module):
    def __init__(self, 
                 bert_model_name='distilbert-base-uncased',
                 hidden_dim=256, 
                 num_classes=2, 
                 dropout=0.3,
                 use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        if 'distilbert' in bert_model_name:
            self.bert = DistilBertModel.from_pretrained(bert_model_name)
        else:
            self.bert = BertModel.from_pretrained(bert_model_name)
        
        bert_dim = 768
        
        self.attention = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_dim, num_classes)
        )
    
    def encode_chunks(self, input_ids, attention_masks, bert_batch_size=32):
        # Process all chunks in batches instead of one at a time
        chunk_vecs = []
        n = input_ids.size(0)
        for start in range(0, n, bert_batch_size):
            end = min(start + bert_batch_size, n)
            output = self.bert(
                input_ids[start:end],
                attention_mask=attention_masks[start:end]
            )
            if hasattr(output, 'pooler_output'):
                vecs = output.pooler_output
            else:
                vecs = output.last_hidden_state[:, 0, :]
            chunk_vecs.append(vecs)
        return torch.cat(chunk_vecs, dim=0)
    
    def forward(self, input_ids, attention_masks):
        chunk_vecs = self.encode_chunks(input_ids, attention_masks)
        n_chunks = chunk_vecs.size(0)
        
        if self.use_attention:
            attn_scores = self.attention(chunk_vecs)
            attn_weights = torch.softmax(attn_scores, dim=0)
        else:
            attn_weights = torch.ones(n_chunks, 1, device=chunk_vecs.device) / n_chunks
        
        doc_vec = (attn_weights * chunk_vecs).sum(dim=0, keepdim=True)
        logits = self.classifier(doc_vec)
        return logits, attn_weights
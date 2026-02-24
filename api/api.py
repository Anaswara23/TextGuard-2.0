from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import fitz
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.han_c import HANC
from transformers import DistilBertTokenizer
from policy_labels import (
    POLICY_LABELS_BY_SECTOR, SECTOR_DESCRIPTIONS,
    SECTOR_DISPLAY, label_to_phrase, label_to_display
)

app = FastAPI(title="TextGuard 2.0 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

weights_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models', 'weights', 'hanc_compliance.pt')

_model = None

def get_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(weights_path):
        raise RuntimeError(
            f"Model weights not found at {weights_path}. "
            "Run 'python3 train/train_hanc.py' first."
        )
    m = HANC()
    m.load_state_dict(torch.load(weights_path, map_location='cpu'))
    m.eval()
    _model = m
    return _model

# ── Embedding helpers ────────────────────────────────────────────────────────
_emb_cache: dict = {}   # 'sectors' | 'labels:{SECTOR}' → Tensor


def _get_bert_embeddings(texts: list) -> torch.Tensor:
    """Mean-pool DistilBERT CLS token for a list of texts. Returns (N, 768)."""
    bert = get_model().bert
    batch_size = 32
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            padding=True, truncation=True,
            max_length=64, return_tensors='pt'
        )
        with torch.no_grad():
            out = bert(enc['input_ids'], attention_mask=enc['attention_mask'])
        # Mean-pool over non-padding tokens
        mask = enc['attention_mask'].unsqueeze(-1).float()
        vecs = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
        all_vecs.append(vecs)
    return torch.cat(all_vecs, dim=0)  # (N, 768)


def _sector_embeddings() -> tuple:
    """Returns (sector_names, sector_embs) — cached after first call."""
    if 'sectors' not in _emb_cache:
        names = list(SECTOR_DESCRIPTIONS.keys())
        descs = [SECTOR_DESCRIPTIONS[n] for n in names]
        _emb_cache['sectors'] = (names, _get_bert_embeddings(descs))
    return _emb_cache['sectors']


def _label_embeddings(sector: str) -> tuple:
    """Returns (label_keys, label_embs) for a sector — cached after first call."""
    key = f'labels:{sector}'
    if key not in _emb_cache:
        labels = POLICY_LABELS_BY_SECTOR.get(sector, [])
        phrases = [label_to_phrase(l) for l in labels]
        _emb_cache[key] = (labels, _get_bert_embeddings(phrases))
    return _emb_cache[key]


def detect_sector(doc_text: str) -> str:
    """Find the most semantically similar sector for the document text."""
    snippet = doc_text[:1000]          # first 1000 chars is enough
    doc_emb = _get_bert_embeddings([snippet])   # (1, 768)
    names, sector_embs = _sector_embeddings()   # (17, 768)
    sims = F.cosine_similarity(doc_emb, sector_embs)  # (17,)
    best = int(sims.argmax().item())
    return names[best]


def match_policy_labels(text: str, sector: str, top_k: int = 2) -> list:
    """Return the top_k most relevant policy label keys for a section text."""
    labels, label_embs = _label_embeddings(sector)
    if not labels:
        return []
    text_emb = _get_bert_embeddings([text[:512]])   # (1, 768)
    sims = F.cosine_similarity(text_emb, label_embs)  # (N,)
    k = min(top_k, len(labels))
    top_idxs = sims.topk(k).indices.tolist()
    return [labels[i] for i in top_idxs]


# ── PDF parsing ───────────────────────────────────────────────────────────────
def extract_sections(pdf_bytes: bytes) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    doc.close()
    
    pattern = r'(?=(?:^|\n)\s*(?:I{1,3}|IV|V|VI|VII|VIII|IX|X)\.?\s+[A-Z][^\n]{5,})'
    raw_sections = re.split(pattern, full_text, flags=re.MULTILINE)
    
    sections = []
    for s in raw_sections:
        s = s.strip()
        if len(s) < 50:
            continue
        lines = s.split('\n')
        title = lines[0].strip()
        body = '\n'.join(lines[1:]).strip()
        if len(body) > 30:
            sections.append({'title': title, 'text': body})
    
    if len(sections) < 2:
        paragraphs = [p.strip() for p in full_text.split('\n\n') if len(p.strip()) > 100]
        for i, p in enumerate(paragraphs[:10]):
            sections.append({'title': f'Section {i+1}', 'text': p})
    
    return sections

def analyze_section(section: dict, sector: str) -> dict:
    chunks = [p.strip() for p in section['text'].split('\n\n') if len(p.strip()) > 30]
    if not chunks:
        chunks = [section['text'][:500]]
    
    encodings = tokenizer(
        chunks,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        logits, attn_weights = get_model()(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        probs = torch.softmax(logits, dim=-1)
        non_compliant_prob = float(probs[0][1])
    
    weights = attn_weights.squeeze().tolist()
    if isinstance(weights, float):
        weights = [weights]
    top_idx = int(torch.tensor(weights).argmax().item())
    evidence = chunks[top_idx] if chunks else ""

    # Match policy labels only for flagged sections (saves time on compliant ones)
    policy_labels = []
    if non_compliant_prob > 0.4:
        matched = match_policy_labels(section['text'], sector, top_k=2)
        policy_labels = [label_to_display(l) for l in matched]
    
    return {
        'section': section['title'],
        'risk_score': round(non_compliant_prob * 100),
        'compliance': 'Non-Compliant' if non_compliant_prob > 0.5 else 'Compliant',
        'evidence': evidence,
        'attention_weights': weights,
        'policy_labels': policy_labels,
        'n_chunks': len(chunks)
    }

@app.post("/analyze")
async def analyze_document(file: UploadFile, sector: str = Form(None)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    
    # Ensure model is loaded (raises 503 if weights are missing)
    try:
        get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    
    pdf_bytes = await file.read()
    if len(pdf_bytes) > 50_000_000:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    sections = extract_sections(pdf_bytes)
    if not sections:
        raise HTTPException(status_code=422, detail="Could not parse PDF sections")

    # Use caller-supplied sector if valid, otherwise auto-detect
    valid_sectors = set(POLICY_LABELS_BY_SECTOR.keys())
    sector_source = "selected"
    if sector and sector.upper() in valid_sectors:
        sector = sector.upper()
    else:
        doc_preview = ' '.join(s['text'] for s in sections)[:1500]
        sector = detect_sector(doc_preview)
        sector_source = "detected"

    section_results = [analyze_section(s, sector) for s in sections]
    
    total_chunks = sum(r['n_chunks'] for r in section_results)
    doc_score = sum(r['risk_score'] * r['n_chunks'] for r in section_results) / total_chunks
    
    flagged = [r for r in section_results if r['risk_score'] > 40]
    # Collect unique policy labels across all flagged sections
    all_policy_labels = list(dict.fromkeys(
        lbl for r in flagged for lbl in r.get('policy_labels', [])
    ))

    return {
        'filename': file.filename,
        'overall_risk': round(doc_score),
        'overall_compliance': 'Non-Compliant' if doc_score > 50 else 'Compliant',
        'sections_analyzed': len(section_results),
        'sections_flagged': len(flagged),
        'detected_sector': sector,
        'sector_source': sector_source,
        'sector_display': SECTOR_DISPLAY.get(sector, sector.replace('_', ' ').title()),
        'policy_labels': all_policy_labels,
        'sections': section_results
    }

@app.get("/health")
async def health():
    weights_ready = os.path.exists(weights_path)
    return {
        "status":        "ok",
        "model":         "HAN-C distilbert",
        "weights_ready": weights_ready,
        "weights_path":  weights_path,
    }
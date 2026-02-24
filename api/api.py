from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import fitz
import re
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.han_c import HANC
from transformers import DistilBertTokenizer

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

def analyze_section(section: dict) -> dict:
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
    
    text_lower = section['text'].lower()
    detected_policies = []
    keyword_map = {
        'Environmental': ['environment', 'climate', 'biodiversity', 'pollution'],
        'Social': ['gender', 'indigenous', 'community', 'social', 'vulnerable'],
        'Fiduciary': ['procurement', 'financial', 'audit', 'budget'],
        'Operational': ['implementation', 'supervision', 'monitoring'],
        'Food Security': ['food', 'agriculture', 'crop', 'nutrition'],
    }
    for category, keywords in keyword_map.items():
        if any(kw in text_lower for kw in keywords):
            detected_policies.append(category)
    
    return {
        'section': section['title'],
        'risk_score': round(non_compliant_prob * 100),
        'compliance': 'Non-Compliant' if non_compliant_prob > 0.5 else 'Compliant',
        'evidence': evidence,
        'attention_weights': weights,
        'policy_categories': detected_policies[:3] if detected_policies else ['General'],
        'n_chunks': len(chunks)
    }

@app.post("/analyze")
async def analyze_document(file: UploadFile):
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
    
    section_results = [analyze_section(s) for s in sections]
    
    total_chunks = sum(r['n_chunks'] for r in section_results)
    doc_score = sum(r['risk_score'] * r['n_chunks'] for r in section_results) / total_chunks
    
    flagged = [r for r in section_results if r['risk_score'] > 40]
    all_policies = list(set(p for r in section_results for p in r['policy_categories']))
    
    return {
        'filename': file.filename,
        'overall_risk': round(doc_score),
        'overall_compliance': 'Non-Compliant' if doc_score > 50 else 'Compliant',
        'sections_analyzed': len(section_results),
        'sections_flagged': len(flagged),
        'policy_categories': all_policies,
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
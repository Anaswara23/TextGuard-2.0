# TextGuard 2.0

**Automated risk and compliance auditor for IDB project documents.**

TextGuard 2.0 uses Hierarchical Attention Networks (HAN-C) built on DistilBERT to analyse PDF project documents section-by-section, producing risk scores and surfacing the exact evidence driving each score.

---

## Features

- **Section-level risk scoring** — each section of a PDF is scored independently, not just the whole document
- **Hierarchical Attention Network (HAN-C)** — DistilBERT encodes text chunks; a learned attention layer weights them to produce a document vector
- **Evidence extraction** — highlights the sentences that most influenced the risk prediction
- **Sector-aware analysis** — policy label sets tailored to sector (energy, transport, environment, etc.) or auto-detected
- **History panel** — previous analyses stored in session state for quick comparison
- **Dark / light mode** — full theme toggle with explicit colour palette

---

## Architecture

```
┌─────────────────────┐        HTTP        ┌──────────────────────┐
│   Streamlit UI      │  ──────────────►  │   FastAPI backend    │
│   app/app.py        │  ◄──────────────  │   api/api.py         │
│   port 8501         │    JSON results    │   port 8000          │
└─────────────────────┘                   └──────────────────────┘
                                                     │
                                          ┌──────────▼───────────┐
                                          │  HAN-C model         │
                                          │  models/han_c.py     │
                                          │  + DistilBERT        │
                                          └──────────────────────┘
```

The Streamlit frontend sends the uploaded PDF to the FastAPI backend, which:
1. Extracts and segments text using PyMuPDF
2. Tokenises chunks with DistilBERT
3. Runs the HAN-C model with temperature-scaled softmax (T = 2.0)
4. Returns per-section risk scores, labels, and evidence spans

---

## Project Structure

```
TextGuard-2.0/
├── app/
│   └── app.py              # Streamlit frontend (two-page: home + results)
├── api/
│   └── api.py              # FastAPI backend
├── models/
│   ├── han_c.py            # HAN-C model definition
│   ├── dataset.py          # Dataset loader
│   ├── focal_loss.py       # Focal loss for class imbalance
│   └── weights/            # Trained .pt files (not in repo — see below)
├── train/
│   ├── train_hanc.py       # Training script
│   └── generate_heatmaps.py
├── data/
│   └── create_splits.py    # Data preparation
├── requirements.txt
├── render.yaml             # Render.com deployment config
└── .streamlit/
    └── config.toml
```

---

## Model

| Component | Detail |
|---|---|
| Base encoder | `distilbert-base-uncased` |
| Architecture | Hierarchical Attention Network with Cross-attention (HAN-C) |
| Attention | Single-layer attention over DistilBERT chunk vectors |
| Classifier | Dropout(0.3) → Linear(768, 2) |
| Output | Binary: `risk` / `no_risk` |
| Temperature scaling | T = 2.0 (corrects bias from imbalanced training data) |

Model weights are excluded from this repository due to size (~256 MB each). Upload them to Hugging Face Hub or mount them via Render's persistent disk.

---

## Running Locally

### Prerequisites

- Python 3.9+
- Model weights at `models/weights/hanc_risk.pt`

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start the API

```bash
cd api
uvicorn api:app --reload --port 8000
```

### Start the UI

```bash
# from project root
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501).

---

## Deployment

### Backend — Render.com

A `render.yaml` is included. Create a new Web Service on [render.com](https://render.com), connect this repo, and add a 1 GB persistent disk mounted at `/opt/render/project/src/models/weights`. Upload the `.pt` weight files to the disk via the Render shell.

### Frontend — Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and connect this repo
2. Set **Main file path** to `app/app.py`
3. Under **Advanced settings → Secrets**, add:

```toml
API_URL = "https://your-render-service.onrender.com"
```

The app reads `API_URL` from environment / Streamlit secrets at runtime.

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

| Package | Version |
|---|---|
| `torch` | 2.3.0 |
| `transformers` | 4.40.0 |
| `streamlit` | 1.50.0 |
| `fastapi` | 0.128.8 |
| `PyMuPDF` | 1.26.5 |
| `numpy` | 1.26.4 |

---

## License

For academic and research use.

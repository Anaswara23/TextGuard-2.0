import streamlit as st
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime
import fitz  # PyMuPDF

st.set_page_config(
    page_title="TextGuard 2.0",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
for _key, _default in [
    ("history",        []),
    ("page",           "home"),   # "home" | "results"
    ("active_result",  None),
    ("active_filename", None),
    ("dark_mode",      False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Theme palette (always explicit — no reliance on Streamlit system theme) ────
if st.session_state.dark_mode:
    _bg, _bg2, _border = "#0d1117", "#161b22", "#30363d"
    _text, _muted, _accent = "#e6edf3", "#8b949e", "#4493f8"
    _card_shadow = "rgba(0,0,0,0.4)"
else:
    _bg, _bg2, _border = "#ffffff", "#f6f8fa", "#d0d7de"
    _text, _muted, _accent = "#1f2328", "#656d76", "#0969da"
    _card_shadow = "rgba(0,0,0,0.08)"

st.markdown(f"""
<style>
/* ── Reset & base ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {{
    background-color: {_bg} !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {_bg2} !important;
}}
p, li, div, span, h1, h2, h3, h4, h5, label, strong, b,
.stMarkdown, .stText {{
    color: {_text} !important;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {{
    color: {_text} !important;
}}
[data-testid="stMetricDelta"] {{
    color: {_muted} !important;
}}
[data-testid="stMetricDelta"][data-direction="up"],
[data-testid="stMetricDelta"][data-direction="down"],
[data-testid="stMetricDelta"] > div {{
    color: {_muted} !important;
}}
hr {{ border-color: {_border} !important; }}

/* ── Inputs ── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stFileUploader"] {{
    background-color: {_bg2} !important;
    border-color: {_border} !important;
    color: {_text} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="input"] input {{ color: {_text} !important; }}

/* ── File uploader dropzone ── */
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"] > div,
[data-testid="stFileUploadDropzone"] section,
[data-testid="stFileUploadDropzone"] button,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div,
[data-testid="stFileUploaderDropzoneInstructions"],
[data-testid="stFileUploaderDropzoneInstructions"] > div {{
    background-color: {_bg2} !important;
    color: {_text} !important;
    border-color: {_border} !important;
}}
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] p {{
    color: {_muted} !important;
}}
[data-testid="stFileUploadDropzone"] svg,
[data-testid="stFileUploaderDropzoneInstructions"] svg {{
    fill: {_muted} !important;
    stroke: {_muted} !important;
}}
/* browse files button inside uploader */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button span {{
    background-color: {_bg} !important;
    color: {_text} !important;
    border-color: {_border} !important;
}}

/* ── Selectbox dropdown popup ── */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[data-baseweb="menu"] > ul,
[data-baseweb="menu"] ul,
[data-baseweb="option"],
[data-baseweb="select"] [role="listbox"],
[role="listbox"],
[role="option"],
ul[data-baseweb="menu"] {{
    background-color: {_bg} !important;
    color: {_text} !important;
    border-color: {_border} !important;
}}
[data-baseweb="option"]:hover,
[role="option"]:hover {{
    background-color: {_bg2} !important;
}}
[data-baseweb="option"] span,
[data-baseweb="option"] div,
[role="option"] span,
[role="option"] div,
[role="option"] {{
    color: {_text} !important;
    background-color: transparent !important;
}}
[aria-selected="true"][role="option"] {{
    background-color: {_bg2} !important;
}}
[data-testid="stSelectbox"] [data-baseweb="select"] svg {{
    fill: {_text} !important;
}}

/* ── Dropdown portal (BaseWeb appends to body, outside .stApp) ── */
body > div[data-baseweb="layer"] {{
    background: transparent !important;
}}
body > div[data-baseweb="layer"] > div,
body > div[data-baseweb="layer"] [data-baseweb="popover"],
body > div[data-baseweb="layer"] [data-baseweb="menu"],
body > div[data-baseweb="layer"] ul,
body > div[data-baseweb="layer"] [role="listbox"] {{
    background: {_bg} !important;
    background-color: {_bg} !important;
    border-color: {_border} !important;
}}
body > div[data-baseweb="layer"] li,
body > div[data-baseweb="layer"] [role="option"] {{
    background: {_bg} !important;
    background-color: {_bg} !important;
    color: {_text} !important;
}}
body > div[data-baseweb="layer"] [role="option"]:hover,
body > div[data-baseweb="layer"] li:hover {{
    background: {_bg2} !important;
    background-color: {_bg2} !important;
}}
body > div[data-baseweb="layer"] span,
body > div[data-baseweb="layer"] p {{
    color: {_text} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    color: {_text} !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
}}
.stButton > button:hover {{
    border-color: {_accent} !important;
    color: {_accent} !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    background-color: {_bg} !important;
    border-bottom: 1px solid {_border} !important;
}}
[data-testid="stTabs"] [data-baseweb="tab"] {{
    color: {_muted} !important;
    background-color: transparent !important;
}}
[data-testid="stTabs"] [aria-selected="true"] {{
    color: {_text} !important;
    border-bottom: 2px solid {_accent} !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
    padding: 0.85rem 1rem !important;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 1.05rem !important;
    white-space: normal !important;
    word-break: break-word !important;
    overflow: visible !important;
    text-overflow: unset !important;
}}

/* ── Download button ── */
[data-testid="stDownloadButton"] button {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    color: {_text} !important;
    border-radius: 6px !important;
    font-size: 0.875rem !important;
}}
[data-testid="stDownloadButton"] button:hover {{
    border-color: {_accent} !important;
    color: {_accent} !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] > details > summary,
details > summary {{
    background-color: {_bg2} !important;
    color: {_text} !important;
    fill: {_text} !important;
}}
details summary p,
details summary span,
details summary svg,
[data-testid="stExpander"] summary p {{
    color: {_text} !important;
    fill: {_text} !important;
    background-color: transparent !important;
}}

/* ── Force full width inside expanders ── */
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {{
    align-items: stretch !important;
}}
[data-testid="stExpander"] [data-testid="element-container"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] > div {{
    width: 100% !important;
    max-width: 100% !important;
    align-self: stretch !important;
    box-sizing: border-box !important;
    min-width: 0 !important;
}}

/* ── Evidence blockquote styling ── */
[data-testid="stExpander"] blockquote {{
    border-left: 3px solid {_accent} !important;
    background: {_bg} !important;
    padding: 0.75rem 1rem !important;
    margin: 0.5rem 0 !important;
    border-radius: 3px !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}}
[data-testid="stExpander"] blockquote p,
[data-testid="stExpander"] blockquote em {{
    color: {_text} !important;
}}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {{
    background-color: {_accent} !important;
}}

/* ── Reduce Streamlit default top padding ── */
.block-container, [data-testid="stMainBlockContainer"] {{
    padding-top: 1rem !important;
}}

/* ── Page 1: hero ── */
.hero {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 2rem 1rem 2.5rem;
}}
.hero-wordmark {{
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {_accent};
    margin-bottom: 0.75rem;
}}
.hero-title {{
    font-size: 3.4rem;
    font-weight: 700;
    color: {_text};
    letter-spacing: -0.8px;
    line-height: 1.2;
    margin-bottom: 1rem;
}}
.hero-sub {{
    font-size: 1rem;
    color: {_muted};
    line-height: 1.65;
    max-width: 520px;
    margin-bottom: 2.5rem;
}}

/* ── Upload card on page 1 ── */
.upload-card {{
    background: {_bg2};
    border: 1px solid {_border};
    border-radius: 12px;
    padding: 2rem 2.5rem;
    max-width: 560px;
    margin: 0 auto 3rem;
    box-shadow: 0 1px 8px {_card_shadow};
}}
.upload-label {{
    font-size: 0.78rem;
    font-weight: 600;
    color: {_muted};
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.4rem;
}}

/* ── History section ── */
.history-title {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: {_muted};
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {_border};
}}

/* ── History cards ── */
.doc-card {{
    background: {_bg2};
    border: 1px solid {_border};
    border-radius: 8px;
    padding: 1rem;
    transition: box-shadow 0.15s, border-color 0.15s;
    cursor: pointer;
}}
.doc-card:hover {{
    box-shadow: 0 2px 14px {_card_shadow};
    border-color: {_accent};
}}
.doc-card-name {{
    font-size: 0.82rem;
    font-weight: 600;
    color: {_text};
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 0.2rem;
}}
.doc-card-date {{
    font-size: 0.7rem;
    color: {_muted};
    margin-bottom: 0.65rem;
}}
.doc-card-stats {{
    display: flex;
    gap: 1.2rem;
    margin-top: 0.4rem;
}}
.stat {{ font-size: 0.7rem; color: {_muted}; line-height: 1.4; }}
.stat-val {{ font-weight: 700; font-size: 0.95rem; display: block; color: {_text}; }}
.stat-risk {{ color: #cf222e !important; }}
.stat-ok   {{ color: #1a7f37 !important; }}
.stat-score {{ color: #9a6700 !important; }}

/* ── Risk badges ── */
.badge {{
    display: inline-block;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.69rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}
.badge-high   {{ background: #ffebe9 !important; color: #cf222e !important; }}
.badge-medium {{ background: #fff8c5 !important; color: #9a6700 !important; }}
.badge-low    {{ background: #dafbe1 !important; color: #1a7f37 !important; }}
.badge-high *,
.badge-medium *,
.badge-low * {{
    color: inherit !important;
}}

/* ── Results page ── */
.results-back {{
    display: inline-block;
    font-size: 0.85rem;
    color: {_muted};
    cursor: pointer;
    margin-bottom: 1.5rem;
}}
.results-title {{
    font-size: 1.4rem;
    font-weight: 700;
    color: {_text};
    margin-bottom: 0.25rem;
    word-break: break-word;
}}

/* ── Evidence box ── */
.evidence-box {{
    background: {_bg};
    border-left: 3px solid {_accent};
    border-radius: 3px;
    padding: 0.75rem 1rem;
    font-style: italic;
    font-size: 0.88rem;
    color: {_text};
    margin-top: 0.5rem;
    line-height: 1.7;
    white-space: normal;
    word-break: break-word;
    overflow-wrap: break-word;
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box;
    display: block;
}}
.evidence-box p {{
    margin: 0 0 0.6rem 0;
    color: {_text};
    width: 100%;
    max-width: 100%;
}}
/* ensure the Streamlit markdown wrapper doesn't constrain width */
[data-testid="stMarkdownContainer"]:has(.evidence-box) {{
    width: 100% !important;
    max-width: 100% !important;
}}

/* ── Policy chips ── */
.chip {{
    display: inline-block;
    background: transparent;
    color: {_accent};
    border: 1px solid {_accent};
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.72rem;
    margin: 2px;
}}

/* ── PDF legend ── */
.pdf-legend {{
    display: flex;
    gap: 1.5rem;
    align-items: center;
    font-size: 0.81rem;
    color: {_muted};
    padding: 0.6rem 0.9rem;
    background: {_bg2};
    border: 1px solid {_border};
    border-radius: 6px;
    margin-bottom: 1rem;
}}
.swatch {{
    width: 12px; height: 12px;
    border-radius: 2px;
    display: inline-block;
    margin-right: 5px;
    vertical-align: middle;
}}

/* ── Theme toggle ── */
.stButton[data-testid="baseButton-secondary"] > button {{
    font-size: 0.78rem !important;
}}
</style>
""", unsafe_allow_html=True)

# ── Inject portal CSS directly into page DOM (no iframe) ─────────────────────
# st.html() injects without a sandbox so it affects BaseWeb portal divs
# that are appended to document.body outside the React tree.
st.html(f"""<style>
body > div[data-baseweb="layer"] > div,
body > div[data-baseweb="layer"] [data-baseweb="popover"],
body > div[data-baseweb="layer"] [data-baseweb="menu"],
body > div[data-baseweb="layer"] ul,
body > div[data-baseweb="layer"] [role="listbox"] {{
  background: {_bg} !important;
  background-color: {_bg} !important;
  border-color: {_border} !important;
}}
body > div[data-baseweb="layer"] li,
body > div[data-baseweb="layer"] [role="option"] {{
  background: {_bg} !important;
  background-color: {_bg} !important;
  color: {_text} !important;
}}
body > div[data-baseweb="layer"] [role="option"]:hover,
body > div[data-baseweb="layer"] li:hover {{
  background: {_bg2} !important;
  background-color: {_bg2} !important;
}}
body > div[data-baseweb="layer"] span,
body > div[data-baseweb="layer"] p {{
  color: {_text} !important;
}}

/* ── Expander content full width ── */
[data-testid="stExpander"] [data-testid="stVerticalBlock"],
[data-testid="stExpander"] details > div {{
  align-items: stretch !important;
  width: 100% !important;
}}
[data-testid="stExpander"] [data-testid="element-container"],
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {{
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  align-self: stretch !important;
  box-sizing: border-box !important;
}}
</style>
<script>
(function fixEvidenceWidth() {{
  var doc;
  try {{ doc = window.parent.document; }} catch(e) {{ doc = document; }}
  function applyFix() {{
    doc.querySelectorAll('[data-testid="stExpander"] [data-testid="stMarkdownContainer"]').forEach(function(el) {{
      var node = el;
      for (var i = 0; i < 6; i++) {{
        if (!node) break;
        node.style.setProperty('width', '100%', 'important');
        node.style.setProperty('max-width', '100%', 'important');
        node.style.setProperty('min-width', '0', 'important');
        node.style.setProperty('align-self', 'stretch', 'important');
        node.style.setProperty('box-sizing', 'border-box', 'important');
        if (node.parentElement && node.parentElement.getAttribute('data-testid') === 'stExpander') break;
        node = node.parentElement;
      }}
    }});
    doc.querySelectorAll('[data-testid="stExpander"] [data-testid="stVerticalBlock"]').forEach(function(el) {{
      el.style.setProperty('align-items', 'stretch', 'important');
      el.style.setProperty('width', '100%', 'important');
    }});
  }}
  applyFix();
  try {{ new MutationObserver(applyFix).observe(doc.body, {{subtree: true, childList: true}}); }} catch(e) {{}}
}})();
</script>""")

# ── Config ─────────────────────────────────────────────────────────────────────
try:
    API_URL = st.secrets["app"]["API_URL"]
except Exception:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

SECTOR_OPTIONS = {
    "Auto-detect from document": "",
    "Transport": "TRANSPORT",
    "Agriculture & Rural Development": "AGRICULTURE_AND_RURAL_DEVELOPMENT",
    "Education": "EDUCATION",
    "Energy": "ENERGY",
    "Environment & Natural Disasters": "ENVIRONMENT_AND_NATURAL_DISASTERS",
    "Financial Markets": "FINANCIAL_MARKETS",
    "Health": "HEALTH",
    "Industry": "INDUSTRY",
    "Private Firms & SME Development": "PRIVATE_FIRMS_AND_SME_DEVELOPMENT",
    "Reform & Modernization of the State": "REFORM_MODERNIZATION_OF_THE_STATE",
    "Regional Integration": "REGIONAL_INTEGRATION",
    "Science & Technology": "SCIENCE_AND_TECHNOLOGY",
    "Social Investment": "SOCIAL_INVESTMENT",
    "Sustainable Tourism": "SUSTAINABLE_TOURISM",
    "Trade": "TRADE",
    "Urban Development & Housing": "URBAN_DEVELOPMENT_AND_HOUSING",
    "Water & Sanitation": "WATER_AND_SANITATION",
}


# ── PDF highlight helper ───────────────────────────────────────────────────────
def _clean_evidence(text: str) -> str:
    """Normalize PDF-extracted text, remove non-substantive lines, replace tables."""
    import re
    text = text.strip().strip('"').strip("'")
    # Normalize 3+ newlines to a double newline (paragraph separator)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse multiple spaces/tabs on the same line (but not newlines)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Detect and replace table-like paragraphs
    def _is_table(p: str) -> bool:
        lines = [l for l in p.split('\n') if l.strip()]
        if not lines:
            return False
        # Table indicator: majority of lines have 2+ pipe chars
        pipe_lines = sum(1 for l in lines if l.count('|') >= 2)
        if pipe_lines >= max(2, len(lines) * 0.5):
            return True
        # OR: majority of lines are very short (< 25 chars) — column data rows
        short_lines = sum(1 for l in lines if len(l.strip()) < 25)
        if len(lines) >= 4 and short_lines >= len(lines) * 0.7:
            return True
        return False

    def _keep_para(p: str) -> bool:
        p = p.strip()
        if not p:
            return False
        if re.match(r'^[\dIVXivx]+(\.[\dIVXivx]+)*\.?$', p):
            return False
        if re.match(r'^[a-zA-Z]\)$', p):
            return False
        if len(p) < 40 and not re.search(r'[.!?:,]', p):
            return False
        return True

    paras = text.split('\n\n')
    result_paras = []
    for p in paras:
        if not p.strip():
            continue
        if _is_table(p):
            result_paras.append('[TABLE — check document for full table]')
        elif _keep_para(p):
            # Within paragraph, filter non-substantive lines
            kept = [l for l in p.split('\n') if _keep_para(l)]
            if kept:
                result_paras.append('\n'.join(kept))
    return '\n\n'.join(result_paras).strip()


def render_highlighted_pdf(pdf_bytes: bytes, sections: list) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    highlights = []
    for s in sections:
        if s["risk_score"] > 40:
            evidence = s.get("evidence", "").strip().strip('"').strip("'")
            if not evidence:
                continue
            color = (1.0, 0.22, 0.22) if s["risk_score"] > 70 else (1.0, 0.82, 0.1)
            highlights.append((evidence[:70], color))
    for page in doc:
        for snippet, color in highlights:
            for rect in page.search_for(snippet):
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()
    mat = fitz.Matrix(1.5, 1.5)
    imgs = [p.get_pixmap(matrix=mat, alpha=False).tobytes("png") for p in doc]
    doc.close()
    return imgs


def render_section_pdf(pdf_bytes: bytes, section: dict) -> list:
    """Render only the PDF pages where this section's evidence appears, highlighted."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    evidence = section.get("evidence", "").strip().strip('"').strip("'")
    snippet = evidence[:70] if evidence else ""
    color = (1.0, 0.22, 0.22) if section["risk_score"] > 70 else (1.0, 0.82, 0.1)
    hit_pages = set()
    for i, page in enumerate(doc):
        if snippet and page.search_for(snippet):
            hit_pages.add(i)
    if not hit_pages:  # fallback: return first page
        hit_pages.add(0)
    for i in hit_pages:
        page = doc[i]
        for rect in page.search_for(snippet):
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=color)
            annot.update()
    mat = fitz.Matrix(1.8, 1.8)
    imgs = []
    for i in sorted(hit_pages):
        imgs.append(doc[i].get_pixmap(matrix=mat, alpha=False).tobytes("png"))
    doc.close()
    return imgs


# ── Helpers ────────────────────────────────────────────────────────────────────
def nav_to(page: str):
    st.session_state.page = page
    st.rerun()


def badge_cls(score: int) -> str:
    return "badge-high" if score > 70 else "badge-medium" if score > 40 else "badge-low"


def badge_lbl(score: int) -> str:
    return "High Risk" if score > 70 else "Medium Risk" if score > 40 else "Low Risk"


# ── Global nav bar (theme toggle, always visible) ──────────────────────────────
_left, _right = st.columns([14, 1])
with _right:
    _lbl = "Light" if st.session_state.dark_mode else "Dark"
    if st.button(_lbl, key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":

    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-wordmark">IDB Compliance Tool</div>
        <div class="hero-title">TextGuard 2.0</div>
        <div class="hero-sub">
            Automated risk and compliance auditor for IDB project documents.<br>
            Upload a PDF to receive section-level risk scores powered by
            Hierarchical Attention Networks.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload card — centered via columns
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown('<div class="upload-label">Document</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload PDF",
            type="pdf",
            label_visibility="collapsed",
            key="home_uploader",
        )

        st.markdown('<div class="upload-label" style="margin-top:1rem">Sector</div>',
                    unsafe_allow_html=True)
        sector_choice = st.selectbox(
            "Sector",
            options=list(SECTOR_OPTIONS.keys()),
            label_visibility="collapsed",
            key="home_sector",
        )

        st.markdown("<div style='margin-top:1.25rem'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button(
            "Analyze Document",
            disabled=uploaded is None,
            use_container_width=True,
            type="primary",
            key="home_analyze",
        )

    # Run analysis
    if uploaded and analyze_clicked:
        selected_sector = SECTOR_OPTIONS[sector_choice]
        pdf_bytes = uploaded.getvalue()
        with st.spinner(f"Analyzing {uploaded.name}..."):
            try:
                post_data = {"sector": selected_sector} if selected_sector else {}
                response = requests.post(
                    f"{API_URL}/analyze",
                    files={"file": (uploaded.name, pdf_bytes, "application/pdf")},
                    data=post_data,
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to the TextGuard API. "
                    "Make sure the backend is running on port 8000."
                )
                st.stop()
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.stop()

        # Save to history
        st.session_state.history = [
            h for h in st.session_state.history if h["filename"] != uploaded.name
        ]
        st.session_state.history.insert(0, {
            "filename": uploaded.name,
            "date": datetime.now().strftime("%b %d, %Y  %I:%M %p"),
            "data": data,
            "pdf_bytes": pdf_bytes,
        })
        st.session_state.active_result = data
        st.session_state.active_filename = uploaded.name
        nav_to("results")

    # Previously analyzed documents
    if st.session_state.history:
        _, hist_col, _ = st.columns([1, 4, 1])
        with hist_col:
            st.markdown(
                '<div class="history-title">Previously Analyzed</div>',
                unsafe_allow_html=True,
            )
            COLS = 4
            for row_start in range(0, len(st.session_state.history), COLS):
                row_entries = st.session_state.history[row_start : row_start + COLS]
                cols = st.columns(COLS)
                for col, entry in zip(cols, row_entries):
                    d = entry["data"]
                    risk = d["overall_risk"]
                    n_risk = d["sections_flagged"]
                    n_ok = d["sections_analyzed"] - d["sections_flagged"]
                    name = entry["filename"]
                    short = name[:28] + "..." if len(name) > 28 else name
                    bc = badge_cls(risk)
                    bl = badge_lbl(risk)
                    with col:
                        st.markdown(f"""
                        <div class="doc-card">
                            <div class="doc-card-name" title="{name}">{short}</div>
                            <div class="doc-card-date">{entry['date']}</div>
                            <span class="badge {bc}">{bl}</span>
                            <div class="doc-card-stats">
                                <div class="stat">Flagged<span class="stat-val stat-risk">{n_risk}</span></div>
                                <div class="stat">Clear<span class="stat-val stat-ok">{n_ok}</span></div>
                                <div class="stat">Score<span class="stat-val stat-score">{risk}%</span></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        if st.button(
                            "View results",
                            key=f"hist_{name}",
                            use_container_width=True,
                        ):
                            st.session_state.active_result = d
                            st.session_state.active_filename = name
                            nav_to("results")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "results":

    data = st.session_state.active_result
    filename = st.session_state.active_filename
    active_hist = next(
        (h for h in st.session_state.history if h.get("filename") == filename), None
    )

    if not data:
        nav_to("home")

    # Back button + title row
    back_col, title_col, dl_col = st.columns([1, 6, 2])
    with back_col:
        if st.button("Back", key="back_btn", use_container_width=True):
            nav_to("home")
    with title_col:
        risk = data["overall_risk"]
        sector_display = data.get("sector_display", "")
        st.markdown(f'<div class="results-title">{filename}</div>', unsafe_allow_html=True)
        if sector_display:
            prefix = "Detected" if data.get("sector_source") == "detected" else "Sector"
            st.badge(f"{prefix}: {sector_display}", color="blue")
    with dl_col:
        st.download_button(
            "Download Report",
            data=json.dumps(data, indent=2),
            file_name=f"textguard_{filename.replace('.pdf', '')}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()

    # Tabs
    tab_analysis, tab_pdf = st.tabs(["Analysis", "Highlighted PDF"])

    # ── Analysis tab ──────────────────────────────────────────────────────────
    with tab_analysis:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "Overall Risk Score", f"{risk}%",
            "High Risk" if risk > 70 else "Medium Risk" if risk > 40 else "Low Risk",
        )
        mc2.metric("Compliance Status", data["overall_compliance"])
        mc3.metric(
            "Sections Flagged",
            f"{data['sections_flagged']} / {data['sections_analyzed']}",
        )
        sector_short = (sector_display[:22] + "…") if sector_display and len(sector_display) > 22 else (sector_display or "—")
        mc4.metric("Sector", sector_short)

        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

        col_gauge, col_breakdown = st.columns([1, 2])
        gauge_color = "#cf222e" if risk > 70 else "#9a6700" if risk > 40 else "#1a7f37"

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                title={"text": "Risk Score", "font": {"size": 13, "color": _text}},
                number={"suffix": "%", "font": {"color": _text}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": _muted},
                    "bar": {"color": gauge_color},
                    "bgcolor": "rgba(0,0,0,0)",
                    "bordercolor": _border,
                    "steps": [
                        {"range": [0, 40],   "color": "rgba(26,127,55,0.10)"},
                        {"range": [40, 70],  "color": "rgba(154,103,0,0.10)"},
                        {"range": [70, 100], "color": "rgba(207,34,46,0.10)"},
                    ],
                },
            ))
            fig.update_layout(
                height=240,
                margin=dict(t=40, b=10, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": _text},
            )
            st.plotly_chart(fig, use_container_width=True)

            policy_labels = data.get("policy_labels", [])
            if policy_labels:
                st.markdown("**Policy Risk Labels**")
                chips = " ".join(
                    f'<span class="chip">{lbl}</span>' for lbl in policy_labels
                )
                st.markdown(chips, unsafe_allow_html=True)

        with col_breakdown:
            st.subheader("Section Breakdown")
            for s in sorted(data["sections"], key=lambda x: -x["risk_score"]):
                bc = badge_cls(s["risk_score"])
                bl = badge_lbl(s["risk_score"])
                st.markdown(
                    f"**{s['section']}** &nbsp;&mdash;&nbsp; "
                    f"{s['risk_score']}% &nbsp;|&nbsp; {s['compliance']} &nbsp;"
                    f'<span class="badge {bc}">{bl}</span>',
                    unsafe_allow_html=True,
                )
                st.progress(s["risk_score"] / 100)

        st.divider()

        flagged = [s for s in data["sections"] if s["risk_score"] > 40]
        if flagged:
            st.subheader(f"Flagged Sections ({len(flagged)})")
            for s in sorted(flagged, key=lambda x: -x["risk_score"]):
                with st.expander(
                    f"{s['section']}  —  {s['risk_score']}% risk  |  {s['compliance']}"
                ):
                    st.markdown("**Key Evidence** *(highest-attention chunk)*")
                    cleaned = _clean_evidence(s["evidence"])
                    paras = cleaned.split("\n\n")
                    # Blockquote: escape markdown special chars to prevent garbled rendering
                    def _escape_md(line: str) -> str:
                        import re
                        # escape markdown special chars: * _ ` [ ] ( ) # + - . !
                        return re.sub(r'([\*\_\`\[\]\(\)\#\+\-\.\!])', r'\\\1', line)
                    bq_lines = []
                    for para in paras:
                        if para.strip():
                            for line in para.strip().split("\n"):
                                if line.strip():
                                    bq_lines.append(f"> {_escape_md(line.strip())}")
                            bq_lines.append(">")
                    st.markdown("\n".join(bq_lines))

                    # View in Document button
                    vid_key = f"vid_{s['section']}"
                    if vid_key not in st.session_state:
                        st.session_state[vid_key] = False
                    if st.button(
                        "Hide Document View" if st.session_state[vid_key] else "View in Document",
                        key=f"vidbtn_{s['section']}",
                    ):
                        st.session_state[vid_key] = not st.session_state[vid_key]
                        st.rerun()
                    if st.session_state.get(vid_key):
                        pdf_bytes_local = active_hist.get("pdf_bytes") if active_hist else None
                        if pdf_bytes_local:
                            with st.spinner("Loading pages..."):
                                sec_pages = render_section_pdf(pdf_bytes_local, s)
                            swatch_color = "rgba(207,34,46,0.55)" if s["risk_score"] > 70 else "rgba(255,210,25,0.8)"
                            st.markdown(
                                f'<span style="display:inline-block;width:12px;height:12px;'
                                f'background:{swatch_color};border-radius:2px;margin-right:6px;vertical-align:middle"></span>'
                                f'<span style="font-size:0.8rem;color:{_muted}">Highlighted passage — {len(sec_pages)} page(s) shown</span>',
                                unsafe_allow_html=True,
                            )
                            for pg_img in sec_pages:
                                st.image(pg_img, use_container_width=True)
                        else:
                            st.info("Re-upload the document to enable the document viewer.")

                    sec_labels = s.get("policy_labels", [])
                    if sec_labels:
                        st.markdown("**Policy Risk Labels**")
                        chips = " ".join(
                            f'<span class="chip">{lbl}</span>' for lbl in sec_labels
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                    if len(s["attention_weights"]) > 1:
                        fig_a = go.Figure(go.Bar(
                            x=[f"Chunk {i}" for i in range(len(s["attention_weights"]))],
                            y=s["attention_weights"],
                            marker_color=[
                                gauge_color if w == max(s["attention_weights"]) else _accent
                                for w in s["attention_weights"]
                            ],
                            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
                        ))
                        fig_a.update_layout(
                            title={"text": "Chunk Attention Weights",
                                   "font": {"size": 12, "color": _text}},
                            height=200,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=35, b=15, l=15, r=15),
                            font={"color": _text},
                            xaxis={"gridcolor": _border, "tickcolor": _muted},
                            yaxis={"gridcolor": _border, "tickcolor": _muted},
                        )
                        st.plotly_chart(fig_a, use_container_width=True)
        else:
            st.success("No high-risk sections detected.")

    # ── Highlighted PDF tab ───────────────────────────────────────────────────
    with tab_pdf:
        pdf_bytes = active_hist.get("pdf_bytes") if active_hist else None
        if not pdf_bytes:
            st.info(
                "PDF data is not available for this result. "
                "Re-upload the document to enable the highlighted viewer."
            )
        else:
            flagged_secs = [s for s in data["sections"] if s["risk_score"] > 40]
            if not flagged_secs:
                st.success("No risky sections — nothing to highlight.")
            else:
                st.markdown(f"""
                <div class="pdf-legend">
                    <span>
                        <span class="swatch" style="background:rgba(207,34,46,0.55)"></span>
                        High risk (&gt;70%)
                    </span>
                    <span>
                        <span class="swatch" style="background:rgba(255,210,25,0.8)"></span>
                        Medium risk (40&ndash;70%)
                    </span>
                    <span>
                        Highlighted text is the highest-attention evidence chunk per section.
                    </span>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("Rendering highlighted PDF..."):
                    page_images = render_highlighted_pdf(pdf_bytes, data["sections"])

                st.caption(
                    f"{len(page_images)} page(s)  ·  "
                    f"{len(flagged_secs)} section(s) flagged  ·  "
                    "scroll to review all highlighted passages"
                )
                for i, img_bytes in enumerate(page_images):
                    st.image(img_bytes, use_container_width=True, caption=f"Page {i + 1}")


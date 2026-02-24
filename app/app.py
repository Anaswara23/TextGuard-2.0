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
    ("history", []),
    ("active_result", None),
    ("active_filename", None),
    ("dark_mode", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Explicit theme palette — always injected so light mode is truly white ──────
if st.session_state.dark_mode:
    _bg, _bg2, _border = "#0d1117", "#161b22", "#30363d"
    _text, _muted, _accent = "#e6edf3", "#8b949e", "#4493f8"
else:
    _bg, _bg2, _border = "#ffffff", "#f6f8fa", "#d0d7de"
    _text, _muted, _accent = "#1f2328", "#656d76", "#0969da"

st.markdown(f"""
<style>
/* ── Page base ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {{
    background-color: {_bg} !important;
}}
section[data-testid="stSidebar"] {{
    background-color: {_bg2} !important;
}}

/* ── Typography ── */
p, li, div, span, h1, h2, h3, h4, h5, label,
.stMarkdown, .stText,
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {{
    color: {_text} !important;
}}
hr {{ border-color: {_border} !important; }}

/* ── Inputs / selects ── */
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-testid="stFileUploader"] {{
    background-color: {_bg2} !important;
    border-color: {_border} !important;
    color: {_text} !important;
}}
[data-baseweb="select"] span,
[data-baseweb="input"] input {{
    color: {_text} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    color: {_text} !important;
    border-radius: 6px !important;
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

/* ── Metrics ── */
[data-testid="metric-container"] {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
}}

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
}}
details summary {{ color: {_text} !important; }}

/* ── Hero section ── */
.hero-wrap {{
    text-align: center;
    padding: 2.5rem 0 1.8rem;
    border-bottom: 1px solid {_border};
    margin-bottom: 2rem;
}}
.hero-title {{
    font-size: 1.9rem;
    font-weight: 700;
    color: {_text};
    letter-spacing: -0.5px;
    margin-bottom: 0.4rem;
}}
.hero-sub {{
    font-size: 0.92rem;
    color: {_muted};
    line-height: 1.6;
    max-width: 640px;
    margin: 0 auto;
}}

/* ── Section label ── */
.section-header {{
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {_muted};
    margin-bottom: 0.9rem;
    margin-top: 0.2rem;
}}

/* ── Document cards ── */
.doc-card {{
    background: {_bg2};
    border: 1px solid {_border};
    border-radius: 8px;
    padding: 0.9rem 1rem 0.8rem;
    transition: box-shadow 0.15s;
}}
.doc-card:hover {{ box-shadow: 0 2px 12px rgba(0,0,0,0.10); }}
.doc-card-title {{
    font-size: 0.8rem;
    font-weight: 600;
    color: {_text};
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    margin-bottom: 0.18rem;
}}
.doc-card-date {{
    font-size: 0.7rem;
    color: {_muted};
    margin-bottom: 0.55rem;
}}
.doc-card-stats {{
    display: flex;
    gap: 1.1rem;
    margin-top: 0.5rem;
}}
.stat-item {{ font-size: 0.7rem; color: {_muted}; }}
.stat-val  {{ font-size: 0.95rem; font-weight: 700; color: {_text}; }}
.stat-risk  {{ color: #cf222e !important; }}
.stat-ok    {{ color: #1a7f37 !important; }}
.stat-score {{ color: #9a6700 !important; }}

/* ── Risk badges ── */
.risk-badge-high {{
    background: #ffebe9; color: #cf222e;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.7rem; font-weight: 700;
}}
.risk-badge-medium {{
    background: #fff8c5; color: #9a6700;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.7rem; font-weight: 700;
}}
.risk-badge-low {{
    background: #dafbe1; color: #1a7f37;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.7rem; font-weight: 700;
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
    line-height: 1.55;
}}

/* ── Policy label chips ── */
.label-chip {{
    background: transparent;
    color: {_accent};
    border: 1px solid {_accent};
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.73rem;
    margin: 2px;
    display: inline-block;
}}

/* ── PDF legend bar ── */
.pdf-legend {{
    display: flex;
    gap: 1.4rem;
    align-items: center;
    font-size: 0.82rem;
    color: {_muted};
    margin-bottom: 0.8rem;
    padding: 0.6rem 0.9rem;
    background: {_bg2};
    border: 1px solid {_border};
    border-radius: 6px;
}}
.legend-swatch {{
    width: 12px; height: 12px;
    border-radius: 2px;
    display: inline-block;
    margin-right: 5px;
    vertical-align: middle;
}}
</style>
""", unsafe_allow_html=True)

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
def render_highlighted_pdf(pdf_bytes: bytes, sections: list) -> list:
    """Highlight risky evidence chunks in the PDF. Returns PNG bytes per page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    highlights = []
    for s in sections:
        if s["risk_score"] > 40:
            evidence = s.get("evidence", "").strip().strip('"').strip("'")
            if not evidence:
                continue
            snippet = evidence[:70]
            color = (1.0, 0.22, 0.22) if s["risk_score"] > 70 else (1.0, 0.82, 0.1)
            highlights.append((snippet, color))

    for page in doc:
        for snippet, color in highlights:
            for rect in page.search_for(snippet):
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()

    mat = fitz.Matrix(1.5, 1.5)
    page_images = [page.get_pixmap(matrix=mat, alpha=False).tobytes("png") for page in doc]
    doc.close()
    return page_images


# ── Top bar: theme toggle ──────────────────────────────────────────────────────
_nav_l, _nav_r = st.columns([14, 1])
with _nav_r:
    _label = "Light mode" if st.session_state.dark_mode else "Dark mode"
    if st.button(_label, key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-title">TextGuard 2.0</div>
  <div class="hero-sub">
    Automated risk and compliance auditor for IDB project documents.
    Upload a PDF, select the sector, and receive section-level risk scores
    powered by Hierarchical Attention Networks.
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload panel ───────────────────────────────────────────────────────────────
up_col, sec_col, btn_col = st.columns([3, 2, 1])
with up_col:
    uploaded = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
with sec_col:
    sector_choice = st.selectbox(
        "Sector", options=list(SECTOR_OPTIONS.keys()), label_visibility="collapsed"
    )
with btn_col:
    analyze_clicked = st.button(
        "Analyze", disabled=uploaded is None, use_container_width=True, type="primary"
    )

st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
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
    st.rerun()

# ── Document history grid ──────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown(
        '<div class="section-header">Analyzed Documents</div>', unsafe_allow_html=True
    )
    COLS = 5
    for row_start in range(0, len(st.session_state.history), COLS):
        row_entries = st.session_state.history[row_start : row_start + COLS]
        grid = st.columns(COLS)
        for col, entry in zip(grid, row_entries):
            d = entry["data"]
            risk = d["overall_risk"]
            n_risk = d["sections_flagged"]
            n_ok = d["sections_analyzed"] - d["sections_flagged"]
            badge_cls = (
                "risk-badge-high" if risk > 70
                else "risk-badge-medium" if risk > 40
                else "risk-badge-low"
            )
            badge_lbl = "High Risk" if risk > 70 else "Medium Risk" if risk > 40 else "Low Risk"
            fname_short = (
                entry["filename"][:26] + "..." if len(entry["filename"]) > 26
                else entry["filename"]
            )
            active_style = (
                f"border-color:{_accent};"
                if entry["filename"] == st.session_state.active_filename
                else ""
            )
            with col:
                st.markdown(f"""
                <div class="doc-card" style="{active_style}">
                    <div class="doc-card-title" title="{entry['filename']}">{fname_short}</div>
                    <div class="doc-card-date">{entry['date']}</div>
                    <span class="{badge_cls}">{badge_lbl}</span>
                    <div class="doc-card-stats">
                        <div class="stat-item">Flagged<br>
                            <span class="stat-val stat-risk">{n_risk}</span></div>
                        <div class="stat-item">Clear<br>
                            <span class="stat-val stat-ok">{n_ok}</span></div>
                        <div class="stat-item">Score<br>
                            <span class="stat-val stat-score">{risk}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View", key=f"view_{entry['filename']}", use_container_width=True):
                    st.session_state.active_result = entry["data"]
                    st.session_state.active_filename = entry["filename"]
                    st.rerun()

# ── Result detail ──────────────────────────────────────────────────────────────
data = st.session_state.active_result
filename = st.session_state.active_filename
active_history = next(
    (h for h in st.session_state.history if h.get("filename") == filename), None
)

if data:
    st.divider()
    tab_analysis, tab_pdf = st.tabs(["Analysis", "Highlighted PDF"])

    # ── Tab 1: Analysis ───────────────────────────────────────────────────────
    with tab_analysis:
        risk = data["overall_risk"]
        sector_display = data.get("sector_display", "")

        hcol1, hcol2 = st.columns([5, 1])
        with hcol1:
            st.subheader(filename)
            if sector_display:
                prefix = (
                    "Detected Sector"
                    if data.get("sector_source") == "detected"
                    else "Sector"
                )
                st.badge(f"{prefix}: {sector_display}", color="blue")
        with hcol2:
            st.download_button(
                "Download Report",
                data=json.dumps(data, indent=2),
                file_name=f"textguard_{filename.replace('.pdf', '')}.json",
                mime="application/json",
                use_container_width=True,
            )

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
        mc4.metric("Sector", sector_display or "—")

        st.divider()
        col_gauge, col_breakdown = st.columns([1, 2])

        with col_gauge:
            gauge_color = (
                "#cf222e" if risk > 70 else "#9a6700" if risk > 40 else "#1a7f37"
            )
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
                        {"range": [0, 40],   "color": "rgba(26,127,55,0.1)"},
                        {"range": [40, 70],  "color": "rgba(154,103,0,0.1)"},
                        {"range": [70, 100], "color": "rgba(207,34,46,0.1)"},
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
                tags = " ".join(
                    f'<span class="label-chip">{lbl}</span>' for lbl in policy_labels
                )
                st.markdown(tags, unsafe_allow_html=True)

        with col_breakdown:
            st.subheader("Section Risk Breakdown")
            for s in sorted(data["sections"], key=lambda x: -x["risk_score"]):
                level = (
                    "High" if s["risk_score"] > 70
                    else "Medium" if s["risk_score"] > 40
                    else "Low"
                )
                bc = (
                    "high" if s["risk_score"] > 70
                    else "medium" if s["risk_score"] > 40
                    else "low"
                )
                st.markdown(
                    f"**{s['section']}** &nbsp;&mdash;&nbsp; "
                    f"{s['risk_score']}% risk &nbsp;|&nbsp; {s['compliance']} "
                    f'&nbsp;<span class="risk-badge-{bc}">{level}</span>',
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
                    st.markdown(
                        f'<div class="evidence-box">"{s["evidence"]}"</div>',
                        unsafe_allow_html=True,
                    )

                    section_labels = s.get("policy_labels", [])
                    if section_labels:
                        st.markdown("**Policy Risk Labels**")
                        tags = " ".join(
                            f'<span class="label-chip">{lbl}</span>'
                            for lbl in section_labels
                        )
                        st.markdown(tags, unsafe_allow_html=True)

                    if len(s["attention_weights"]) > 1:
                        fig_attn = go.Figure(go.Bar(
                            x=[
                                f"Chunk {i}"
                                for i in range(len(s["attention_weights"]))
                            ],
                            y=s["attention_weights"],
                            marker_color=[
                                gauge_color
                                if w == max(s["attention_weights"])
                                else _accent
                                for w in s["attention_weights"]
                            ],
                            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
                        ))
                        fig_attn.update_layout(
                            title={
                                "text": "Chunk Attention Weights",
                                "font": {"size": 12, "color": _text},
                            },
                            height=200,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=35, b=15, l=15, r=15),
                            font={"color": _text},
                            xaxis={"gridcolor": _border, "tickcolor": _muted},
                            yaxis={"gridcolor": _border, "tickcolor": _muted},
                        )
                        st.plotly_chart(fig_attn, use_container_width=True)
        else:
            st.success("No high-risk sections detected.")

    # ── Tab 2: Highlighted PDF viewer ─────────────────────────────────────────
    with tab_pdf:
        pdf_bytes = active_history.get("pdf_bytes") if active_history else None

        if not pdf_bytes:
            st.info(
                "PDF data is not available for this result. "
                "Re-upload the document to use the highlighted viewer."
            )
        else:
            flagged_secs = [s for s in data["sections"] if s["risk_score"] > 40]
            if not flagged_secs:
                st.success("No risky sections — nothing to highlight.")
            else:
                st.markdown(f"""
                <div class="pdf-legend">
                    <span>
                        <span class="legend-swatch"
                              style="background:rgba(207,34,46,0.55)"></span>
                        High risk (&gt;70%)
                    </span>
                    <span>
                        <span class="legend-swatch"
                              style="background:rgba(255,210,25,0.8)"></span>
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

else:
    st.markdown(f"""
    <div style="text-align:center; padding:4rem 0; color:{_muted};">
        <div style="font-size:1.4rem; font-weight:600; margin-bottom:0.5rem; color:{_text};">
            No document loaded
        </div>
        <div style="font-size:0.92rem;">
            Upload a PDF above and click <strong>Analyze</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)


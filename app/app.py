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
p, li, div, span, h1, h2, h3, h4, h5, label,
.stMarkdown, .stText {{
    color: {_text} !important;
}}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {{
    color: {_text} !important;
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

/* ── Expanders ── */
[data-testid="stExpander"] {{
    background-color: {_bg2} !important;
    border: 1px solid {_border} !important;
    border-radius: 8px !important;
}}
details summary {{ color: {_text} !important; }}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {{
    background-color: {_accent} !important;
}}

/* ── Page 1: hero ── */
.hero {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    padding: 5rem 1rem 2.5rem;
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
    font-size: 2.4rem;
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
.badge-high   {{ background: #ffebe9; color: #cf222e; }}
.badge-medium {{ background: #fff8c5; color: #9a6700; }}
.badge-low    {{ background: #dafbe1; color: #1a7f37; }}

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
    line-height: 1.6;
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
        mc4.metric("Sector", sector_display or "—")

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
                    st.markdown(
                        f'<div class="evidence-box">"{s["evidence"]}"</div>',
                        unsafe_allow_html=True,
                    )
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


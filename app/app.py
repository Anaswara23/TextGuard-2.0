import streamlit as st
import plotly.graph_objects as go
import requests
import json
import os
from datetime import datetime

st.set_page_config(
    page_title="TextGuard 2.0",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Adaptive hero ─────────────────────────────────────────── */
.hero-title {
    font-size: 2.6rem; font-weight: 800;
    color: var(--text-color);
    text-align: center; margin-top: 2rem; margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem; color: var(--text-color); opacity: 0.6;
    text-align: center; margin-bottom: 2rem;
}
.section-header {
    font-size: 1.05rem; font-weight: 600; color: var(--text-color);
    opacity: 0.85; margin-bottom: 0.8rem; margin-top: 0.5rem;
    border-bottom: 1px solid rgba(128,128,128,0.2); padding-bottom: 0.4rem;
}

/* ── Document history cards ────────────────────────────────── */
.doc-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px; padding: 1rem 1.1rem 0.9rem;
    transition: box-shadow 0.2s;
}
.doc-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.12); }
.doc-card-title {
    font-size: 0.82rem; font-weight: 600; color: var(--text-color);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    margin-bottom: 0.2rem;
}
.doc-card-date {
    font-size: 0.72rem; color: var(--text-color); opacity: 0.5;
    margin-bottom: 0.6rem;
}
.doc-card-stats { display: flex; gap: 1.2rem; margin-top: 0.6rem; }
.stat-item { font-size: 0.73rem; color: var(--text-color); opacity: 0.65; }
.stat-val  { font-size: 1rem; font-weight: 700; opacity: 1; color: var(--text-color); }
.stat-risk  { color: #e53e3e !important; }
.stat-ok    { color: #38a169 !important; }
.stat-score { color: #d69e2e !important; }

/* ── Risk badges (work in both modes) ─────────────────────── */
.risk-badge-high {
    background: #fed7d7; color: #c53030;
    border-radius: 6px; padding: 2px 9px;
    font-size: 0.72rem; font-weight: 700;
}
.risk-badge-medium {
    background: #fefcbf; color: #975a16;
    border-radius: 6px; padding: 2px 9px;
    font-size: 0.72rem; font-weight: 700;
}
.risk-badge-low {
    background: #c6f6d5; color: #276749;
    border-radius: 6px; padding: 2px 9px;
    font-size: 0.72rem; font-weight: 700;
}

/* ── Evidence box ──────────────────────────────────────────── */
.evidence-box {
    background: var(--secondary-background-color);
    border-left: 3px solid #4a90d9; border-radius: 4px;
    padding: 0.8rem 1rem; font-style: italic; font-size: 0.9rem;
    color: var(--text-color); margin-top: 0.5rem; line-height: 1.5;
}

/* ── Policy label chips ────────────────────────────────────── */
.label-chip {
    background: rgba(74,144,217,0.15); color: #4a90d9;
    border: 1px solid rgba(74,144,217,0.3);
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.76rem; margin: 2px; display: inline-block;
}
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

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "active_result" not in st.session_state:
    st.session_state.active_result = None
if "active_filename" not in st.session_state:
    st.session_state.active_filename = None

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🛡️ TextGuard 2.0</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Automated risk &amp; compliance auditor for IDB project documents<br>'
    'Upload a PDF, select the sector, and get section-level risk scores powered by Hierarchical Attention Networks.</div>',
    unsafe_allow_html=True
)

# ── Upload panel ───────────────────────────────────────────────────────────────
up_col, sec_col, btn_col = st.columns([3, 2, 1])
with up_col:
    uploaded = st.file_uploader("Upload PDF", type="pdf", label_visibility="collapsed")
with sec_col:
    sector_choice = st.selectbox("Sector", options=list(SECTOR_OPTIONS.keys()), label_visibility="collapsed")
with btn_col:
    analyze_clicked = st.button("Analyze →", disabled=uploaded is None,
                                use_container_width=True, type="primary")

st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
if uploaded and analyze_clicked:
    selected_sector = SECTOR_OPTIONS[sector_choice]
    with st.spinner(f"Analyzing {uploaded.name} …"):
        try:
            post_data = {"sector": selected_sector} if selected_sector else {}
            response = requests.post(
                f"{API_URL}/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                data=post_data,
                timeout=120
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to TextGuard API. Make sure the backend is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.stop()

    st.session_state.history = [h for h in st.session_state.history if h["filename"] != uploaded.name]
    st.session_state.history.insert(0, {
        "filename": uploaded.name,
        "date": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
        "data": data,
    })
    st.session_state.active_result = data
    st.session_state.active_filename = uploaded.name
    st.rerun()

# ── History grid ───────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown('<div class="section-header">Previously Analyzed Documents</div>', unsafe_allow_html=True)
    COLS = 5
    for row_start in range(0, len(st.session_state.history), COLS):
        row_entries = st.session_state.history[row_start:row_start + COLS]
        grid = st.columns(COLS)
        for col, entry in zip(grid, row_entries):
            d = entry["data"]
            risk = d["overall_risk"]
            n_risk = d["sections_flagged"]
            n_ok = d["sections_analyzed"] - d["sections_flagged"]
            badge_cls = "risk-badge-high" if risk > 70 else "risk-badge-medium" if risk > 40 else "risk-badge-low"
            badge_lbl = "High Risk" if risk > 70 else "Medium Risk" if risk > 40 else "Low Risk"
            fname_short = entry["filename"][:26] + "…" if len(entry["filename"]) > 26 else entry["filename"]
            active_border = "#58a6ff" if entry["filename"] == st.session_state.active_filename else "#30363d"

            with col:
                st.markdown(f"""
                <div class="doc-card" style="border-color:{active_border}">
                    <div class="doc-card-title" title="{entry['filename']}">{fname_short}</div>
                    <div class="doc-card-date">{entry['date']}</div>
                    <span class="{badge_cls}">{badge_lbl}</span>
                    <div class="doc-card-stats">
                        <div class="stat-item">Risk<br><span class="stat-val stat-risk">{n_risk}</span></div>
                        <div class="stat-item">No risk<br><span class="stat-val stat-ok">{n_ok}</span></div>
                        <div class="stat-item">Risk score<br><span class="stat-val stat-score">{risk}%</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("View →", key=f"view_{entry['filename']}", use_container_width=True):
                    st.session_state.active_result = entry["data"]
                    st.session_state.active_filename = entry["filename"]
                    st.rerun()

# ── Result detail ──────────────────────────────────────────────────────────────
data = st.session_state.active_result
filename = st.session_state.active_filename

if data:
    st.divider()
    risk = data["overall_risk"]

    hcol1, hcol2 = st.columns([5, 1])
    with hcol1:
        st.subheader(f"📄 {filename}")
        sector_display = data.get("sector_display", "")
        if sector_display:
            prefix = "Detected Sector" if data.get("sector_source") == "detected" else "Sector"
            st.badge(f"🏷️ {prefix}: {sector_display}", color="blue")
    with hcol2:
        st.download_button(
            "📥 Download Report",
            data=json.dumps(data, indent=2),
            file_name=f"textguard_{filename.replace('.pdf','')}.json",
            mime="application/json",
            use_container_width=True,
        )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Overall Risk Score", f"{risk}%",
               "High Risk" if risk > 70 else "Medium Risk" if risk > 40 else "Low Risk")
    mc2.metric("Compliance Status", data["overall_compliance"])
    mc3.metric("Sections Flagged", f"{data['sections_flagged']} / {data['sections_analyzed']}")
    mc4.metric("Sector", sector_display or "—")

    st.divider()
    col_gauge, col_breakdown = st.columns([1, 2])

    with col_gauge:
        gauge_color = "crimson" if risk > 70 else "orange" if risk > 40 else "green"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            title={"text": "Document Risk Score", "font": {"size": 13}},
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 40],   "color": "rgba(56,161,105,0.15)"},
                    {"range": [40, 70],  "color": "rgba(214,158,46,0.15)"},
                    {"range": [70, 100], "color": "rgba(229,62,62,0.15)"},
                ],
                "bgcolor": "rgba(0,0,0,0)",
            }
        ))
        fig.update_layout(
            height=260, margin=dict(t=40, b=10, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        policy_labels = data.get("policy_labels", [])
        if policy_labels:
            st.markdown("**Policy Risk Labels:**")
            tags = " ".join(f'<span class="label-chip">{lbl}</span>' for lbl in policy_labels)
            st.markdown(tags, unsafe_allow_html=True)

    with col_breakdown:
        st.subheader("Section Risk Breakdown")
        for s in sorted(data["sections"], key=lambda x: -x["risk_score"]):
            icon = "🔴" if s["risk_score"] > 70 else "🟡" if s["risk_score"] > 40 else "🟢"
            st.markdown(f"{icon} **{s['section']}** — {s['risk_score']}% | {s['compliance']}")
            st.progress(s["risk_score"] / 100)

    st.divider()
    flagged = [s for s in data["sections"] if s["risk_score"] > 40]
    if flagged:
        st.subheader(f"⚠️ Flagged Sections ({len(flagged)})")
        for s in sorted(flagged, key=lambda x: -x["risk_score"]):
            with st.expander(f"**{s['section']}** — {s['risk_score']}% risk | {s['compliance']}"):
                st.markdown("**Key Evidence** *(highest-attention chunk)*:")
                st.markdown(f'<div class="evidence-box">"{s["evidence"]}"</div>', unsafe_allow_html=True)

                section_labels = s.get("policy_labels", [])
                if section_labels:
                    st.markdown("**Policy Risk Labels:**")
                    tags = " ".join(f'<span class="label-chip">{lbl}</span>' for lbl in section_labels)
                    st.markdown(tags, unsafe_allow_html=True)

                if len(s["attention_weights"]) > 1:
                    fig_attn = go.Figure(go.Bar(
                        x=[f"Chunk {i}" for i in range(len(s["attention_weights"]))],
                        y=s["attention_weights"],
                        marker_color=[
                            "#e74c3c" if w == max(s["attention_weights"]) else "#4a90d9"
                            for w in s["attention_weights"]
                        ],
                        hovertemplate="%{x}: %{y:.4f}<extra></extra>"
                    ))
                    fig_attn.update_layout(
                        title="Chunk Attention Weights", height=220,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=35, b=15, l=15, r=15)
                    )
                    st.plotly_chart(fig_attn, use_container_width=True)
    else:
        st.success("✅ No high-risk sections detected.")

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#8b949e">
        <div style="font-size:2.5rem">📂</div>
        <div style="margin-top:0.5rem">Upload a document and click <strong>Analyze →</strong> to get started</div>
    </div>
    """, unsafe_allow_html=True)
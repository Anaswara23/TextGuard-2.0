import streamlit as st
import plotly.graph_objects as go
import requests
import pandas as pd
import json
import os

st.set_page_config(
    page_title="TextGuard 2.0",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700;}
    .sub-header {color: #aaa; font-size: 1rem; margin-bottom: 2rem;}
    .evidence-box {
        background: #1e2a3a;
        border-left: 3px solid #4a90d9;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-style: italic;
        font-size: 0.9rem;
        color: #d0e4f7;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Use Streamlit secrets (Streamlit Cloud) → env var → localhost fallback
try:
    API_URL = st.secrets["app"]["API_URL"]
except Exception:
    API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.markdown('<p class="main-header">🛡️ TextGuard 2.0</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Automated risk & compliance triage for IDB project documents · Powered by Hierarchical Attention Networks</p>', unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    uploaded = st.file_uploader("Upload IDB Project Document (PDF)", type="pdf")

with col_info:
    st.markdown("**What TextGuard analyzes:**")
    st.markdown("- Risk signals in each section")
    st.markdown("- Compliance with IDB policy requirements")
    st.markdown("- Which specific text triggered each flag")

if uploaded:
    with st.spinner(f"Analyzing {uploaded.name}..."):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
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
    
    st.success(f"Analysis complete — {data['sections_analyzed']} sections analyzed")

    sector_display = data.get('sector_display', '')
    if sector_display:
        st.markdown(
            f'<span style="background:#1e4d8c;color:#fff;padding:5px 14px;'
            f'border-radius:14px;font-size:0.85rem;font-weight:600;display:inline-block;'
            f'margin-bottom:0.5rem">🏷️ Detected Sector: {sector_display}</span>',
            unsafe_allow_html=True
        )
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    risk = data['overall_risk']
    status_text = "High Risk" if risk > 70 else "Medium Risk" if risk > 40 else "Low Risk"
    col1.metric("Overall Risk Score", f"{risk}%", status_text)
    col2.metric("Compliance Status", data['overall_compliance'])
    col3.metric("Sections Flagged", f"{data['sections_flagged']} / {data['sections_analyzed']}")
    col4.metric("Document", uploaded.name[:25] + "..." if len(uploaded.name) > 25 else uploaded.name)
    
    st.divider()
    col_gauge, col_breakdown = st.columns([1, 2])
    
    with col_gauge:
        gauge_color = "crimson" if risk > 70 else "orange" if risk > 40 else "green"
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk,
            title={'text': "Document Risk Score", 'font': {'size': 14}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40],  'color': "#d4edda"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100],'color': "#f8d7da"}
                ],
            }
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=20, l=20, r=20),
                          template="plotly_dark",
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
        policy_labels = data.get('policy_labels', [])
        if policy_labels:
            st.markdown("**Policy Risk Labels:**")
            tags_html = " ".join(
                f'<span style="background:#1a3a5c;color:#7eb8f7;padding:3px 10px;'
                f'border-radius:10px;font-size:0.8rem;margin:2px;display:inline-block">'
                f'{lbl}</span>'
                for lbl in policy_labels
            )
            st.markdown(tags_html, unsafe_allow_html=True)
    
    with col_breakdown:
        st.subheader("Section Risk Breakdown")
        sections = sorted(data['sections'], key=lambda x: -x['risk_score'])
        for s in sections:
            icon = "🔴" if s['risk_score'] > 70 else "🟡" if s['risk_score'] > 40 else "🟢"
            st.markdown(f"{icon} **{s['section']}** — {s['risk_score']}% risk | {s['compliance']}")
            st.progress(s['risk_score'] / 100)
    
    st.divider()
    flagged_sections = [s for s in data['sections'] if s['risk_score'] > 40]
    
    if flagged_sections:
        st.subheader(f"⚠️ Flagged Sections — Evidence ({len(flagged_sections)} sections)")
        for s in sorted(flagged_sections, key=lambda x: -x['risk_score']):
            with st.expander(f"**{s['section']}** — {s['risk_score']}% risk | {s['compliance']}"):
                st.markdown("**Key Evidence** *(highest-attention chunk)*:")
                st.markdown(f'<div class="evidence-box">"{s["evidence"]}"</div>', unsafe_allow_html=True)

                section_labels = s.get('policy_labels', [])
                if section_labels:
                    st.markdown("**Policy Risk Labels:**")
                    tags_html = " ".join(
                        f'<span style="background:#1a3a5c;color:#7eb8f7;padding:3px 10px;'
                        f'border-radius:10px;font-size:0.8rem;margin:2px;display:inline-block">'
                        f'{lbl}</span>'
                        for lbl in section_labels
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)

                if len(s['attention_weights']) > 1:
                    weights_df = pd.DataFrame({
                        'Chunk': [f"Chunk {i}" for i in range(len(s['attention_weights']))],
                        'Attention': s['attention_weights']
                    })
                    fig_attn = go.Figure(go.Bar(
                        x=weights_df['Chunk'],
                        y=weights_df['Attention'],
                        marker_color=['#e74c3c' if w == max(s['attention_weights']) else '#4a90d9'
                                     for w in s['attention_weights']],
                        hovertemplate='%{x}: %{y:.4f}<extra></extra>'
                    ))
                    fig_attn.update_layout(
                        title="Chunk Attention Weights",
                        height=240,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=35, b=20, l=20, r=20),
                        font=dict(color="#ccc")
                    )
                    st.plotly_chart(fig_attn, use_container_width=True)
    else:
        st.success("✅ No high-risk sections detected. Document appears compliant.")
    
    report_json = json.dumps(data, indent=2)
    st.download_button(
        label="📥 Download Full Report (JSON)",
        data=report_json,
        file_name=f"textguard_report_{uploaded.name.replace('.pdf', '')}.json",
        mime="application/json"
    )

else:
    st.info("👆 Upload an IDB project PDF to begin analysis")
    st.markdown("---")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown("**1. Parse**\nPDF → sections")
    col_b.markdown("**2. Encode**\nBERT encodes chunks")
    col_c.markdown("**3. Attend**\nAttention weighs chunks")
    col_d.markdown("**4. Score**\nRisk + compliance scores")
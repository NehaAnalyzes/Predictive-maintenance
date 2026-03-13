# =============================================================================
# PREDICTIVE MAINTENANCE — STREAMLIT DASHBOARD (Dark Industrial UI)
# File    : dashboard/app.py
# Run     : python -m streamlit run dashboard/app.py
# Note    : Start FastAPI first — python -m uvicorn api.main:app --reload
# =============================================================================

import streamlit as st
import requests

st.set_page_config(
    page_title="PredictiveMaint",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://127.0.0.1:8000/predict"

# =============================================================================
# CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #0f1117 !important; }

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2d40 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; font-size: 12px !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #475569 !important; font-size: 10px !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }

.stSlider label { color: #94a3b8 !important; font-size: 12px !important; }
.stSlider [data-testid="stTickBar"] { display: none; }

.stButton > button {
    background: #2563eb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 12px !important;
    letter-spacing: 0.03em !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1d4ed8 !important; }

.stSelectbox > div > div {
    background: #0f1117 !important;
    border: 1px solid #1e2d40 !important;
    color: #e2e8f0 !important;
    border-radius: 6px !important;
}

div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3 { color: #e2e8f0 !important; }

.topbar {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 14px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}
.topbar-brand { font-size: 16px; font-weight: 500; color: #e2e8f0; display: flex; align-items: center; gap: 10px; }
.live-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #22c55e; }
.live-lbl { font-size: 11px; color: #22c55e; font-weight: 500; letter-spacing: 0.06em; }
.topbar-right { font-size: 12px; color: #475569; }

.kpi-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 10px;
    padding: 16px 18px;
    height: 100%;
}
.kpi-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 10px; }
.kpi-name { font-size: 10px; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; }
.badge-ok   { font-size: 10px; font-weight: 500; padding: 2px 7px; border-radius: 20px; background: #052e16; color: #22c55e; }
.badge-bad  { font-size: 10px; font-weight: 500; padding: 2px 7px; border-radius: 20px; background: #2d0707; color: #ef4444; }
.badge-warn { font-size: 10px; font-weight: 500; padding: 2px 7px; border-radius: 20px; background: #431407; color: #f97316; }
.kpi-num  { font-size: 26px; font-weight: 500; color: #f1f5f9; letter-spacing: -0.02em; }
.kpi-unit { font-size: 12px; color: #475569; }
.kpi-sub  { font-size: 11px; color: #475569; margin-top: 4px; }

.result-normal {
    background: #052e16; border: 1px solid #166534; border-radius: 10px;
    padding: 22px 24px; display: flex; align-items: center; gap: 16px;
}
.result-failure {
    background: #2d0707; border: 1px solid #7f1d1d; border-radius: 10px;
    padding: 22px 24px; display: flex; align-items: center; gap: 16px;
}
.result-idle {
    background: #161b27; border: 1px solid #1e2d40; border-radius: 10px;
    padding: 22px 24px; display: flex; align-items: center; gap: 16px;
}
.res-icon  { font-size: 36px; flex-shrink: 0; }
.res-title-ok   { font-size: 20px; font-weight: 500; color: #22c55e; }
.res-title-fail { font-size: 20px; font-weight: 500; color: #ef4444; }
.res-title-idle { font-size: 20px; font-weight: 500; color: #475569; }
.res-sub { font-size: 12px; margin-top: 4px; color: #64748b; }

.prob-bar-track { background: #1e2d40; border-radius: 99px; height: 8px; overflow: hidden; margin: 8px 0; }
.prob-section { background: #161b27; border: 1px solid #1e2d40; border-radius: 10px; padding: 16px 18px; margin-top: 12px; }
.prob-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.prob-lbl  { font-size: 10px; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }

.info-panel { background: #161b27; border: 1px solid #1e2d40; border-radius: 10px; padding: 16px 18px; }
.info-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1e2d40; font-size: 12px; }
.info-row:last-child { border-bottom: none; }
.info-key { color: #64748b; }
.info-val { color: #e2e8f0; font-weight: 500; }
.panel-title { font-size: 10px; color: #475569; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px; }

.flag-bad { display:inline-block; background:#2d0707; color:#fca5a5; font-size:11px; padding:3px 9px; border-radius:5px; margin:2px; }
.flag-ok  { display:inline-block; background:#052e16; color:#4ade80; font-size:11px; padding:3px 9px; border-radius:5px; margin:2px; }

.section-div { border: none; border-top: 1px solid #1e2d40; margin: 14px 0; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙ Sensor Inputs")
    st.markdown('<hr style="border-top:1px solid #1e2d40;margin:12px 0;">', unsafe_allow_html=True)

    st.markdown('<p style="font-size:10px;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">MACHINE</p>', unsafe_allow_html=True)
    machine_type = st.selectbox("Machine Type", options=[0,1,2],
        format_func=lambda x: {0:"L — Low",1:"M — Medium",2:"H — High"}[x],
        label_visibility="collapsed")

    st.markdown('<p style="font-size:10px;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin:14px 0 6px;">THERMAL</p>', unsafe_allow_html=True)
    air_temp     = st.slider("Air Temperature (K)",     295.0, 305.0, 300.0, 0.1)
    process_temp = st.slider("Process Temperature (K)", 305.0, 315.0, 310.0, 0.1)

    st.markdown('<p style="font-size:10px;color:#475569;letter-spacing:0.1em;text-transform:uppercase;margin:14px 0 6px;">MECHANICAL</p>', unsafe_allow_html=True)
    rpm       = st.slider("Rotational Speed (RPM)", 1000, 3000, 1500, 10)
    torque    = st.slider("Torque (Nm)",            0.0,  80.0, 40.0, 0.5)
    tool_wear = st.slider("Tool Wear (min)",        0,    300,  100,  1)

    st.markdown('<hr style="border-top:1px solid #1e2d40;margin:16px 0;">', unsafe_allow_html=True)
    predict_clicked = st.button("Run Prediction", type="primary", use_container_width=True)

    st.markdown(f"""
    <div style="margin-top:16px; font-size:11px; color:#475569; line-height:2;">
        <b style="color:#64748b">Model</b><br>
        Random Forest · F1 0.81 · AUC 0.98<br>
        Threshold: 0.65 (tuned)
    </div>""", unsafe_allow_html=True)

# =============================================================================
# Compute derived features
# =============================================================================
import math
power     = torque * (rpm * 2 * math.pi / 60)
temp_diff = process_temp - air_temp
strain    = tool_wear * torque
pwf_risk  = not (3500 <= power <= 9000)
hdf_risk  = temp_diff < 8.6
osf_risk  = strain > 11460
flag_count = sum([pwf_risk, hdf_risk, osf_risk])

# =============================================================================
# Topbar
# =============================================================================
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        🔧 PredictiveMaint
        <span class="live-dot"></span>
        <span class="live-lbl">LIVE</span>
    </div>
    <div class="topbar-right">Random Forest &nbsp;|&nbsp; F1: 0.81 &nbsp;|&nbsp; AUC: 0.98 &nbsp;|&nbsp; Threshold: 0.65</div>
</div>""", unsafe_allow_html=True)

# =============================================================================
# KPI Row
# =============================================================================
c1, c2, c3, c4 = st.columns(4)

def badge(ok, ok_label="OK", bad_label="RISK"):
    cls = "badge-ok" if ok else "badge-bad"
    lbl = ok_label if ok else bad_label
    return f'<span class="{cls}">{lbl}</span>'

with c1:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-top"><span class="kpi-name">Power Output</span>{badge(not pwf_risk)}</div>
        <div><span class="kpi-num">{power:,.0f}</span><span class="kpi-unit"> W</span></div>
        <div class="kpi-sub">Safe: 3,500 – 9,000 W</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-top"><span class="kpi-name">Temp Difference</span>{badge(not hdf_risk)}</div>
        <div><span class="kpi-num">{temp_diff:.1f}</span><span class="kpi-unit"> K</span></div>
        <div class="kpi-sub">Min: 8.6 K (HDF risk)</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-top"><span class="kpi-name">Strain Index</span>{badge(not osf_risk)}</div>
        <div><span class="kpi-num">{strain:,.0f}</span></div>
        <div class="kpi-sub">Max: 11,460 (OSF risk)</div>
    </div>""", unsafe_allow_html=True)

with c4:
    flag_badge = f'<span class="badge-ok">CLEAR</span>' if flag_count == 0 else \
                 f'<span class="badge-warn">{flag_count} RISK</span>' if flag_count == 1 else \
                 f'<span class="badge-bad">{flag_count} RISKS</span>'
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-top"><span class="kpi-name">Risk Flags</span>{flag_badge}</div>
        <div><span class="kpi-num">{flag_count}</span><span class="kpi-unit"> / 3</span></div>
        <div class="kpi-sub">{"All parameters normal" if flag_count == 0 else f"{flag_count} parameter(s) out of range"}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

# =============================================================================
# Prediction Result + Info
# =============================================================================
col_main, col_info = st.columns([3, 2])

with col_main:
    if predict_clicked:
        payload = {
            "Type": machine_type,
            "air_temperature": air_temp,
            "process_temperature": process_temp,
            "rotational_speed": rpm,
            "torque": torque,
            "tool_wear": tool_wear,
        }
        try:
            r    = requests.post(API_URL, json=payload, timeout=10).json()
            prob = r["failure_probability"]
            pred = r["prediction"]
            thresh = r["threshold_used"]

            if pred == 1:
                st.markdown(f"""<div class="result-failure">
                    <div class="res-icon">⚠</div>
                    <div>
                        <div class="res-title-fail">Failure Predicted</div>
                        <div class="res-sub">Immediate maintenance recommended</div>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="result-normal">
                    <div class="res-icon">✓</div>
                    <div>
                        <div class="res-title-ok">Normal Operation</div>
                        <div class="res-sub">Machine operating within safe parameters</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            color = "#ef4444" if prob >= 0.65 else "#f97316" if prob >= 0.4 else "#22c55e"
            st.markdown(f"""
            <div class="prob-section">
                <div class="prob-head">
                    <span class="prob-lbl">Failure Probability</span>
                    <span style="font-size:22px;font-weight:500;color:{color};">{prob*100:.1f}%</span>
                </div>
                <div class="prob-bar-track">
                    <div style="width:{prob*100:.1f}%;height:8px;border-radius:99px;background:{color};"></div>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:10px;color:#475569;margin-top:4px;">
                    <span>0%</span><span style="color:#f97316;">▲ {thresh*100:.0f}% threshold</span><span>100%</span>
                </div>
            </div>""", unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.markdown("""<div class="result-idle">
                <div class="res-icon">✕</div>
                <div>
                    <div class="res-title-idle">API Disconnected</div>
                    <div class="res-sub">Run: python -m uvicorn api.main:app --reload</div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""<div class="result-idle">
            <div class="res-icon">—</div>
            <div>
                <div class="res-title-idle">Awaiting Prediction</div>
                <div class="res-sub">Adjust sliders and click Run Prediction</div>
            </div>
        </div>""", unsafe_allow_html=True)

with col_info:
    type_label = {0:"L — Low", 1:"M — Medium", 2:"H — High"}[machine_type]
    st.markdown(f"""
    <div class="info-panel">
        <div class="panel-title">Input Summary</div>
        <div class="info-row"><span class="info-key">Machine Type</span><span class="info-val">{type_label}</span></div>
        <div class="info-row"><span class="info-key">Air Temp</span><span class="info-val">{air_temp:.1f} K</span></div>
        <div class="info-row"><span class="info-key">Process Temp</span><span class="info-val">{process_temp:.1f} K</span></div>
        <div class="info-row"><span class="info-key">Rotational Speed</span><span class="info-val">{rpm} RPM</span></div>
        <div class="info-row"><span class="info-key">Torque</span><span class="info-val">{torque:.1f} Nm</span></div>
        <div class="info-row"><span class="info-key">Tool Wear</span><span class="info-val">{tool_wear} min</span></div>
    </div>""", unsafe_allow_html=True)

    flags_html = ""
    flags_html += f'<span class="{"flag-bad" if pwf_risk else "flag-ok"}">{"⚡ PWF risk" if pwf_risk else "Power OK"}</span>'
    flags_html += f'<span class="{"flag-bad" if hdf_risk else "flag-ok"}">{"🌡 HDF risk" if hdf_risk else "Temp OK"}</span>'
    flags_html += f'<span class="{"flag-bad" if osf_risk else "flag-ok"}">{"🔩 OSF risk" if osf_risk else "Strain OK"}</span>'

    st.markdown(f"""
    <div class="info-panel" style="margin-top:12px;">
        <div class="panel-title">Risk Flags</div>
        {flags_html}
    </div>""", unsafe_allow_html=True)
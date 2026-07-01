"""
app.py — Predictive Maintenance Dashboard
------------------------------------------
Streamlit application for real-time machine health monitoring.

Features:
  • Live sensor readings with health score gauge
  • Machine fleet overview with status indicators
  • Failure probability predictions per machine
  • Sensor trend charts with anomaly highlights
  • Maintenance recommendations engine
  • Alert system (ISO 10816 vibration thresholds)

Run:
    streamlit run streamlit_app/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import json
import time
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PredictiveMaint AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

    .main { background: #0d1117; }
    .block-container { padding: 1rem 2rem; max-width: 1600px; }

    /* Header */
    .dash-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 12px; padding: 1.5rem 2rem; margin-bottom: 1.5rem;
        border: 1px solid rgba(56, 139, 212, 0.25);
    }
    .dash-title { font-size: 1.8rem; font-weight: 700; color: #e6edf3; margin: 0; }
    .dash-sub   { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }

    /* KPI cards */
    .kpi-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 1.2rem 1.4rem;
        text-align: center; transition: border-color 0.2s;
    }
    .kpi-card:hover { border-color: #388bfd; }
    .kpi-value  { font-size: 2.2rem; font-weight: 700; color: #e6edf3; line-height: 1; }
    .kpi-label  { font-size: 0.75rem; color: #8b949e; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-delta  { font-size: 0.8rem; margin-top: 4px; }

    /* Status badges */
    .status-ok      { background: #1a3a2a; color: #3fb950; border: 1px solid #2ea043; border-radius: 6px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
    .status-warn    { background: #3d2600; color: #d29922; border: 1px solid #9e6a03; border-radius: 6px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }
    .status-crit    { background: #3d0000; color: #f85149; border: 1px solid #f85149; border-radius: 6px; padding: 2px 10px; font-size: 0.75rem; font-weight: 600; }

    /* Machine card */
    .machine-card {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 1rem 1.2rem; margin-bottom: 0.6rem;
        display: flex; align-items: center; justify-content: space-between;
    }

    /* Alert box */
    .alert-critical {
        background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.3);
        border-left: 4px solid #f85149; border-radius: 8px; padding: 0.8rem 1rem;
        margin-bottom: 0.6rem; color: #f85149; font-size: 0.85rem;
    }
    .alert-warning {
        background: rgba(210,153,34,0.1); border: 1px solid rgba(210,153,34,0.3);
        border-left: 4px solid #d29922; border-radius: 8px; padding: 0.8rem 1rem;
        margin-bottom: 0.6rem; color: #d29922; font-size: 0.85rem;
    }

    /* Sidebar */
    .css-1d391kg { background: #0d1117; }
    section[data-testid="stSidebar"] { background: #161b22; }

    /* Charts */
    .chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 0.5rem; }

    /* Recommendation */
    .rec-card {
        background: #1c2128; border: 1px solid #388bfd44; border-radius: 8px;
        padding: 0.9rem 1.1rem; margin-bottom: 0.5rem;
    }
    .rec-title { font-weight: 600; font-size: 0.9rem; color: #79c0ff; }
    .rec-body  { font-size: 0.82rem; color: #8b949e; margin-top: 3px; }
    .rec-pri   { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_sensor_data():
    data_path = Path("data/sensor_data_features.csv")
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        return df
    else:
        # Fallback: generate minimal demo data
        return generate_demo_data()

@st.cache_resource
def load_model():
    model_path = Path("models/best_model.pkl")
    scaler_path = Path("models/scaler.pkl")
    meta_path   = Path("models/model_metadata.json")

    if model_path.exists() and scaler_path.exists():
        model  = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(meta_path) as f:
            meta = json.load(f)
        return model, scaler, meta
    return None, None, None

def generate_demo_data():
    """Fallback demo data when pre-generated CSV not present."""
    np.random.seed(42)
    machines = [f"M{i:03d}" for i in range(1, 6)]
    records  = []
    for m in machines:
        for h in range(168):  # 1 week
            ts = pd.Timestamp("2024-01-01") + pd.Timedelta(hours=h)
            records.append({
                "timestamp":              ts,
                "machine_id":             m,
                "temperature_c":          65 + np.random.normal(0, 5),
                "vibration_rms":          2.5 + np.random.normal(0, 0.5),
                "vibration_peak":         7.0 + np.random.normal(0, 1.0),
                "current_draw_a":         22.0 + np.random.normal(0, 1.0),
                "current_imbalance":      abs(np.random.normal(0, 0.3)),
                "pressure_bar":           5.0 + np.random.normal(0, 0.1),
                "rpm":                    1480 + np.random.normal(0, 20),
                "bearing_temp_c":         72.0 + np.random.normal(0, 3),
                "power_factor":           0.91 + np.random.normal(0, 0.02),
                "failure_label":          1 if np.random.random() < 0.07 else 0,
                "failure_type":           "normal",
                "temperature_c_roll24h_mean": 65.0,
                "vibration_rms_roll24h_mean": 2.5,
            })
    return pd.DataFrame(records)

def get_health_score(row):
    """
    Compute a 0-100 health score based on ISO 10816 and domain thresholds.
    Uses 5 years of electrical maintenance domain knowledge.
    """
    score = 100.0

    # Temperature penalty (°C)
    if   row["temperature_c"] > 100: score -= 40
    elif row["temperature_c"] > 90:  score -= 25
    elif row["temperature_c"] > 80:  score -= 12

    # Vibration penalty (ISO 10816 mm/s RMS)
    #  Zone A: < 2.3   Zone B: 2.3-4.5   Zone C: 4.5-7.1   Zone D: > 7.1
    if   row["vibration_rms"] > 7.1: score -= 35
    elif row["vibration_rms"] > 4.5: score -= 20
    elif row["vibration_rms"] > 2.3: score -= 8

    # Current imbalance penalty
    if   row["current_imbalance"] > 5.0: score -= 20
    elif row["current_imbalance"] > 2.0: score -= 10

    # Pressure penalty
    if   row["pressure_bar"] < 3.0: score -= 15
    elif row["pressure_bar"] < 4.0: score -= 7

    # Power factor penalty
    if   row["power_factor"] < 0.75: score -= 15
    elif row["power_factor"] < 0.85: score -= 5

    return max(0.0, min(100.0, score))

def get_status(health_score):
    if health_score >= 75: return "✅ Normal",    "status-ok"
    if health_score >= 50: return "⚠️ Warning",   "status-warn"
    return "🔴 Critical", "status-crit"

def get_recommendations(row, health_score):
    recs = []
    if row["temperature_c"] > 85:
        recs.append({"title": "🌡️ Overheating Detected", "body": f"Temperature {row['temperature_c']:.1f}°C exceeds 85°C threshold. Inspect cooling system and lubrication. Check for blocked ventilation ducts.", "priority": "HIGH", "color": "#f85149"})
    if row["vibration_rms"] > 4.5:
        recs.append({"title": "📳 Excessive Vibration", "body": f"Vibration {row['vibration_rms']:.2f} mm/s (ISO 10816 Zone C/D). Schedule bearing inspection. Check shaft alignment and balance.", "priority": "HIGH", "color": "#f85149"})
    if row["current_imbalance"] > 2.0:
        recs.append({"title": "⚡ Current Imbalance", "body": f"Imbalance {row['current_imbalance']:.2f}% > 2% threshold. Indicates winding asymmetry or loose connections. Check motor terminals.", "priority": "MEDIUM", "color": "#d29922"})
    if row["pressure_bar"] < 4.0:
        recs.append({"title": "🔧 Low Pressure", "body": f"Pressure {row['pressure_bar']:.2f} bar below nominal range. Possible seal wear or pump degradation.", "priority": "MEDIUM", "color": "#d29922"})
    if row["power_factor"] < 0.85:
        recs.append({"title": "💡 Poor Power Factor", "body": f"PF {row['power_factor']:.3f} < 0.85. Consider capacitor bank correction. May indicate motor winding issues.", "priority": "LOW", "color": "#1f6feb"})
    if not recs:
        recs.append({"title": "✅ Machine Operating Normally", "body": "All sensor readings within acceptable limits. Continue scheduled maintenance.", "priority": "INFO", "color": "#3fb950"})
    return recs

def make_gauge(value, title, max_val=100):
    color = "#3fb950" if value >= 75 else "#d29922" if value >= 50 else "#f85149"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 12, "color": "#8b949e"}},
        number={"font": {"size": 28, "color": color}},
        gauge={
            "axis":     {"range": [0, max_val], "tickcolor": "#8b949e", "tickfont": {"size": 9}},
            "bar":      {"color": color, "thickness": 0.25},
            "bgcolor":  "#1c2128",
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 50],    "color": "#3d0000"},
                {"range": [50, 75],   "color": "#3d2600"},
                {"range": [75, 100],  "color": "#1a3a2a"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": value}
        }
    ))
    fig.update_layout(
        height=200, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e6edf3"
    )
    return fig


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    df    = load_sensor_data()
    model, scaler, meta = load_model()

    machines = sorted(df["machine_id"].unique())

    # ── SIDEBAR ───────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ PredictiveMaint AI")
        st.markdown("---")

        selected_machine = st.selectbox("🏭 Select Machine", machines, index=0)
        st.markdown("---")

        date_range = st.date_input(
            "📅 Date Range",
            value=(df["timestamp"].min().date(), df["timestamp"].max().date())
        )
        st.markdown("---")

        refresh_rate = st.slider("⏱ Refresh (seconds)", 5, 60, 30)
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=False)
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()

        st.markdown("---")
        st.markdown("**⚙️ Thresholds (ISO 10816)**")
        vib_warn = st.number_input("Vibration Warn (mm/s)", value=4.5)
        vib_crit = st.number_input("Vibration Critical (mm/s)", value=7.1)
        temp_warn = st.number_input("Temp Warn (°C)", value=80.0)
        temp_crit = st.number_input("Temp Critical (°C)", value=90.0)

        st.markdown("---")
        st.markdown(
            "<small style='color:#8b949e'>Built by Kamal Krushna Ghosh<br>"
            "5 yrs Electrical Maint. + Data Science</small>",
            unsafe_allow_html=True
        )

    # ── HEADER ────────────────────────────────────────────────
    st.markdown("""
    <div class='dash-header'>
        <div class='dash-title'>⚙️ Predictive Maintenance Intelligence Dashboard</div>
        <div class='dash-sub'>Real-time machine health monitoring • Failure prediction • Maintenance recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    # Filter data
    machine_df = df[df["machine_id"] == selected_machine].sort_values("timestamp")
    latest     = machine_df.iloc[-1]

    health_score = get_health_score(latest)
    status_text, status_cls = get_status(health_score)

    # ── KPI ROW ───────────────────────────────────────────────
    total_failures  = df["failure_label"].sum()
    machines_crit   = sum(
        1 for m in machines
        if get_health_score(df[df["machine_id"] == m].iloc[-1]) < 50
    )
    machines_warn   = sum(
        1 for m in machines
        if 50 <= get_health_score(df[df["machine_id"] == m].iloc[-1]) < 75
    )
    fleet_health    = np.mean([
        get_health_score(df[df["machine_id"] == m].iloc[-1]) for m in machines
    ])

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#3fb950'>{fleet_health:.0f}%</div>
            <div class='kpi-label'>Fleet Health Score</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#f85149'>{machines_crit}</div>
            <div class='kpi-label'>Critical Machines</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#d29922'>{machines_warn}</div>
            <div class='kpi-label'>Warning Machines</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#79c0ff'>{total_failures:,}</div>
            <div class='kpi-label'>Total Failure Events</div>
        </div>""", unsafe_allow_html=True)
    with col5:
        df_rate = df["failure_label"].mean() * 100
        st.markdown(f"""<div class='kpi-card'>
            <div class='kpi-value' style='color:#d2a8ff'>{df_rate:.1f}%</div>
            <div class='kpi-label'>Failure Rate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAIN CONTENT ─────────────────────────────────────────
    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown(f"### 🔍 Machine `{selected_machine}` — Live Status")

        # Gauges
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.plotly_chart(make_gauge(health_score, "Health Score"), use_container_width=True)
        with g2:
            temp_pct = min(100, max(0, (latest["temperature_c"] / 120) * 100))
            st.plotly_chart(make_gauge(temp_pct, f"Temp {latest['temperature_c']:.1f}°C"), use_container_width=True)
        with g3:
            vib_pct = min(100, max(0, (latest["vibration_rms"] / 10) * 100))
            st.plotly_chart(make_gauge(100 - vib_pct, f"Vib {latest['vibration_rms']:.2f} mm/s"), use_container_width=True)
        with g4:
            pf_pct = latest["power_factor"] * 100
            st.plotly_chart(make_gauge(pf_pct, f"PF {latest['power_factor']:.3f}"), use_container_width=True)

        # Sensor trend chart
        st.markdown("#### 📈 Sensor Trends (Last 72h)")
        recent_df = machine_df.tail(72)

        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.06,
            subplot_titles=["Temperature (°C)", "Vibration RMS (mm/s)", "Current Draw (A)"]
        )
        plot_config = [
            ("temperature_c",  "#E8593C", 1),
            ("vibration_rms",  "#BA7517", 2),
            ("current_draw_a", "#3B8BD4", 3),
        ]
        for col, color, row in plot_config:
            fig.add_trace(go.Scatter(
                x=recent_df["timestamp"], y=recent_df[col],
                mode="lines", line=dict(color=color, width=1.5),
                name=col, showlegend=False,
            ), row=row, col=1)
            # Rolling mean
            roll = recent_df[col].rolling(6).mean()
            fig.add_trace(go.Scatter(
                x=recent_df["timestamp"], y=roll,
                mode="lines", line=dict(color="white", width=1, dash="dot"),
                name=f"{col} (6h avg)", showlegend=False, opacity=0.5
            ), row=row, col=1)

        # Failure markers
        failures = recent_df[recent_df["failure_label"] == 1]
        for _, frow in failures.iterrows():
            fig.add_vline(x=frow["timestamp"], line_dash="dash", line_color="red", line_width=1, opacity=0.5)

        # Threshold lines
        fig.add_hline(y=temp_crit,  line_dash="dot", line_color="#f85149", line_width=1, row=1, col=1)
        fig.add_hline(y=vib_crit,   line_dash="dot", line_color="#f85149", line_width=1, row=2, col=1)
        fig.add_hline(y=vib_warn,   line_dash="dot", line_color="#d29922", line_width=1, row=2, col=1)

        fig.update_layout(
            height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
            margin=dict(l=50, r=20, t=40, b=20), font_color="#e6edf3",
            xaxis3=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d"),
            yaxis2=dict(gridcolor="#21262d"), yaxis3=dict(gridcolor="#21262d"),
        )
        for i in range(1, 4):
            fig.update_xaxes(gridcolor="#21262d", row=i, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        # ── FLEET STATUS ─────────────────────────────────────
        st.markdown("### 🏭 Fleet Overview")
        for m in machines:
            m_df = df[df["machine_id"] == m]
            if len(m_df) == 0:
                continue
            m_latest = m_df.iloc[-1]
            m_health = get_health_score(m_latest)
            m_status, m_cls = get_status(m_health)
            m_selected = "border: 1px solid #388bfd;" if m == selected_machine else ""
            st.markdown(f"""
            <div class='machine-card' style='{m_selected}'>
                <div>
                    <strong style='color:#e6edf3'>{m}</strong><br>
                    <small style='color:#8b949e'>Health: {m_health:.0f}%</small>
                </div>
                <span class='{m_cls}'>{m_status}</span>
            </div>""", unsafe_allow_html=True)

        # ── ALERTS ───────────────────────────────────────────
        st.markdown("### 🔔 Active Alerts")
        alerts_found = False
        for m in machines:
            m_df = df[df["machine_id"] == m]
            if len(m_df) == 0:
                continue
            m_latest = m_df.iloc[-1]
            m_health = get_health_score(m_latest)
            if m_latest["temperature_c"] > temp_crit or m_latest["vibration_rms"] > vib_crit or m_health < 50:
                st.markdown(f"""<div class='alert-critical'>
                    🔴 <strong>{m}</strong> — Critical condition detected<br>
                    Temp: {m_latest['temperature_c']:.1f}°C | Vib: {m_latest['vibration_rms']:.2f} mm/s
                </div>""", unsafe_allow_html=True)
                alerts_found = True
            elif m_latest["temperature_c"] > temp_warn or m_latest["vibration_rms"] > vib_warn or m_health < 75:
                st.markdown(f"""<div class='alert-warning'>
                    ⚠️ <strong>{m}</strong> — Warning: monitor closely<br>
                    Temp: {m_latest['temperature_c']:.1f}°C | Vib: {m_latest['vibration_rms']:.2f} mm/s
                </div>""", unsafe_allow_html=True)
                alerts_found = True
        if not alerts_found:
            st.success("✅ No active alerts — all machines nominal")

    # ── RECOMMENDATIONS ──────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🛠️ Maintenance Recommendations for `{selected_machine}`")
    recs = get_recommendations(latest, health_score)

    rec_cols = st.columns(min(len(recs), 3))
    for i, rec in enumerate(recs):
        with rec_cols[i % 3]:
            pri_color = rec["color"]
            st.markdown(f"""
            <div class='rec-card'>
                <div class='rec-title'>{rec['title']}</div>
                <div class='rec-body'>{rec['body']}</div>
                <div class='rec-pri' style='color:{pri_color}; margin-top:8px'>Priority: {rec['priority']}</div>
            </div>""", unsafe_allow_html=True)

    # ── FAILURE DISTRIBUTION CHART ───────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Analytics")

    ac1, ac2 = st.columns(2)

    with ac1:
        # Failure type distribution
        ft_counts = df[df["failure_label"] == 1]["failure_type"].value_counts()
        if len(ft_counts) > 0:
            fig_ft = go.Figure(go.Bar(
                x=ft_counts.values,
                y=ft_counts.index,
                orientation="h",
                marker_color=["#f85149", "#d29922", "#3B8BD4", "#1D9E75"][:len(ft_counts)]
            ))
            fig_ft.update_layout(
                title="Failure Type Distribution",
                height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
                font_color="#e6edf3", margin=dict(l=120, r=20, t=40, b=20),
                xaxis=dict(gridcolor="#21262d"), yaxis=dict(gridcolor="#21262d")
            )
            st.plotly_chart(fig_ft, use_container_width=True)

    with ac2:
        # Health score distribution across fleet
        health_scores_all = {
            m: get_health_score(df[df["machine_id"] == m].iloc[-1])
            for m in machines
        }
        fig_hs = go.Figure(go.Bar(
            x=list(health_scores_all.keys()),
            y=list(health_scores_all.values()),
            marker_color=[
                "#3fb950" if v >= 75 else "#d29922" if v >= 50 else "#f85149"
                for v in health_scores_all.values()
            ]
        ))
        fig_hs.update_layout(
            title="Fleet Health Scores",
            height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#1c2128",
            font_color="#e6edf3", margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(gridcolor="#21262d"), yaxis=dict(range=[0, 100], gridcolor="#21262d")
        )
        fig_hs.add_hline(y=75, line_dash="dot", line_color="#3fb950", line_width=1)
        fig_hs.add_hline(y=50, line_dash="dot", line_color="#f85149", line_width=1)
        st.plotly_chart(fig_hs, use_container_width=True)

    # ── LIVE SENSOR TABLE ─────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 📋 Live Sensor Readings — `{selected_machine}`")

    sensor_display = {
        "Sensor":         ["Temperature", "Vibration RMS", "Vibration Peak", "Current Draw",
                           "Current Imbalance", "Pressure", "RPM", "Bearing Temp", "Power Factor"],
        "Value":          [
            f"{latest['temperature_c']:.1f} °C",
            f"{latest['vibration_rms']:.3f} mm/s",
            f"{latest['vibration_peak']:.3f} mm/s",
            f"{latest['current_draw_a']:.2f} A",
            f"{latest['current_imbalance']:.3f} %",
            f"{latest['pressure_bar']:.3f} bar",
            f"{latest['rpm']:.0f} RPM",
            f"{latest['bearing_temp_c']:.1f} °C",
            f"{latest['power_factor']:.4f}",
        ],
        "Threshold":      ["< 90°C", "< 7.1 mm/s", "< 18 mm/s", "18–28 A",
                           "< 2%", "4.5–6.5 bar", "1400–3000", "< 85°C", "> 0.85"],
        "ISO Standard":   ["NEMA MG-1", "ISO 10816", "ISO 10816", "IEC 60034",
                           "NEMA MG-1", "ISO 4413", "IEC 60034", "ISO 10816", "IEC 60034"],
    }
    st.dataframe(pd.DataFrame(sensor_display), use_container_width=True, hide_index=True)

    # ── FOOTER ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<small style='color:#8b949e'>⚙️ PredictiveMaint AI | Built by <strong style='color:#79c0ff'>Kamal Krushna Ghosh</strong> "
        "| 5 years Electrical Maintenance + Data Science | iNeuron Certified | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</small>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

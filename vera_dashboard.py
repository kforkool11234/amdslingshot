"""
V.E.R.A. | Virtual Energy Resilient Assistant
Grid-Edge Intelligence Dashboard
──────────────────────────────────────────────
Drop your model arrays here:
  y_true_load, y_pred_load   → actual vs predicted Load (kW)
  y_true_solar, y_pred_solar → actual vs predicted Solar (kW)
  y_true_wind, y_pred_wind   → actual vs predicted Wind (kW)

If no arrays are provided, the dashboard runs on 24-hour dummy data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import math

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="V.E.R.A. | Grid-Edge Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — Dark Terminal Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap');

    /* ── Root variables ── */
    :root {
        --bg-primary   : #080c12;
        --bg-card      : #0d1520;
        --bg-card2     : #101e2e;
        --accent-green : #00ff88;
        --accent-amber : #ffb300;
        --accent-red   : #ff3d3d;
        --accent-blue  : #00aaff;
        --accent-cyan  : #00e5ff;
        --text-main    : #c8d8e8;
        --text-dim     : #4a6a7a;
        --border       : #1a3040;
        --neon-green   : #39ff14;
    }

    /* ── Full dark background ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-main) !important;
        font-family: 'Share Tech Mono', monospace !important;
    }

    [data-testid="stSidebar"] {
        background-color: #060a0f !important;
        border-right: 1px solid var(--border);
    }

    /* ── Headers ── */
    h1, h2, h3, h4 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 2px;
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
    }
    [data-testid="stMetricLabel"] { color: var(--text-dim) !important; font-size: 11px !important; }
    [data-testid="stMetricValue"] { color: var(--accent-green) !important; font-family: 'Orbitron', sans-serif !important; }
    [data-testid="stMetricDelta"] { font-size: 11px !important; }

    /* ── Dividers ── */
    hr { border-color: var(--border) !important; }

    /* ── Tables ── */
    [data-testid="stDataFrame"] thead tr th {
        background: #0a1522 !important;
        color: var(--accent-cyan) !important;
        font-family: 'Orbitron', sans-serif !important;
        font-size: 11px !important;
        letter-spacing: 1px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: #1a3040; border-radius: 3px; }

    /* ── Custom card component ── */
    .vera-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .vera-card-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 11px;
        letter-spacing: 2px;
        color: var(--text-dim);
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .vera-val-green { font-family: 'Orbitron', sans-serif; font-size: 28px; color: var(--accent-green); }
    .vera-val-amber { font-family: 'Orbitron', sans-serif; font-size: 28px; color: var(--accent-amber); }
    .vera-val-red   { font-family: 'Orbitron', sans-serif; font-size: 28px; color: var(--accent-red);   }
    .vera-val-blue  { font-family: 'Orbitron', sans-serif; font-size: 28px; color: var(--accent-blue); }
    .vera-tag {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 4px;
        font-size: 11px;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        margin-top: 6px;
    }
    .tag-green { background: rgba(0,255,136,0.12); color: var(--accent-green); border: 1px solid var(--accent-green); }
    .tag-amber { background: rgba(255,179,0,0.12); color: var(--accent-amber); border: 1px solid var(--accent-amber); }
    .tag-red   { background: rgba(255,61,61,0.12);  color: var(--accent-red);   border: 1px solid var(--accent-red);   }

    /* ── Gauge bar ── */
    .gauge-bar-bg {
        background: #1a2a3a;
        border-radius: 6px;
        height: 18px;
        width: 100%;
        margin: 8px 0;
        position: relative;
        overflow: hidden;
    }
    .gauge-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    /* ── Survival horizon clock ── */
    .clock-display {
        font-family: 'Orbitron', sans-serif;
        font-size: 52px;
        text-align: center;
        letter-spacing: 6px;
        text-shadow: 0 0 20px currentColor;
        padding: 10px 0;
    }
    .clock-label {
        font-size: 11px;
        letter-spacing: 3px;
        text-align: center;
        color: var(--text-dim);
        margin-top: -4px;
    }

    /* ── P2P table row ── */
    .p2p-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--bg-card2);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 10px 16px;
        margin: 6px 0;
        font-size: 13px;
    }
    .p2p-sell { color: var(--accent-green); font-family:'Orbitron',sans-serif; font-size:11px; }
    .p2p-rev  { color: var(--accent-amber); font-weight: bold; }

    /* ── Status indicator ── */
    .status-dot {
        display: inline-block;
        width: 10px; height: 10px;
        border-radius: 50%;
        background: var(--accent-green);
        box-shadow: 0 0 8px var(--accent-green);
        animation: pulse 2s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0%,100% { opacity:1; } 50% { opacity:0.4; }
    }

    /* ── Section separator ── */
    .section-sep {
        border-top: 1px solid var(--border);
        margin: 18px 0;
        position: relative;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed var(--border) !important;
        border-radius: 8px !important;
        background: var(--bg-card) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ╔══════════════════════════════════════════════════════════════╗
# ║          ██████████ USER DATA HOOK █████████                ║
# ║  Replace the None values with your numpy arrays             ║
# ╚══════════════════════════════════════════════════════════════╝
# ─────────────────────────────────────────────────────────────────────────────

# ── PASTE YOUR ARRAYS HERE ──────────────────────────────────────────────────
# Shape: (N,) where N = number of hourly samples  (at least 24 recommended)

y_true_load  = None   # e.g. np.array([...])  – Actual   Load  (kW)
y_pred_load  = None   # e.g. np.array([...])  – Predicted Load  (kW)

y_true_solar = None   # e.g. np.array([...])  – Actual   Solar (kW)
y_pred_solar = None   # e.g. np.array([...])  – Predicted Solar (kW)

y_true_wind  = None   # e.g. np.array([...])  – Actual   Wind  (kW)
y_pred_wind  = None   # e.g. np.array([...])  – Predicted Wind  (kW)

# ── CARBON / PRICING CONSTANTS ───────────────────────────────────────────────
CO2_KG_PER_KWH  = 0.475   # kg CO₂ avoided per kWh of solar used (grid avg)
P2P_PRICE_USD   = 0.08    # $/kWh for peer-to-peer energy trading


# ─────────────────────────────────────────────────────────────────────────────
# DUMMY DATA GENERATOR  (24-hour synthetic profile)
# ─────────────────────────────────────────────────────────────────────────────
def generate_dummy_data():
    np.random.seed(42)
    hours = np.arange(24)

    # Load: morning/evening peaks
    load_base = 55 + 30 * np.sin(np.pi * (hours - 6) / 12) + \
                15 * np.sin(np.pi * (hours - 17) / 5)
    load_base = np.clip(load_base, 20, 120)

    # Solar: bell curve peaking at noon
    solar_base = np.maximum(0, 80 * np.exp(-0.5 * ((hours - 12) / 3.5) ** 2))

    # Wind: random-ish with mild trend
    wind_base = 30 + 20 * np.cos(np.pi * hours / 12) + \
                10 * np.sin(np.pi * hours / 6)
    wind_base = np.clip(wind_base, 5, 80)

    noise = lambda arr, pct: arr + np.random.normal(0, pct * arr.mean(), 24)

    return {
        "y_true_load" : noise(load_base, 0.05),
        "y_pred_load" : noise(load_base, 0.04),
        "y_true_solar": solar_base,
        "y_pred_solar": noise(solar_base + 2, 0.06),
        "y_true_wind" : wind_base,
        "y_pred_wind" : noise(wind_base, 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<p style="font-family:Orbitron,sans-serif;font-size:14px;'
        'color:#00ff88;letter-spacing:3px;">⚡ V.E.R.A.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        '<p style="font-size:11px;color:#4a6a7a;letter-spacing:2px;">UPLOAD MODEL OUTPUT</p>',
        unsafe_allow_html=True,
    )
    st.info(
        "Drop CSV files with columns:\n"
        "`y_true` and `y_pred`\n\n"
        "One file per energy type.",
        icon="📂",
    )

    load_file  = st.file_uploader("Load CSV",  type="csv", key="load_up")
    solar_file = st.file_uploader("Solar CSV", type="csv", key="solar_up")
    wind_file  = st.file_uploader("Wind CSV",  type="csv", key="wind_up")

    st.markdown("---")
    st.markdown(
        '<p style="font-size:11px;color:#4a6a7a;letter-spacing:2px;">CONSTANTS</p>',
        unsafe_allow_html=True,
    )
    co2_factor = st.slider("CO₂ factor (kg/kWh)", 0.2, 1.0, CO2_KG_PER_KWH, 0.005)
    p2p_price  = st.slider("P2P Price ($/kWh)",   0.01, 0.30, P2P_PRICE_USD, 0.005)

    st.markdown("---")
    st.markdown(
        '<p style="font-size:10px;color:#2a4050;">V.E.R.A. v2.6.1 | Grid-Edge OS</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (uploaded > hardcoded > dummy)
# ─────────────────────────────────────────────────────────────────────────────
def _load_col(file_obj, true_arr, pred_arr, col_true="y_true", col_pred="y_pred"):
    """Return (true_array, pred_array) from upload > code hook > dummy."""
    if file_obj is not None:
        df = pd.read_csv(file_obj)
        return df[col_true].values, df[col_pred].values
    if true_arr is not None and pred_arr is not None:
        return np.asarray(true_arr), np.asarray(pred_arr)
    return None, None


tl, pl  = _load_col(load_file,  y_true_load,  y_pred_load)
ts, ps  = _load_col(solar_file, y_true_solar, y_pred_solar)
tw, pw  = _load_col(wind_file,  y_true_wind,  y_pred_wind)

using_dummy = tl is None or ts is None or tw is None
if using_dummy:
    dummy = generate_dummy_data()
    tl  = dummy["y_true_load"];  pl  = dummy["y_pred_load"]
    ts  = dummy["y_true_solar"]; ps  = dummy["y_pred_solar"]
    tw  = dummy["y_true_wind"];  pw  = dummy["y_pred_wind"]

hours = list(range(len(tl)))
n = len(tl)

# Aligned arrays
pl  = pl[:n];  ts  = ts[:n];  ps  = ps[:n]
tw  = tw[:n];  pw  = pw[:n]


# ─────────────────────────────────────────────────────────────────────────────
# COMPUTED METRICS
# ─────────────────────────────────────────────────────────────────────────────
# Latest hour snapshot
cur_pred_load  = pl[-1]
cur_pred_solar = ps[-1]
cur_pred_wind  = pw[-1]
cur_true_load  = tl[-1]

generation     = cur_pred_solar + cur_pred_wind
surplus        = generation - cur_pred_load          # + = surplus, - = deficit

# USP-1: Carbon
solar_util     = np.mean(ps) / max(np.mean(ts), 1e-6)  # fraction utilized
solar_util     = min(solar_util, 1.0)
co2_saved_kg   = float(np.sum(ps)) * co2_factor          # total over horizon

# USP-2: Survival horizon
diff_series    = np.clip(pw + ps - pl, 0, None)          # hours > 0 = powered
# heuristic: avg surplus * 1 h / avg load -> hours of autonomy
avg_surplus    = float(np.mean(np.maximum(ps + pw - pl, 0)))
avg_load       = float(np.mean(pl))
# battery proxy: assume 2h of buffer storage
BATTERY_KWH    = avg_load * 2
survival_hours = (BATTERY_KWH + avg_surplus * n) / max(avg_load, 1e-3)
survival_hours = min(survival_hours, 9999)

surv_h = int(survival_hours)
surv_m = int((survival_hours - surv_h) * 60)

# USP-3: P2P market moments
p2p_events = []
for i in range(n):
    excess = (ps[i] + pw[i]) - pl[i]
    if excess > 0:
        revenue = excess * p2p_price
        p2p_events.append({
            "Hour"   : f"{i:02d}:00",
            "Excess (kW)" : round(excess, 2),
            "Buyer"  : "Neighbor B",
            "Offer"  : "SELL",
            "Revenue ($)" : f"{revenue:.2e}",
        })

total_p2p_rev = sum((ps[i] + pw[i] - pl[i]) * p2p_price
                    for i in range(n) if (ps[i] + pw[i]) > pl[i])


# ─────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━  HEADER  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="display:flex;align-items:center;justify-content:space-between;
                padding:16px 0 8px 0;border-bottom:1px solid #1a3040;margin-bottom:20px;">
        <div>
            <span style="font-family:Orbitron,sans-serif;font-size:26px;
                         color:#00ff88;letter-spacing:4px;font-weight:900;">
                ⚡ V.E.R.A.
            </span>
            <span style="font-family:Orbitron,sans-serif;font-size:14px;
                         color:#4a6a7a;letter-spacing:3px;margin-left:12px;">
                | Grid-Edge Intelligence
            </span>
        </div>
        <div style="display:flex;align-items:center;gap:8px;">
            <span class="status-dot"></span>
            <span style="font-family:Orbitron,sans-serif;font-size:11px;
                         color:#00ff88;letter-spacing:2px;">
                SYSTEM: ISLAND MODE READY
            </span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if using_dummy:
    st.caption("⚠️  Running on **24-hour synthetic data**. Upload CSVs in the sidebar or paste arrays into the code to use your model output.")

# ─────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━━━━━━━━━━━━━  TOP KPI STRIP  ━━━━━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

grid_state = "SURPLUS" if surplus >= 0 else "DEFICIT"
grid_color = "#00ff88" if surplus >= 0 else "#ff3d3d"
grid_delta = f"{'+' if surplus>=0 else ''}{surplus:.1f} kW vs Load"

k1.metric("Pred. Load  (current hr)", f"{cur_pred_load:.1f} kW",
          f"True: {cur_true_load:.1f} kW")
k2.metric("Pred. Solar (current hr)", f"{cur_pred_solar:.1f} kW")
k3.metric("Pred. Wind  (current hr)", f"{cur_pred_wind:.1f} kW")
k4.metric("Grid Balance", f"{grid_state}",
          delta=grid_delta,
          delta_color="normal" if surplus >= 0 else "inverse")

st.markdown("<div class='section-sep'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━  USP 1 ─ Carbon-Aware Dispatch  ━━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="font-family:Orbitron,sans-serif;font-size:13px;'
    'color:#00e5ff;letter-spacing:3px;">◈ USP 1 — CARBON-AWARE DISPATCH</p>',
    unsafe_allow_html=True,
)

c1a, c1b, c1c = st.columns([2, 2, 2])

with c1a:
    # Carbon intensity gauge: lower solar → higher intensity
    carbon_intensity = max(0, 100 - int(solar_util * 100))  # 0=clean, 100=dirty
    if carbon_intensity < 35:
        gauge_color = "#00ff88"
        ci_label = "LOW CARBON MODE: ACTIVE"
        ci_tag   = "tag-green"
    elif carbon_intensity < 65:
        gauge_color = "#ffb300"
        ci_label = "MIXED GRID: MODERATE"
        ci_tag   = "tag-amber"
    else:
        gauge_color = "#ff3d3d"
        ci_label = "HIGH CARBON MODE: ALERT"
        ci_tag   = "tag-red"

    st.markdown(
        f"""
        <div class="vera-card">
            <div class="vera-card-title">LIVE CARBON INTENSITY</div>
            <div style="font-family:Orbitron,sans-serif;font-size:36px;
                        color:{gauge_color};text-shadow:0 0 14px {gauge_color};">
                {carbon_intensity}<span style="font-size:16px;">%</span>
            </div>
            <div class="gauge-bar-bg">
                <div class="gauge-bar-fill"
                     style="width:{carbon_intensity}%;
                            background:linear-gradient(90deg,#00ff88,{gauge_color});">
                </div>
            </div>
            <span class="vera-tag {ci_tag}">{ci_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c1b:
    st.markdown(
        f"""
        <div class="vera-card">
            <div class="vera-card-title">EST. CO₂ SAVED (horizon)</div>
            <div class="vera-val-green" style="text-shadow:0 0 14px #00ff88;">
                {co2_saved_kg:.1f}
                <span style="font-size:16px;">kg</span>
            </div>
            <div style="font-size:11px;color:#4a6a7a;margin-top:8px;">
                ≈ {co2_saved_kg/21:.1f} trees/day offset
            </div>
            <div style="font-size:11px;color:#4a6a7a;">
                Solar utilization :
                <span style="color:#00ff88;">{solar_util*100:.1f}%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c1c:
    # Renewable share doughnut (simple bar)
    ren_share = min(generation / max(cur_pred_load, 1) * 100, 100)
    ren_color = "#00ff88" if ren_share > 80 else ("#ffb300" if ren_share > 50 else "#ff3d3d")
    st.markdown(
        f"""
        <div class="vera-card">
            <div class="vera-card-title">RENEWABLE SHARE (current hr)</div>
            <div style="font-family:Orbitron,sans-serif;font-size:36px;
                        color:{ren_color};text-shadow:0 0 14px {ren_color};">
                {ren_share:.0f}<span style="font-size:16px;">%</span>
            </div>
            <div class="gauge-bar-bg">
                <div class="gauge-bar-fill"
                     style="width:{ren_share:.0f}%;
                            background:linear-gradient(90deg,#00aaff,{ren_color});">
                </div>
            </div>
            <div style="font-size:11px;color:#4a6a7a;margin-top:6px;">
                Solar {cur_pred_solar:.1f} kW + Wind {cur_pred_wind:.1f} kW
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-sep'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━  USP 2 ─ Island-Mode Resiliency  ━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="font-family:Orbitron,sans-serif;font-size:13px;'
    'color:#00e5ff;letter-spacing:3px;">◈ USP 2 — ISLAND-MODE RESILIENCY</p>',
    unsafe_allow_html=True,
)

c2a, c2b = st.columns([1.5, 2.5])

with c2a:
    if surv_h > 48:
        clock_color = "#00ff88"
        surv_status = "EXTENDED AUTONOMY"
        surv_tag    = "tag-green"
    elif surv_h > 12:
        clock_color = "#ffb300"
        surv_status = "NOMINAL ISLAND MODE"
        surv_tag    = "tag-amber"
    else:
        clock_color = "#ff3d3d"
        surv_status = "CRITICAL — GRID NEEDED"
        surv_tag    = "tag-red"

    st.markdown(
        f"""
        <div class="vera-card" style="text-align:center;">
            <div class="vera-card-title" style="text-align:center;">SURVIVAL HORIZON</div>
            <div class="clock-display" style="color:{clock_color};
                 text-shadow:0 0 24px {clock_color};">
                {surv_h:04d}<span style="font-size:24px;opacity:.6;">h</span>
                {surv_m:02d}<span style="font-size:24px;opacity:.6;">m</span>
            </div>
            <div class="clock-label">ESTIMATED ISLAND RUNTIME</div>
            <br/>
            <span class="vera-tag {surv_tag}">{surv_status}</span>
            <div style="font-size:11px;color:#4a6a7a;margin-top:10px;">
                Avg surplus feed-in: {avg_surplus:.1f} kW<br/>
                Battery reserve: {BATTERY_KWH:.0f} kWh
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2b:
    # Hourly power balance chart
    balance_df = pd.DataFrame(
        {
            "Hour"       : hours,
            "Load (kW)"  : pl,
            "Gen  (kW)"  : ps + pw,
        }
    ).set_index("Hour")

    st.markdown(
        '<p style="font-size:11px;color:#4a6a7a;letter-spacing:2px;margin-bottom:4px;">'
        'POWER BALANCE: GENERATION vs LOAD</p>',
        unsafe_allow_html=True,
    )
    st.area_chart(
        balance_df,
        color=["#00aaff", "#00ff88"],
        height=200,
        use_container_width=True,
    )

    # Mini surplus histogram
    surplus_series = pd.Series(ps + pw - pl, name="Net Balance (kW)")
    st.markdown(
        '<p style="font-size:11px;color:#4a6a7a;letter-spacing:2px;margin-top:4px;">'
        'HOURLY NET BALANCE (+ = island surplus)</p>',
        unsafe_allow_html=True,
    )
    st.bar_chart(surplus_series, height=100, color="#ffb300")

st.markdown("<div class='section-sep'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━  USP 3 ─ P2P Local Energy Market  ━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="font-family:Orbitron,sans-serif;font-size:13px;'
    'color:#00e5ff;letter-spacing:3px;">◈ USP 3 — P2P NEIGHBORHOOD ENERGY MARKET</p>',
    unsafe_allow_html=True,
)

c3a, c3b = st.columns([2.5, 1.5])

with c3a:
    if p2p_events:
        p2p_df = pd.DataFrame(p2p_events)
        st.dataframe(
            p2p_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hour"        : st.column_config.TextColumn("🕐 HOUR"),
                "Excess (kW)" : st.column_config.NumberColumn("EXCESS ↑ kW",  format="%.2f kW"),
                "Buyer"       : st.column_config.TextColumn("BUYER"),
                "Offer"       : st.column_config.TextColumn("ACTION"),
                "Revenue ($)" : st.column_config.TextColumn("REVENUE ($)"),
            },
            height=250,
        )
    else:
        st.markdown(
            '<div class="vera-card"><div class="vera-card-title">NO SURPLUS HOURS</div>'
            '<p style="color:#4a6a7a;">Generation ≤ Load in all hours. No P2P trades available.</p></div>',
            unsafe_allow_html=True,
        )

with c3b:
    rev_exp = f"{total_p2p_rev:.2e}"
    surplus_count = len(p2p_events)
    deficit_count = n - surplus_count

    st.markdown(
        f"""
        <div class="vera-card">
            <div class="vera-card-title">POTENTIAL P2P REVENUE</div>
            <div style="font-family:Orbitron,sans-serif;font-size:30px;
                        color:#ffb300;text-shadow:0 0 12px #ffb300;">
                ${rev_exp}
            </div>
            <div style="font-size:11px;color:#4a6a7a;margin-top:10px;">
                Sell windows : <span style="color:#00ff88;">{surplus_count}h</span><br/>
                Idle windows : <span style="color:#ff3d3d;">{deficit_count}h</span><br/>
                Rate         : <span style="color:#00aaff;">${p2p_price:.3f}/kWh</span>
            </div>
        </div>
        <div class="vera-card" style="margin-top:10px;">
            <div class="vera-card-title">ACTIVE TRADE OFFER</div>
            <div style="font-size:12px;">
                <span style="color:#4a6a7a;">FROM:</span>
                <span style="color:#00ff88;"> NODE_VERA_01</span><br/>
                <span style="color:#4a6a7a;">TO  :</span>
                <span style="color:#00aaff;"> NEIGHBOR_B</span><br/>
                <span style="color:#4a6a7a;">TYPE:</span>
                <span style="color:#ffb300;"> SELL OFFER (LIVE)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-sep'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ━━━━━━━━  GRID PULSE OVERLAY CHARTS  ━━━━━━━━━━━━━━━━━━━━━━━━━
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p style="font-family:Orbitron,sans-serif;font-size:13px;'
    'color:#00e5ff;letter-spacing:3px;">◈ ACTUAL GRID PULSE vs V.E.R.A. PREDICTION</p>',
    unsafe_allow_html=True,
)

tabs = st.tabs(["⚡ LOAD", "☀️ SOLAR", "💨 WIND"])

for tab, true_arr, pred_arr, label, unit in zip(
    tabs,
    [tl, ts, tw],
    [pl, ps, pw],
    ["Load", "Solar", "Wind"],
    ["kW", "kW", "kW"],
):
    with tab:
        chart_df = pd.DataFrame(
            {
                f"Actual Grid Pulse ({label})"  : true_arr,
                f"V.E.R.A. Prediction ({label})": pred_arr,
            },
            index=hours,
        )
        chart_df.index.name = "Hour"
        # Blue = Actual, Neon Green = Prediction
        st.line_chart(
            chart_df,
            color=["#00aaff", "#39ff14"],
            height=260,
            use_container_width=True,
        )

        mae  = float(np.mean(np.abs(true_arr - pred_arr)))
        rmse = float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))
        mape = float(np.mean(np.abs((true_arr - pred_arr) / np.maximum(true_arr, 1e-6)))) * 100

        m1, m2, m3 = st.columns(3)
        m1.metric(f"MAE ({label})",  f"{mae:.2f} {unit}")
        m2.metric(f"RMSE ({label})", f"{rmse:.2f} {unit}")
        m3.metric(f"MAPE ({label})", f"{mape:.2f} %")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;padding:24px 0 8px 0;
                border-top:1px solid #1a3040;margin-top:20px;">
        <span style="font-family:Orbitron,sans-serif;font-size:10px;
                     color:#2a4050;letter-spacing:3px;">
            V.E.R.A. v2.6.1 · PROJECT AMDSHACK · GRID-EDGE INTELLIGENCE PLATFORM
            · ALL METRICS PREDICTIVE — NOT FINANCIAL ADVICE
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

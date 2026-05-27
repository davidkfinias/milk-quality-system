"""
================================================================================
MILK QUALITY MONITORING SYSTEM — Streamlit Prototype
================================================================================
Dissertation demo application.
Uses the trained K-Means model from the clustering pipeline to classify
new sensor readings in real time.

Run:  streamlit run app.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
import os

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Milk Quality Monitor",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global */
    .stApp { font-family: 'Plus Jakarta Sans', sans-serif; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.7; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }
    .metric-card h1 { margin: 8px 0 0 0; font-size: 32px; font-weight: 800; }

    /* Quality badge */
    .quality-badge {
        display: inline-block;
        padding: 12px 32px;
        border-radius: 50px;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .badge-fresh        { background: linear-gradient(135deg, #27ae60, #2ecc71); }
    .badge-semi-spoiled { background: linear-gradient(135deg, #e67e22, #f39c12); }
    .badge-spoiled      { background: linear-gradient(135deg, #c0392b, #e74c3c); }

    /* Threshold table */
    .thresh-table { width: 100%; border-collapse: collapse; font-size: 14px; }
    .thresh-table th { background: #1a1a2e; color: white; padding: 12px 16px; text-align: center; }
    .thresh-table td { padding: 10px 16px; text-align: center; border-bottom: 1px solid #eee; }
    .thresh-table tr:nth-child(even) { background: #f8f9fa; }
    .fresh-cell { background: rgba(39,174,96,0.12) !important; color: #1e8449; font-weight: 600; }
    .semi-cell  { background: rgba(243,156,18,0.12) !important; color: #b7950b; font-weight: 600; }
    .spoiled-cell { background: rgba(231,76,60,0.12) !important; color: #c0392b; font-weight: 600; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 32px 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .main-header h1 { font-size: 28px; font-weight: 800; margin: 0; }
    .main-header p  { font-size: 14px; opacity: 0.7; margin: 4px 0 0 0; }

    /* Section headers */
    .section-header {
        font-size: 18px;
        font-weight: 700;
        color: #1a1a2e;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #e0e0e0;
    }

    div[data-testid="stSidebar"] { background: #f8f9fa; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ────────────────────────────────────────────────────────────

MODEL_DIR = "./output"  # Same dir as pipeline output

CLUSTER_FEATURES = ['ph_actual', 'gas_raw_mq135']
FEAT_LABELS = {
    'ph_actual': 'pH Level',
    'gas_raw_mq135': 'Gas MQ-135 (ppm)',
    'temp_c_dht': 'Temperature (°C)',
}
CLASS_ORDER = ['Fresh', 'Semi-Spoiled', 'Spoiled']
COLORS_HEX = {'Fresh': '#27ae60', 'Semi-Spoiled': '#f39c12', 'Spoiled': '#e74c3c'}


@st.cache_resource
def load_model():
    """Load trained model artifacts."""
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    model_path = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
    artifacts_path = os.path.join(MODEL_DIR, 'model_artifacts.json')

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(model_path, 'rb') as f:
            kmeans = pickle.load(f)
        artifacts = {}
        if os.path.exists(artifacts_path):
            with open(artifacts_path) as f:
                artifacts = json.load(f)
        return scaler, kmeans, artifacts
    else:
        return None, None, None


def classify_sample(scaler, kmeans, ph, gas):
    """Classify a single sample and return class + confidence."""
    X = np.array([[ph, gas]])
    X_scaled = scaler.transform(X)
    cluster = kmeans.predict(X_scaled)[0]

    # Map cluster to class name by pH centroid ordering (highest pH = Fresh)
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    order = np.argsort(-centroids[:, 0])  # pH descending: highest=Fresh
    class_map = {order[0]: 'Fresh', order[1]: 'Semi-Spoiled', order[2]: 'Spoiled'}
    label = class_map[cluster]

    # Confidence: 1 - normalized distance to assigned centroid
    dists = np.linalg.norm(X_scaled - kmeans.cluster_centers_, axis=1)
    min_dist = dists.min()
    max_possible = dists.max()
    confidence = 1 - (min_dist / max_possible) if max_possible > 0 else 1.0

    return label, confidence, X_scaled[0]


def classify_batch(scaler, kmeans, df):
    """Classify a batch DataFrame."""
    X = df[CLUSTER_FEATURES].values
    X_scaled = scaler.transform(X)
    clusters = kmeans.predict(X_scaled)

    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    order = np.argsort(-centroids[:, 0])  # pH descending: highest=Fresh
    class_map = {order[0]: 'Fresh', order[1]: 'Semi-Spoiled', order[2]: 'Spoiled'}

    labels = [class_map[c] for c in clusters]
    dists = np.linalg.norm(X_scaled[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2)
    min_dists = dists.min(axis=1)
    confidence = 1 - min_dists / min_dists.max()

    df_out = df.copy()
    df_out['Milk_Quality'] = labels
    df_out['Confidence'] = np.round(confidence, 4)
    return df_out


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🥛 Navigation")
    page = st.radio("", [
        "🔬 Single Sample Test",
        "📊 Batch Classification",
        "📉 Degradation Timeline",
        "📈 Model Performance",
        "📋 Reference Thresholds"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    **IoT Milk Freshness System**

    This prototype classifies milk quality
    using sensor data from:
    - ⚗️ pH Sensor (primary)
    - 💨 MQ-135 Gas Sensor (primary)
    - 🌡️ DHT22 Temp & Humidity (logged)

    Classification: pH + Gas via K-Means
    """)

    st.markdown("---")
    st.caption("Final Year Dissertation Prototype")


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🥛 Milk Quality Classification System</h1>
    <p>IoT-Based Freshness Detection Using Unsupervised Clustering</p>
</div>
""", unsafe_allow_html=True)


# ── Load Model ───────────────────────────────────────────────────────────────

scaler, kmeans, artifacts = load_model()

if scaler is None:
    st.warning("⚠️ Model files not found. Please run the clustering pipeline first to generate `scaler.pkl` and `kmeans_model.pkl` in the `./output` directory.")
    st.code("python milk_clustering_pipeline.py", language="bash")
    st.stop()


# ── Page: Single Sample Test ────────────────────────────────────────────────

if page == "🔬 Single Sample Test":
    st.markdown('<div class="section-header">Real-Time Single Sample Classification</div>', unsafe_allow_html=True)
    st.markdown("Simulate a live sensor reading by adjusting the sliders below. Classification uses **pH** and **Gas** readings.")

    col1, col2 = st.columns(2)
    with col1:
        ph_val = st.slider("⚗️ pH Level", min_value=4.0, max_value=7.5, value=6.6, step=0.01)
    with col2:
        gas_val = st.slider("💨 Gas MQ-135 (ppm)", min_value=100, max_value=1400, value=250, step=5)

    label, confidence, x_scaled = classify_sample(scaler, kmeans, ph_val, gas_val)

    st.markdown("---")

    # Result display
    c1, c2, c3 = st.columns([1.5, 1, 1])
    with c1:
        if label == "Fresh":
            badge_class = "badge-fresh"
        elif label == "Semi-Spoiled":
            badge_class = "badge-semi-spoiled"
        else:
            badge_class = "badge-spoiled"
        st.markdown(f"""
        <div style="text-align:center; padding: 20px;">
            <p style="font-size:13px; color:#666; margin-bottom:8px; text-transform:uppercase; letter-spacing:1px;">Classification Result</p>
            <span class="quality-badge {badge_class}">{label}</span>
            <p style="margin-top:12px; font-size:14px; color:#666;">Confidence: <strong>{confidence:.1%}</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>pH Level</h3>
            <h1>{ph_val:.2f}</h1>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Gas (ppm)</h3>
            <h1>{gas_val}</h1>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Gauge charts
    st.markdown('<div class="section-header">Sensor Gauge Readings</div>', unsafe_allow_html=True)
    gc1, gc2 = st.columns(2)

    def make_gauge(value, title, min_val, max_val, thresholds, colors_list):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'size': 16}},
            gauge={
                'axis': {'range': [min_val, max_val], 'tickwidth': 1},
                'bar': {'color': "#1a1a2e"},
                'steps': [{'range': [t[0], t[1]], 'color': c} for t, c in zip(thresholds, colors_list)],
                'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.8, 'value': value}
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=10))
        return fig

    with gc1:
        st.plotly_chart(make_gauge(ph_val, "pH Level", 4, 7.5,
                                   [(4, 6), (6, 6.5), (6.5, 7.5)],
                                   ["rgba(231,76,60,0.3)", "rgba(243,156,18,0.3)", "rgba(39,174,96,0.3)"]),
                        use_container_width=True)
    with gc2:
        st.plotly_chart(make_gauge(gas_val, "Gas MQ-135 (ppm)", 100, 1400,
                                   [(100, 300), (300, 350), (350, 1400)],
                                   ["rgba(39,174,96,0.3)", "rgba(243,156,18,0.3)", "rgba(231,76,60,0.3)"]),
                        use_container_width=True)


# ── Page: Batch Classification ──────────────────────────────────────────────

elif page == "📊 Batch Classification":
    st.markdown('<div class="section-header">Batch File Classification</div>', unsafe_allow_html=True)
    st.markdown("Upload a CSV with sensor data to classify all rows at once.")

    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"**Loaded:** {len(df_up):,} rows × {len(df_up.columns)} columns")
        st.dataframe(df_up.head(10), use_container_width=True)

        # Check required columns
        missing = [c for c in CLUSTER_FEATURES if c not in df_up.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.info(f"Expected columns: {CLUSTER_FEATURES}")
        else:
            if st.button("🚀 Classify All Rows", type="primary"):
                with st.spinner(f"Classifying {len(df_up):,} samples..."):
                    df_result = classify_batch(scaler, kmeans, df_up)
                # Persist so the Timeline page can reuse it without re-uploading
                st.session_state['labeled_data'] = df_result
                st.success(f"✅ Classified {len(df_result):,} samples! The Degradation Timeline page will now load these results automatically.")

            # Show results if available in session (freshly classified or from before)
            if 'labeled_data' in st.session_state:
                df_result = st.session_state['labeled_data']

                # Summary
                dist = df_result['Milk_Quality'].value_counts()
                c1, c2, c3 = st.columns(3)
                for col_st, cls in zip([c1, c2, c3], CLASS_ORDER):
                    count = dist.get(cls, 0)
                    pct = count / len(df_result) * 100
                    with col_st:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{cls}</h3>
                            <h1 style="color:{COLORS_HEX[cls]}">{count:,}</h1>
                            <p style="opacity:0.6; margin:4px 0 0 0;">{pct:.1f}%</p>
                        </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # Distribution chart
                fig = px.histogram(df_result, x='Milk_Quality', color='Milk_Quality',
                                   color_discrete_map=COLORS_HEX,
                                   category_orders={'Milk_Quality': CLASS_ORDER},
                                   title='Classification Distribution')
                fig.update_layout(showlegend=False, height=350)
                st.plotly_chart(fig, use_container_width=True)

                # Scatter plot — pH vs Gas colored by quality
                fig = px.scatter(df_result.sample(min(3000, len(df_result)), random_state=42),
                                    x='ph_actual', y='gas_raw_mq135',
                                    color='Milk_Quality', color_discrete_map=COLORS_HEX,
                                    category_orders={'Milk_Quality': CLASS_ORDER},
                                    opacity=0.6, title='pH vs Gas by Quality Class',
                                    labels={'ph_actual': 'pH Level', 'gas_raw_mq135': 'Gas MQ-135 (ppm)'})
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

                # Preview + download
                st.markdown('<div class="section-header">Labeled Data Preview</div>', unsafe_allow_html=True)
                st.dataframe(df_result.head(20), use_container_width=True)

                csv = df_result.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Labeled CSV", csv,
                                   "milk_quality_labeled.csv", "text/csv",
                                   type="primary")
    else:
        st.info("Upload a CSV file with columns: `ph_actual`, `gas_raw_mq135` (other columns are preserved but not used for classification)")

        # Demo with random data
        if st.button("🎲 Generate Demo Data (1000 samples)"):
            demo = pd.DataFrame({
                'ts_ms': np.arange(0, 1000000, 1000),
                'ph_raw': np.random.randint(5000, 7000, 1000),
                'ph_actual': np.round(np.random.uniform(4.5, 6.9, 1000), 4),
                'gas_raw_mq135': np.random.randint(180, 1300, 1000),
                'temp_c_dht': np.round(np.random.uniform(23, 37, 1000), 4),
                'hum_pct_dht': np.round(np.random.uniform(51, 69, 1000), 4),
            })
            df_demo = classify_batch(scaler, kmeans, demo)
            st.dataframe(df_demo.head(20), use_container_width=True)

            dist = df_demo['Milk_Quality'].value_counts()
            fig = px.pie(values=dist.values, names=dist.index,
                         color=dist.index, color_discrete_map=COLORS_HEX,
                         title="Demo Classification Distribution")
            st.plotly_chart(fig, use_container_width=True)


# ── Page: Degradation Timeline ──────────────────────────────────────────────

elif page == "📉 Degradation Timeline":
    st.markdown('<div class="section-header">Milk Degradation Timeline</div>', unsafe_allow_html=True)

    # Source of data: session (from Batch page) takes priority; upload is optional override
    df_tl = None
    has_session = 'labeled_data' in st.session_state

    if has_session:
        st.success("✅ Using the dataset you classified on the Batch Classification page — no upload needed.")
        uploaded_tl = None
        with st.expander("Want to use a different file instead? Upload here"):
            uploaded_tl = st.file_uploader("Upload a labeled CSV", type=['csv'], key='timeline')
        if uploaded_tl is not None:
            df_tl = pd.read_csv(uploaded_tl)
        else:
            df_tl = st.session_state['labeled_data'].copy()
    else:
        st.markdown("No classified data yet. Either classify a file on the **Batch Classification** page (recommended — then this page loads automatically), or upload a labeled CSV below.")
        uploaded_tl = st.file_uploader("Upload labeled CSV (milk_quality_labeled.csv)", type=['csv'], key='timeline')
        if uploaded_tl is not None:
            df_tl = pd.read_csv(uploaded_tl)

    if df_tl is not None:
        required = ['ts_ms', 'ph_actual', 'gas_raw_mq135', 'Milk_Quality']
        missing = [c for c in required if c not in df_tl.columns]
        if missing:
            st.error(f"Missing columns: {missing}. The data needs a timestamp (ts_ms) and Milk_Quality label.")
            st.stop()

        df_tl = df_tl.sort_values('ts_ms').reset_index(drop=True)
        df_tl['time_hr'] = df_tl['ts_ms'] / 3_600_000

        total_hrs = df_tl['time_hr'].max()
        st.markdown(f"**Recording duration:** {total_hrs:.1f} hours ({total_hrs/24:.1f} days)")

        # Class breakdown
        c1, c2, c3 = st.columns(3)
        for col_st, cls in zip([c1, c2, c3], CLASS_ORDER):
            count = (df_tl['Milk_Quality'] == cls).sum()
            pct = count / len(df_tl) * 100
            with col_st:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{cls}</h3>
                    <h1 style="color:{COLORS_HEX[cls]}">{count:,}</h1>
                    <p style="opacity:0.6; margin:4px 0 0 0;">{pct:.1f}%</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Smooth data for plotting
        window = max(1, len(df_tl) // 2000)
        df_tl['ph_smooth'] = df_tl['ph_actual'].rolling(window, center=True, min_periods=1).mean()
        df_tl['gas_smooth'] = df_tl['gas_raw_mq135'].rolling(window, center=True, min_periods=1).mean()

        # Sample for Plotly performance
        sample_n = min(3000, len(df_tl))
        sample_idx = np.linspace(0, len(df_tl)-1, sample_n, dtype=int)
        dp = df_tl.iloc[sample_idx].copy()

        # ── Compute per-class value ranges and time spans for region highlighting ──
        def cls_range(col, c):
            v = df_tl[df_tl['Milk_Quality'] == c][col]
            return (np.percentile(v, 2), np.percentile(v, 98)) if len(v) else (None, None)
        def cls_tspan(c):
            t = df_tl[df_tl['Milk_Quality'] == c]['time_hr']
            return (t.min(), t.max()) if len(t) else (None, None)

        # ── Chart 1: pH with highlighted class regions ──
        st.markdown('<div class="section-header">1. Quality Classification Over Time</div>', unsafe_allow_html=True)
        st.markdown("The shaded boxes highlight each of the 3 quality classes. The line is colored by the class assigned at that moment — watch the milk transition Fresh → Semi-Spoiled → Spoiled.")

        fig = go.Figure()
        # Shaded class-region rectangles
        for cls in CLASS_ORDER:
            lo, hi = cls_range('ph_actual', cls)
            t_lo, t_hi = cls_tspan(cls)
            if lo is not None:
                fig.add_shape(type="rect", x0=t_lo, x1=t_hi, y0=lo, y1=hi,
                              fillcolor=COLORS_HEX[cls], opacity=0.13,
                              line=dict(color=COLORS_HEX[cls], width=2), layer="below")
                fig.add_annotation(x=(t_lo+t_hi)/2, y=hi, text=f"<b>{cls}</b>",
                                   showarrow=False, font=dict(color=COLORS_HEX[cls], size=13),
                                   yshift=10)
        for cls in CLASS_ORDER:
            mask = dp['Milk_Quality'] == cls
            fig.add_trace(go.Scatter(
                x=dp[mask]['time_hr'], y=dp[mask]['ph_smooth'],
                mode='markers', name=cls,
                marker=dict(color=COLORS_HEX[cls], size=4, opacity=0.8),
                hovertemplate=f'{cls}<br>Time: %{{x:.2f}} hrs<br>pH: %{{y:.3f}}<extra></extra>'
            ))
        fig.update_layout(
            title='pH Level Over Time — Class Regions Highlighted',
            xaxis_title='Time (hours)', yaxis_title='pH Level',
            height=450, template='plotly_white', legend_title='Quality'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Chart 2: Gas with highlighted class regions ──
        fig = go.Figure()
        for cls in CLASS_ORDER:
            lo, hi = cls_range('gas_raw_mq135', cls)
            t_lo, t_hi = cls_tspan(cls)
            if lo is not None:
                fig.add_shape(type="rect", x0=t_lo, x1=t_hi, y0=lo, y1=hi,
                              fillcolor=COLORS_HEX[cls], opacity=0.13,
                              line=dict(color=COLORS_HEX[cls], width=2), layer="below")
                fig.add_annotation(x=(t_lo+t_hi)/2, y=hi, text=f"<b>{cls}</b>",
                                   showarrow=False, font=dict(color=COLORS_HEX[cls], size=13),
                                   yshift=10)
        for cls in CLASS_ORDER:
            mask = dp['Milk_Quality'] == cls
            fig.add_trace(go.Scatter(
                x=dp[mask]['time_hr'], y=dp[mask]['gas_smooth'],
                mode='markers', name=cls,
                marker=dict(color=COLORS_HEX[cls], size=4, opacity=0.8),
                hovertemplate=f'{cls}<br>Time: %{{x:.2f}} hrs<br>Gas: %{{y:.0f}} ppm<extra></extra>'
            ))
        fig.update_layout(
            title='Gas Sensor (MQ-135) Over Time — Class Regions Highlighted',
            xaxis_title='Time (hours)', yaxis_title='Gas (ppm)',
            height=450, template='plotly_white', legend_title='Quality'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Chart 3: Temperature & Humidity (context) ──
        if 'temp_c_dht' in df_tl.columns and 'hum_pct_dht' in df_tl.columns:
            st.markdown('<div class="section-header">2. Environmental Context (Temperature & Humidity)</div>', unsafe_allow_html=True)
            st.markdown("These sensors captured ambient conditions — shown here for context.")

            df_tl['temp_smooth'] = df_tl['temp_c_dht'].rolling(window, center=True, min_periods=1).mean()
            df_tl['hum_smooth'] = df_tl['hum_pct_dht'].rolling(window, center=True, min_periods=1).mean()
            dp_env = df_tl.iloc[sample_idx].copy()

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                subplot_titles=('Temperature (°C)', 'Humidity (%)'))
            for cls in CLASS_ORDER:
                mask = dp_env['Milk_Quality'] == cls
                fig.add_trace(go.Scatter(
                    x=dp_env[mask]['time_hr'], y=dp_env[mask]['temp_smooth'],
                    mode='markers', name=cls, legendgroup=cls,
                    marker=dict(color=COLORS_HEX[cls], size=3, opacity=0.6),
                    showlegend=True
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=dp_env[mask]['time_hr'], y=dp_env[mask]['hum_smooth'],
                    mode='markers', name=cls, legendgroup=cls,
                    marker=dict(color=COLORS_HEX[cls], size=3, opacity=0.6),
                    showlegend=False
                ), row=2, col=1)
            fig.update_layout(height=500, template='plotly_white', legend_title='Quality')
            fig.update_xaxes(title_text='Time (hours)', row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)

        # ── Chart 4: Stacked area — class proportion over time ──
        st.markdown('<div class="section-header">3. Quality Composition Over Time</div>', unsafe_allow_html=True)
        st.markdown("How the proportion of each class evolved throughout the recording.")

        n_bins = 80
        df_tl['time_bin'] = pd.cut(df_tl['time_hr'], bins=n_bins, labels=False)
        prop = df_tl.groupby('time_bin')['Milk_Quality'].value_counts(normalize=True).unstack(fill_value=0)
        for cls in CLASS_ORDER:
            if cls not in prop.columns:
                prop[cls] = 0
        prop = prop[CLASS_ORDER]
        bin_centers = df_tl.groupby('time_bin')['time_hr'].mean()

        fig = go.Figure()
        def hex_to_rgba(hex_color, alpha=0.6):
            h = hex_color.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return f'rgba({r},{g},{b},{alpha})'
        for cls in CLASS_ORDER:
            fig.add_trace(go.Scatter(
                x=bin_centers.values, y=prop[cls].values,
                mode='lines', name=cls, stackgroup='one',
                line=dict(width=0.5, color=COLORS_HEX[cls]),
                fillcolor=hex_to_rgba(COLORS_HEX[cls], 0.6),
            ))
        fig.update_layout(
            title='Stacked Quality Proportion Over Time',
            xaxis_title='Time (hours)', yaxis_title='Proportion',
            yaxis=dict(range=[0, 1], tickformat='.0%'),
            height=400, template='plotly_white', legend_title='Quality'
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Phase summary ──
        st.markdown('<div class="section-header">4. Phase Summary (Early / Middle / Late)</div>', unsafe_allow_html=True)

        df_tl['phase'] = pd.cut(df_tl['time_hr'], bins=3, labels=['Early', 'Middle', 'Late'])
        phase_data = []
        for phase in ['Early', 'Middle', 'Late']:
            subset = df_tl[df_tl['phase'] == phase]
            phase_data.append({
                'Phase': phase,
                'Avg pH': f"{subset['ph_actual'].mean():.3f}",
                'Avg Gas (ppm)': f"{subset['gas_raw_mq135'].mean():.0f}",
                'Avg Temp (°C)': f"{subset['temp_c_dht'].mean():.1f}" if 'temp_c_dht' in subset.columns else '—',
                '% Fresh': f"{(subset['Milk_Quality']=='Fresh').mean()*100:.1f}%",
                '% Semi-Spoiled': f"{(subset['Milk_Quality']=='Semi-Spoiled').mean()*100:.1f}%",
                '% Spoiled': f"{(subset['Milk_Quality']=='Spoiled').mean()*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(phase_data), use_container_width=True, hide_index=True)

    else:
        st.info("👆 Classify a file on the **Batch Classification** page first — then come back here and the timeline loads automatically. Or upload a labeled CSV above.")
        st.markdown("""
        This page shows:
        - **pH & Gas timelines** with the 3 quality classes highlighted as colored regions
        - **Temperature & Humidity** environmental context
        - **Stacked area chart** showing how quality composition changed over time
        - **Phase summary** comparing Early/Middle/Late periods of the recording
        """)


# ── Page: Model Performance ─────────────────────────────────────────────────

elif page == "📈 Model Performance":
    st.markdown('<div class="section-header">Model Performance & Comparison</div>', unsafe_allow_html=True)

    if artifacts:
        met = artifacts.get('metrics', {})
        total = artifacts.get('total_samples', 0)
        agree = artifacts.get('agreement_with_rules', 0)
        dist = artifacts.get('class_distribution', {})

        # Key metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card"><h3>Total Samples</h3><h1>{total:,}</h1></div>""", unsafe_allow_html=True)
        with c2:
            km = met.get('kmeans', {})
            st.markdown(f"""<div class="metric-card"><h3>Silhouette Score</h3><h1>{km.get('silhouette', 0):.4f}</h1></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card"><h3>Rule Agreement</h3><h1>{agree:.1%}</h1></div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card"><h3>Clusters</h3><h1>3</h1></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Algorithm comparison table
        st.markdown('<div class="section-header">Algorithm Comparison</div>', unsafe_allow_html=True)
        comp_data = []
        for algo, key in [('K-Means (MiniBatch)', 'kmeans'), ('Agglomerative (Ward)', 'agglomerative'), ('DBSCAN', 'dbscan')]:
            m = met.get(key, {})
            comp_data.append({
                'Algorithm': algo,
                'Silhouette Score': f"{m.get('silhouette', 0):.4f}",
                'Calinski-Harabasz': f"{m.get('calinski_harabasz', '—')}",
                'Davies-Bouldin': f"{m.get('davies_bouldin', '—')}",
                'Noise Points': m.get('noise', 0),
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        # Class distribution
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        for cls in CLASS_ORDER:
            fig.add_trace(go.Bar(
                x=[cls], y=[dist.get(cls, 0)],
                marker_color=COLORS_HEX[cls], name=cls,
                text=[f"{dist.get(cls, 0):,}"], textposition='outside'
            ))
        fig.update_layout(showlegend=False, height=350, yaxis_title="Count",
                          title="Samples per Quality Class")
        st.plotly_chart(fig, use_container_width=True)

        # Centroid details
        st.markdown('<div class="section-header">K-Means Cluster Centroids</div>', unsafe_allow_html=True)
        if 'centroids_original' in artifacts and 'cluster_order' in artifacts:
            cents = np.array(artifacts['centroids_original'])
            order = artifacts['cluster_order']
            cent_df = pd.DataFrame({
                'Quality Class': CLASS_ORDER,
                'pH Level': [f"{cents[order[i], 0]:.3f}" for i in range(3)],
                'Gas MQ-135 (ppm)': [f"{cents[order[i], 1]:.1f}" for i in range(3)],
            })
            st.dataframe(cent_df, use_container_width=True, hide_index=True)

        # Show saved plots if available
        st.markdown('<div class="section-header">Visualization Gallery</div>', unsafe_allow_html=True)
        if os.path.isdir(MODEL_DIR):
            plot_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith('.png')])
        else:
            plot_files = []
        if plot_files:
            cols_per_row = 2
            for i in range(0, len(plot_files), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(plot_files):
                        with col:
                            st.image(os.path.join(MODEL_DIR, plot_files[idx]),
                                     caption=plot_files[idx].replace('.png', '').replace('_', ' ').title(),
                                     use_container_width=True)
        else:
            st.info("No visualization PNGs found. Upload the plot images to the `output/` folder in your repo to see them here.")
    else:
        st.warning("Model artifacts not found. Run the clustering pipeline first.")


# ── Page: Reference Thresholds ──────────────────────────────────────────────

elif page == "📋 Reference Thresholds":
    st.markdown('<div class="section-header">Milk Quality Classification Thresholds</div>', unsafe_allow_html=True)
    st.markdown("These domain-knowledge thresholds define the quality classes that the clustering model learns to separate.")

    st.markdown("""
    <table class="thresh-table">
        <tr>
            <th>Parameter</th>
            <th>🟢 Fresh Milk</th>
            <th>🟡 Semi-Spoiled</th>
            <th>🔴 Spoiled Milk</th>
        </tr>
        <tr>
            <td><strong>Temperature</strong></td>
            <td class="fresh-cell">0 – 4°C<br><small>(ideal storage)</small></td>
            <td class="semi-cell">5 – 8°C<br><small>(rising risk)</small></td>
            <td class="spoiled-cell">&gt; 8°C<br><small>(high spoilage risk)</small></td>
        </tr>
        <tr>
            <td><strong>pH Level</strong></td>
            <td class="fresh-cell">6.5 – 6.8</td>
            <td class="semi-cell">6.1 – 6.4<br><small>(slightly acidic)</small></td>
            <td class="spoiled-cell">&lt; 6.0<br><small>(acidic = spoiled)</small></td>
        </tr>
        <tr>
            <td><strong>Gas Sensor (MQ-135)</strong></td>
            <td class="fresh-cell">200 – 300 ppm</td>
            <td class="semi-cell">300 – 350 ppm<br><small>(early gas formation)</small></td>
            <td class="spoiled-cell">&gt; 350–400 ppm<br><small>(spoilage gases)</small></td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">How the System Works</div>', unsafe_allow_html=True)

    st.markdown("""
    **1. Data Collection** — IoT sensors (pH, MQ-135, DHT22) capture readings from milk samples at regular intervals.

    **2. Preprocessing** — Raw readings are cleaned (null handling, outlier removal) and standardized using StandardScaler.

    **3. Feature Selection** — **pH** and **Gas (MQ-135)** are used as clustering features. Temperature captures ambient conditions in this dataset, so it is logged but not used for classification. All columns are preserved in the output for the prediction phase.

    **4. Unsupervised Clustering** — Three algorithms are compared:
    - **MiniBatch K-Means** (primary) — scales to 1M+ readings, consistent results
    - **Agglomerative Clustering** (Ward linkage) — evaluated on 10K sample for comparison
    - **DBSCAN** — density-based approach, evaluated on 15K sample

    **5. Label Assignment** — Clusters are mapped to quality classes (Fresh / Semi-Spoiled / Spoiled) by ordering centroids by pH level (highest pH = freshest milk).

    **6. Validation** — Cluster labels are compared against rule-based thresholds to measure agreement.

    **7. Handoff** — The labeled dataset is exported for supervised prediction model training by the next team member.
    """)

    st.markdown("---")
    st.markdown('<div class="section-header">System Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    ```
    ┌─────────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
    │  IoT Sensors │───▶│  Raw Sensor   │───▶│  Clustering      │───▶│  Labeled     │
    │  pH, Temp,   │    │  Logs (CSV)   │    │  Pipeline        │    │  Dataset     │
    │  Gas, Hum    │    │  ~1M rows     │    │  (K-Means)       │    │  + Quality   │
    └─────────────┘    └──────────────┘    └──────────────────┘    └──────┬───────┘
                                                                          │
                                                                          ▼
                                           ┌──────────────────┐    ┌──────────────┐
                                           │  Streamlit       │◀───│  Prediction  │
                                           │  Prototype       │    │  Model       │
                                           │  (this app)      │    │  (next phase)│
                                           └──────────────────┘    └──────────────┘
    ```
    """)

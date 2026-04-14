import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Clarity & Dark Theme ---
st.markdown("""
<style>
    /* Hide Streamlit default footer and Deploy button */
    footer {visibility: hidden;}
    header {visibility: visible;}
    .stAppDeployButton {display: none;}

    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global background and typography */
    .stApp, *:not(i):not(.material-icons) {
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    }
    .stApp {
        background: linear-gradient(180deg, #0A0D14 0%, #0D1017 50%, #0A0D14 100%);
    }

    /* Headings (Main Body) */
    .main h1, .main h2, .main h3, .main h4 {
        color: #F8F9FA !important;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111520 0%, #0D1017 100%) !important;
        border-right: 1px solid #1E2330;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {
        color: #F8F9FA !important;
        font-weight: 700;
    }
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown p {
        color: #A0AABF !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(22, 25, 34, 0.6);
        border-radius: 12px;
        padding: 4px;
        border: 1px solid #2B313F;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #8A94A6;
        font-weight: 500;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00ffcc22, #00ffcc11) !important;
        color: #00ffcc !important;
        border-bottom: none !important;
    }

    /* Metric cards */
    .metric-container {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .metric-card {
        background: linear-gradient(145deg, #161922, #1E2330);
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid #2B313F;
        flex: 1;
        min-width: 200px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, transparent, #2B313F, transparent);
        transition: background 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 255, 204, 0.08);
        border-color: #00ffcc44;
    }
    .metric-card:hover::before {
        background: linear-gradient(90deg, transparent, #00ffcc, transparent);
    }
    .metric-title {
        color: #6B7280;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .metric-desc {
        color: #5A6170;
        font-size: 0.82rem;
        margin-bottom: 16px;
        line-height: 1.4;
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .metric-value.hist {
        color: #FFFFFF;
        text-shadow: 0 0 20px rgba(255,255,255,0.15);
    }
    .metric-value.arima {
        color: #00ffcc;
        text-shadow: 0 0 20px rgba(0,255,204,0.2);
    }
    .metric-value.nn {
        color: #FF4B4B;
        text-shadow: 0 0 20px rgba(255,75,75,0.2);
    }

    /* Info cards for About section */
    .info-card {
        background: linear-gradient(145deg, #161922, #1E2330);
        padding: 28px;
        border-radius: 16px;
        border: 1px solid #2B313F;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .info-card:hover {
        border-color: #00ffcc33;
        box-shadow: 0 8px 24px rgba(0,255,204,0.05);
    }
    .info-card h4 {
        color: #00ffcc !important;
        margin-bottom: 12px !important;
        font-size: 1.1rem;
    }
    .info-card p, .info-card li {
        color: #A0AABF !important;
        line-height: 1.7;
        font-size: 0.92rem;
    }

    /* Badge pill */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-green { background: #00ffcc22; color: #00ffcc; border: 1px solid #00ffcc44; }
    .badge-red { background: #FF4B4B22; color: #FF4B4B; border: 1px solid #FF4B4B44; }
    .badge-blue { background: #3B82F622; color: #60A5FA; border: 1px solid #3B82F644; }
    .badge-purple { background: #8B5CF622; color: #A78BFA; border: 1px solid #8B5CF644; }

    /* Divider */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, rgba(43,49,63,0) 0%, rgba(43,49,63,1) 50%, rgba(43,49,63,0) 100%);
        margin: 30px 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0A0D14; }
    ::-webkit-scrollbar-thumb { background: #2B313F; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3B4252; }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: #161922 !important;
        border-radius: 12px !important;
        color: #A0AABF !important;
    }

    /* Hero section */
    .hero-subtitle {
        text-align: center;
        color: #6B7280;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: -10px;
        margin-bottom: 5px;
        letter-spacing: 0.2px;
    }
    .hero-badges {
        text-align: center;
        margin-bottom: 20px;
    }

    /* Model metrics table */
    .metrics-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2B313F;
    }
    .metrics-table th {
        background: #161922;
        color: #6B7280;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 14px 20px;
        text-align: left;
        font-weight: 600;
    }
    .metrics-table td {
        background: #0D1017;
        color: #F8F9FA;
        padding: 14px 20px;
        border-top: 1px solid #1E2330;
        font-size: 0.95rem;
    }
    .metrics-table tr:hover td {
        background: #161922;
    }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'processed_monthly_passengers.csv')
raw_data_path = os.path.join(base_dir, 'data', 'PIA_2026_Advanced_Kaggle_Dataset.csv')
arima_path = os.path.join(base_dir, 'models', 'arima_model.pkl')
lstm_path = os.path.join(base_dir, 'models', 'lstm_model.pkl')
scaler_path = os.path.join(base_dir, 'models', 'lstm_scaler.pkl')
metrics_path = os.path.join(base_dir, 'models', 'evaluation_metrics.csv')

# --- Functions for Real-Time Forecast ---
@st.cache_resource
def load_models():
    with open(arima_path, 'rb') as f: arima = pickle.load(f)
    with open(lstm_path, 'rb') as f: lstm = pickle.load(f)
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    return arima, lstm, scaler

@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def load_metrics():
    if os.path.exists(metrics_path):
        return pd.read_csv(metrics_path)
    return None

def forecast_future_arima(model, steps):
    return model.forecast(steps=steps).values

def forecast_future_lstm(model, scaler, last_sequence, steps):
    curr_seq = last_sequence.copy()
    predictions = []
    for _ in range(steps):
        pred_scaled = model.predict(curr_seq.reshape(1, -1))
        curr_seq = np.append(curr_seq[1:], pred_scaled[0])
        predictions.append(pred_scaled[0])
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()


# --- Load Data ---
df_hist = load_data()

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("## ✈️ Control Panel")
    st.markdown("---")

    forecast_months = st.slider(
        "📅 Forecast Horizon (Months)",
        min_value=1, max_value=60, value=12, step=1,
        help="How many months into the future to project."
    )

    st.markdown("")
    st.markdown("**🧠 Algorithm Selection**")
    model_choice = st.radio(
        "Choose which model to display:",
        ("Compare Both", "ARIMA (Statistical)", "Neural Network (Deep Learning)"),
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**📊 Historical Window**")
    hist_window = st.slider(
        "Months of history to display",
        min_value=12, max_value=len(df_hist), value=min(48, len(df_hist)), step=6,
        help="Number of trailing historical months to show."
    )

    st.markdown("---")
    st.markdown(
        "<p style='color:#4B5563; font-size:0.75rem; text-align:center;'>"
        "Built by Nikhil Kale<br>Applied Data Science Project</p>",
        unsafe_allow_html=True
    )


# --- Header ---
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0; font-size:2.4rem;'>"
    "✈️ Airline Passenger AI Forecaster</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p class='hero-subtitle'>"
    "Real-time predictive modeling with ARIMA & Neural Networks — interactive, responsive, and production-ready."
    "</p>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='hero-badges'>"
    "<span class='badge badge-green'>LIVE INFERENCE</span>&nbsp;&nbsp;"
    "<span class='badge badge-blue'>ARIMA</span>&nbsp;&nbsp;"
    "<span class='badge badge-red'>NEURAL NET</span>&nbsp;&nbsp;"
    "<span class='badge badge-purple'>PLOTLY INTERACTIVE</span>"
    "</div>",
    unsafe_allow_html=True
)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Load Models & Run Real-Time Inference ---
with st.spinner('🔄 Loading models and computing forecasts...'):
    arima_model, lstm_model, lstm_scaler = load_models()

    future_arima = forecast_future_arima(arima_model, forecast_months)

    last_12 = df_hist['Passengers'].values[-12:]
    last_12_scaled = lstm_scaler.transform(last_12.reshape(-1, 1)).flatten()
    future_lstm = forecast_future_lstm(lstm_model, lstm_scaler, last_12_scaled, forecast_months)

    future_dates = pd.date_range(
        start=df_hist['Date'].iloc[-1] + pd.DateOffset(months=1),
        periods=forecast_months, freq='MS'
    )

# Calculate Metric Totals
last_year_passengers = int(df_hist[df_hist['Year'] == df_hist['Year'].max()]['Passengers'].sum())
projected_arima = int(future_arima.sum())
projected_lstm = int(future_lstm.sum())
avg_monthly_arima = int(future_arima.mean())
avg_monthly_lstm = int(future_lstm.mean())

# ======================== TABS ========================
tab_forecast, tab_eda, tab_models, tab_about = st.tabs([
    "🚀 Forecast", "📊 Exploratory Analysis", "🧠 Model Performance", "ℹ️ About"
])

# ======================== TAB 1: FORECAST ========================
with tab_forecast:

    # --- Metrics Row ---
    num_cols = 3 if model_choice == "Compare Both" else 2
    cols = st.columns(num_cols)

    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Historical Baseline</div>
            <div class="metric-desc">Total passengers in the most recent 12 months of data.</div>
            <div class="metric-value hist">{last_year_passengers:,}</div>
        </div>
        """, unsafe_allow_html=True)

    if model_choice in ["ARIMA (Statistical)", "Compare Both"]:
        with cols[1]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">ARIMA Projection</div>
                <div class="metric-desc">Total estimated volume over the <b>next {forecast_months} months</b>.</div>
                <div class="metric-value arima">{projected_arima:,}</div>
            </div>
            """, unsafe_allow_html=True)

    if model_choice in ["Neural Network (Deep Learning)", "Compare Both"]:
        target_col = cols[2] if model_choice == "Compare Both" else cols[1]
        with target_col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Neural Net Projection</div>
                <div class="metric-desc">Total estimated volume over the <b>next {forecast_months} months</b>.</div>
                <div class="metric-value nn">{projected_lstm:,}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    # --- Interactive Real-Time Chart ---
    st.markdown(f"### 📈 Timeline Simulation — {forecast_months}-Month Horizon")

    hist_visible = df_hist.tail(hist_window).copy()
    chart_data = pd.DataFrame({
        'Date': hist_visible['Date'],
        'Passengers': hist_visible['Passengers'],
        'Legend': 'Historical Actuals'
    })

    last_hist_date = hist_visible['Date'].iloc[-1]
    last_hist_val = hist_visible['Passengers'].iloc[-1]

    if model_choice in ["ARIMA (Statistical)", "Compare Both"]:
        arima_df = pd.DataFrame({
            'Date': [last_hist_date] + list(future_dates),
            'Passengers': [last_hist_val] + list(future_arima),
            'Legend': 'ARIMA Forecast'
        })
        chart_data = pd.concat([chart_data, arima_df], ignore_index=True)

    if model_choice in ["Neural Network (Deep Learning)", "Compare Both"]:
        nn_df = pd.DataFrame({
            'Date': [last_hist_date] + list(future_dates),
            'Passengers': [last_hist_val] + list(future_lstm),
            'Legend': 'Neural Net Forecast'
        })
        chart_data = pd.concat([chart_data, nn_df], ignore_index=True)

    color_map = {
        'Historical Actuals': '#FFFFFF',
        'ARIMA Forecast': '#00ffcc',
        'Neural Net Forecast': '#FF4B4B'
    }

    fig = px.line(
        chart_data, x="Date", y="Passengers", color="Legend",
        color_discrete_map=color_map,
    )
    fig.update_traces(mode='lines+markers', marker=dict(size=5), line=dict(width=2.5))
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#A0AABF', family='Inter, Segoe UI, sans-serif', size=13),
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor='#1E2330', gridwidth=1, title="", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#1E2330', gridwidth=1, title="Monthly Passengers", zeroline=False),
        legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=12)),
        height=450
    )
    st.plotly_chart(fig, width='stretch')

    # --- Summary Statistics ---
    st.markdown("### 📋 Forecast Summary")
    sum_cols = st.columns(4)

    with sum_cols[0]:
        st.metric("Forecast Horizon", f"{forecast_months} mo")
    with sum_cols[1]:
        st.metric("Historical Avg (Monthly)", f"{int(df_hist['Passengers'].mean()):,}")
    with sum_cols[2]:
        if model_choice != "Neural Network (Deep Learning)":
            st.metric("ARIMA Avg (Monthly)", f"{avg_monthly_arima:,}")
        else:
            st.metric("NN Avg (Monthly)", f"{avg_monthly_lstm:,}")
    with sum_cols[3]:
        if model_choice == "Compare Both":
            diff = projected_arima - projected_lstm
            st.metric("Model Divergence", f"{abs(diff):,}", delta=f"{'ARIMA higher' if diff > 0 else 'NN higher'}")
        elif model_choice == "Neural Network (Deep Learning)":
            st.metric("NN Avg (Monthly)", f"{avg_monthly_lstm:,}")
        else:
            pct = ((projected_arima - last_year_passengers) / last_year_passengers * 100) if last_year_passengers else 0
            st.metric("Projected Growth", f"{pct:+.1f}%")

    # --- Data Table ---
    with st.expander("📊 View Raw Forecast Data", expanded=False):
        st.markdown("Exportable tabular view of all generated forecast values.")
        view_df = pd.DataFrame({'Date': future_dates})
        if model_choice in ["ARIMA (Statistical)", "Compare Both"]:
            view_df['ARIMA Forecast'] = future_arima
        if model_choice in ["Neural Network (Deep Learning)", "Compare Both"]:
            view_df['Neural Net Forecast'] = future_lstm
        st.dataframe(
            view_df.style.format({col: "{:,.0f}" for col in view_df.columns if col != 'Date'}),
            use_container_width=True, hide_index=True
        )

# ======================== TAB 2: EDA ========================
with tab_eda:
    st.markdown("### 📊 Exploratory Data Analysis")
    st.markdown(
        "<p style='color:#6B7280; margin-bottom:20px;'>"
        "Deep-dive into the underlying data patterns, seasonality, and statistical properties."
        "</p>",
        unsafe_allow_html=True
    )

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        # Full Historical Trend
        st.markdown("#### 📈 Full Historical Trend")
        fig_trend = px.area(
            df_hist, x='Date', y='Passengers',
            color_discrete_sequence=['#00ffcc']
        )
        fig_trend.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#A0AABF', family='Inter, sans-serif'),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, title=""), yaxis=dict(showgrid=True, gridcolor='#1E2330', title="Passengers"),
            height=350
        )
        fig_trend.update_traces(fillcolor='rgba(0,255,204,0.08)', line=dict(width=2))
        st.plotly_chart(fig_trend, use_container_width=True)

    with eda_col2:
        # Monthly Distribution
        st.markdown("#### 📦 Monthly Distribution (Box Plot)")
        fig_box = px.box(
            df_hist, x='Month', y='Passengers',
            color_discrete_sequence=['#60A5FA']
        )
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#A0AABF', family='Inter, sans-serif'),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, title="Month", dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#1E2330', title="Passengers"),
            height=350
        )
        st.plotly_chart(fig_box, use_container_width=True)

    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        # Year-over-Year Seasonal Pattern
        st.markdown("#### 🔄 Year-over-Year Seasonality")
        fig_season = px.line(
            df_hist, x='Month', y='Passengers', color='Year',
            color_discrete_sequence=px.colors.sequential.Viridis,
            markers=True
        )
        fig_season.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#A0AABF', family='Inter, sans-serif'),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False, title="Month", dtick=1),
            yaxis=dict(showgrid=True, gridcolor='#1E2330', title="Passengers"),
            legend=dict(title="Year", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=350
        )
        st.plotly_chart(fig_season, use_container_width=True)

    with eda_col4:
        # Correlation Heatmap
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        corr_cols = ['Year', 'Month', 'Passengers']
        if 'Trend' in df_hist.columns:
            corr_cols.append('Trend')
        corr_matrix = df_hist[corr_cols].corr()

        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.columns.tolist(),
            colorscale=[[0, '#FF4B4B'], [0.5, '#1E2330'], [1, '#00ffcc']],
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont=dict(size=14, color='#F8F9FA'),
            zmin=-1, zmax=1
        ))
        fig_heat.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#A0AABF', family='Inter, sans-serif'),
            margin=dict(l=0, r=0, t=10, b=0),
            height=350
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # Year-wise Aggregated Bar Chart
    st.markdown("#### 📊 Annual Passenger Volume")
    yearly = df_hist.groupby('Year')['Passengers'].sum().reset_index()
    fig_bar = px.bar(
        yearly, x='Year', y='Passengers',
        color='Passengers',
        color_continuous_scale=[[0, '#161922'], [0.5, '#00997a'], [1, '#00ffcc']],
        text='Passengers'
    )
    fig_bar.update_traces(
        texttemplate='%{text:,.0f}', textposition='outside',
        marker_line_color='#00ffcc', marker_line_width=1
    )
    fig_bar.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#A0AABF', family='Inter, sans-serif'),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False, title="", dtick=1),
        yaxis=dict(showgrid=True, gridcolor='#1E2330', title="Total Passengers"),
        coloraxis_showscale=False,
        height=350
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Statistical Summary
    with st.expander("📈 Statistical Summary", expanded=False):
        st.dataframe(
            df_hist[['Passengers', 'Year', 'Month']].describe().T.style.format("{:.1f}"),
            use_container_width=True
        )


# ======================== TAB 3: MODEL PERFORMANCE ========================
with tab_models:
    st.markdown("### 🧠 Model Performance & Evaluation")
    st.markdown(
        "<p style='color:#6B7280; margin-bottom:20px;'>"
        "Comparative analysis of ARIMA vs Neural Network accuracy on the held-out test set."
        "</p>",
        unsafe_allow_html=True
    )

    metrics_df = load_metrics()

    if metrics_df is not None and len(metrics_df) > 0:
        m_col1, m_col2 = st.columns(2)

        arima_row = metrics_df[metrics_df['Model'] == 'ARIMA'].iloc[0] if 'ARIMA' in metrics_df['Model'].values else None
        lstm_row = metrics_df[metrics_df['Model'] == 'LSTM'].iloc[0] if 'LSTM' in metrics_df['Model'].values else None

        with m_col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">ARIMA — Statistical Model</div>
                <div style="margin-top:12px;">
                    <span class="badge badge-green">Autoregressive</span>&nbsp;
                    <span class="badge badge-blue">Integrated</span>&nbsp;
                    <span class="badge badge-purple">Moving Average</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if arima_row is not None:
                a1, a2 = st.columns(2)
                a1.metric("MAE", f"{arima_row['MAE']:,.1f}")
                a2.metric("RMSE", f"{arima_row['RMSE']:,.1f}")
            st.markdown(
                "<p style='color:#6B7280; font-size:0.85rem; padding:0 4px;'>"
                "ARIMA uses statistical autoregression to capture linear trends and seasonal patterns. "
                "Best for short-to-medium term horizons with stable seasonality.</p>",
                unsafe_allow_html=True
            )

        with m_col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Neural Network — Deep Learning</div>
                <div style="margin-top:12px;">
                    <span class="badge badge-red">MLP Regressor</span>&nbsp;
                    <span class="badge badge-blue">Scaled Features</span>&nbsp;
                    <span class="badge badge-purple">Non-Linear</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if lstm_row is not None:
                n1, n2 = st.columns(2)
                n1.metric("MAE", f"{lstm_row['MAE']:,.1f}")
                n2.metric("RMSE", f"{lstm_row['RMSE']:,.1f}")
            st.markdown(
                "<p style='color:#6B7280; font-size:0.85rem; padding:0 4px;'>"
                "The Neural Network (MLPRegressor) captures non-linear dynamics via multi-layer perceptrons. "
                "Excels at detecting complex, non-obvious patterns.</p>",
                unsafe_allow_html=True
            )

        # Comparative Bar Chart
        st.markdown("#### 📊 Error Metrics Comparison")
        if arima_row is not None and lstm_row is not None:
            comp_df = pd.DataFrame({
                'Metric': ['MAE', 'MAE', 'RMSE', 'RMSE'],
                'Model': ['ARIMA', 'Neural Net', 'ARIMA', 'Neural Net'],
                'Value': [arima_row['MAE'], lstm_row['MAE'], arima_row['RMSE'], lstm_row['RMSE']]
            })
            fig_comp = px.bar(
                comp_df, x='Metric', y='Value', color='Model', barmode='group',
                color_discrete_map={'ARIMA': '#00ffcc', 'Neural Net': '#FF4B4B'},
                text='Value'
            )
            fig_comp.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_comp.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#A0AABF', family='Inter, sans-serif'),
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False, title=""),
                yaxis=dict(showgrid=True, gridcolor='#1E2330', title="Error Value"),
                legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=350
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        # Winner declaration
        if arima_row is not None and lstm_row is not None:
            arima_score = arima_row['MAE'] + arima_row['RMSE']
            lstm_score = lstm_row['MAE'] + lstm_row['RMSE']
            winner = "ARIMA" if arima_score < lstm_score else "Neural Network"
            winner_color = "#00ffcc" if winner == "ARIMA" else "#FF4B4B"
            st.markdown(
                f"<div style='text-align:center; padding:20px; background:#161922; border-radius:12px; "
                f"border:1px solid {winner_color}33; margin-top:10px;'>"
                f"<span style='color:#6B7280; text-transform:uppercase; font-size:0.8rem; "
                f"letter-spacing:1px;'>Best Performing Model</span><br>"
                f"<span style='color:{winner_color}; font-size:1.8rem; font-weight:800;'>"
                f"🏆 {winner}</span></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("⚠️ No evaluation metrics found. Run `python src/forecasting.py` to generate metrics.")


# ======================== TAB 4: ABOUT ========================
with tab_about:
    st.markdown("### ℹ️ About This Project")

    st.markdown("""
    <div class="info-card">
        <h4>🎯 Project Overview</h4>
        <p>
            This is an <b>Applied Data Science</b> project that forecasts future airline passenger traffic
            using statistical and machine learning models. The system processes raw historical data,
            identifies seasonal trends, and trains scalable algorithms to display interactive future
            projections in real-time via this Streamlit dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    about_c1, about_c2 = st.columns(2)
    with about_c1:
        st.markdown("""
        <div class="info-card">
            <h4>🔬 Data Pipeline</h4>
            <ul>
                <li><b>Data Ingestion</b> — Raw CSV loading, DateTime structuring, and monthly aggregation.</li>
                <li><b>Synthetic Extension</b> — 5 years of backward-synthesized data with growth trends and noise for robust training.</li>
                <li><b>Feature Engineering</b> — Trend indices, seasonal decomposition, and statistical feature extraction.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with about_c2:
        st.markdown("""
        <div class="info-card">
            <h4>🧠 Models Used</h4>
            <ul>
                <li><b>ARIMA (5,1,0)</b> — Autoregressive Integrated Moving Average for capturing linear trends and seasonality.</li>
                <li><b>MLPRegressor</b> — Multi-Layer Perceptron neural network with ReLU activation and Adam optimizer for non-linear pattern detection.</li>
                <li><b>MinMaxScaler</b> — Feature normalization for optimal neural network convergence.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>🏗️ Tech Stack</h4>
        <p>
            <span class="badge badge-green">Python</span>&nbsp;
            <span class="badge badge-blue">Streamlit</span>&nbsp;
            <span class="badge badge-purple">Plotly</span>&nbsp;
            <span class="badge badge-red">scikit-learn</span>&nbsp;
            <span class="badge badge-green">statsmodels</span>&nbsp;
            <span class="badge badge-blue">pandas</span>&nbsp;
            <span class="badge badge-purple">NumPy</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>👨‍💻 Developer</h4>
        <p>
            <b>Nikhil Kale</b><br>
            Applied Data Science • Machine Learning • Time Series Forecasting<br><br>
            <a href="https://github.com/KaleNikhil7099" target="_blank" style="color:#00ffcc; text-decoration:none;">
                🔗 GitHub Profile
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

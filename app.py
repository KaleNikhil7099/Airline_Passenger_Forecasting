import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import os

# --- Page Configurations ---
st.set_page_config(
    page_title="Airline Passenger Forecaster",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Premium Clarity & Fonts & Hiding Footer ---
st.markdown("""
<style>
    /* Hide Streamlit default footer and Deploy button, but keep the Menu */
    footer {visibility: hidden;}
    header {visibility: visible;} 
    .stAppDeployButton {display: none;}
    
    /* Global background and typography (Native System Fonts for 100% Reliability) */
    .stApp, *:not(i) {
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
    }
    .stApp {
        background-color: #0A0D14;
    }
    
    /* Headings (Main Body) */
    .main h1, .main h2, .main h3, .main h4 {
        color: #F8F9FA !important;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    /* Headings (Sidebar) - Force Dark Text for visibility against light background */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4, [data-testid="stSidebarNav"] * {
        color: #0A0D14 !important;
        font-weight: 700;
    }
    
    /* Metrics panel styling */
    .metric-container {
        display: flex;
        gap: 20px;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(145deg, #161922, #1E2330);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.4);
        border: 1px solid #2B313F;
        flex: 1;
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #00ffcc;
    }
    
    /* Metric Typography */
    .metric-title {
        color: #A0AABF;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-desc {
        color: #6B7280;
        font-size: 0.9rem;
        margin-bottom: 18px;
        line-height: 1.4;
    }
    .metric-value {
        font-size: 2.8rem;
        font-weight: 800;
    }
    .metric-value.hist {
        color: #FFFFFF;
        text-shadow: 0 0 10px rgba(255,255,255,0.3);
    }
    .metric-value.arima {
        color: #00ffcc;
        text-shadow: 0 0 10px rgba(0,255,204,0.3);
    }
    .metric-value.nn {
        color: #FF4B4B;
        text-shadow: 0 0 10px rgba(255,75,75,0.3);
    }
    
    /* Divider */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, rgba(43,49,63,0) 0%, rgba(43,49,63,1) 50%, rgba(43,49,63,0) 100%);
        margin: 30px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, 'data', 'processed_monthly_passengers.csv')
arima_path = os.path.join(base_dir, 'models', 'arima_model.pkl')
lstm_path = os.path.join(base_dir, 'models', 'lstm_model.pkl')
scaler_path = os.path.join(base_dir, 'models', 'lstm_scaler.pkl')

# --- Functions for Real-Time Forecast ---
@st.cache_resource
def load_models():
    with open(arima_path, 'rb') as f: arima = pickle.load(f)
    with open(lstm_path, 'rb') as f: lstm = pickle.load(f)
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    return arima, lstm, scaler

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

# --- Load Database ---
df_hist = pd.read_csv(data_path)
df_hist['Date'] = pd.to_datetime(df_hist['Date'])

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("🛫 Forecast Controls")
    
    forecast_months = st.slider("Forecast Horizon (Months):", min_value=1, max_value=60, value=12, step=1)
    
    st.subheader("✈️ Select Algorithm")
    model_choice = st.radio(
        "Choose which algorithm to isolate and evaluate:", 
        ("ARIMA (Statistical)", "Neural Network (Deep Learning)", "Compare Both")
    )

# --- Header ---
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>✈️ Airline Passenger AI Forecaster</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #8A94A6; font-weight: 400;'>High-clarity, interactive predictive modeling.</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# --- Load Models & Run Real-Time Inference ---
with st.spinner('Orchestrating mathematical forecasts...'):
    arima_model, lstm_model, lstm_scaler = load_models()
    
    # Run ARIMA
    future_arima = forecast_future_arima(arima_model, forecast_months)
    
    # Run Neural Network
    last_12 = df_hist['Passengers'].values[-12:]
    last_12_scaled = lstm_scaler.transform(last_12.reshape(-1, 1)).flatten()
    future_lstm = forecast_future_lstm(lstm_model, lstm_scaler, last_12_scaled, forecast_months)
    
    future_dates = pd.date_range(start=df_hist['Date'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_months, freq='M')

# Calculate Metric Totals
last_year_passengers = int(df_hist[df_hist['Year'] == df_hist['Year'].max()]['Passengers'].sum())
projected_arima = int(future_arima.sum())
projected_lstm = int(future_lstm.sum())

# --- Conditional Metrics Display ---
st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
cols = st.columns(3 if model_choice == "Compare Both" else 2)

with cols[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Historical Baseline</div>
        <div class="metric-desc">Total passengers aggregated over the exact 12 preceding months.</div>
        <div class="metric-value hist">{last_year_passengers:,}</div>
    </div>
    """, unsafe_allow_html=True)

if model_choice in ["ARIMA (Statistical)", "Compare Both"]:
    target_col = cols[1]
    with target_col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ARIMA Projection</div>
            <div class="metric-desc">Total estimated flight volume over the <b>next {forecast_months} months</b>.</div>
            <div class="metric-value arima">{projected_arima:,}</div>
        </div>
        """, unsafe_allow_html=True)

if model_choice in ["Neural Network (Deep Learning)", "Compare Both"]:
    target_col = cols[2] if model_choice == "Compare Both" else cols[1]
    with target_col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Neural Net Projection</div>
            <div class="metric-desc">Total estimated flight volume over the <b>next {forecast_months} months</b>.</div>
            <div class="metric-value nn">{projected_lstm:,}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Interactive Real-Time Charting (Plotly) ---
st.markdown(f"### 📈 Interactive Timeline Simulation ({forecast_months}-Month Horizon)")

# Construct Dataframe for Plotly
hist_visible = df_hist.tail(48).copy()
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

# Define explicit color mapping for 100% precision
color_discrete_map = {
    'Historical Actuals': '#FFFFFF',
    'ARIMA Forecast': '#00ffcc',
    'Neural Net Forecast': '#FF4B4B'
}

# Generate Plotly Express Graph
fig = px.line(
    chart_data, 
    x="Date", 
    y="Passengers", 
    color="Legend", 
    color_discrete_map=color_discrete_map,
)

# Fine-tune the Plotly presentation
fig.update_traces(mode='lines+markers', marker=dict(size=6), line=dict(width=3))
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#A0AABF', family='Segoe UI, system-ui, sans-serif'),
    margin=dict(l=0, r=0, t=20, b=0),
    hovermode="x unified",
    xaxis=dict(showgrid=True, gridcolor='#2B313F', gridwidth=1, title="Timeline"),
    yaxis=dict(showgrid=True, gridcolor='#2B313F', gridwidth=1, title="Monthly Passenger Volume"),
    legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- Data Table Representation ---
with st.expander("📊 View Raw Output Array", expanded=False):
    st.markdown("Exportable view of the isolated data arrays generated by your selected models.")
    
    view_df = pd.DataFrame({'Date': future_dates})
    if model_choice in ["ARIMA (Statistical)", "Compare Both"]:
        view_df['ARIMA Forecast'] = future_arima
    if model_choice in ["Neural Network (Deep Learning)", "Compare Both"]:
        view_df['Neural Net Forecast'] = future_lstm
        
    st.dataframe(view_df.style.format({col: "{:,.0f}" for col in view_df.columns if col != 'Date'}), use_container_width=True)

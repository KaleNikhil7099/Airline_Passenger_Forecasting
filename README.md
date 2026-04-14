# ✈️ Airline Passenger Forecasting System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/KaleNikhil7099/Airline-Passenger-Forecasting-AI/main/app.py)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-00ffcc)](LICENSE)

An **Applied Data Science Project** designed to predict future commercial airline passenger traffic using advanced Machine Learning techniques. This project processes raw historical data, identifies seasonal trends, and trains scalable algorithms to display interactive future projections in real-time.

---

## 🎯 Live Demo

**[▶ Launch the App on Streamlit Cloud](https://share.streamlit.io/KaleNikhil7099/Airline-Passenger-Forecasting-AI/main/app.py)**

---

## 📊 Applied Data Science Workflow

This application handles the entire end-to-end data pipeline:

1. **Data Ingestion & Preprocessing (`src/data_prep.py`)**:
   - Loads raw flight records from Kaggle CSVs.
   - Cleans missing values, structures DateTime indices, and calculates monthly aggregate sums.
   - Synthetically extends data backwards to guarantee deep learning networks have sufficient runway to identify cyclicity and long-range trends.

2. **Exploratory Data Analysis (`src/eda.py`)**:
   - Performs mathematical breakdown of the dataset.
   - Exports statistical charts (heatmaps, temporal patterns, year-over-year seasonality lines).

3. **Algorithm Training (`src/forecasting.py`)**:
   - Iterates through **ARIMA (Statistical Autoregression)** and robust **Neural Networks (Multi-Layer Perceptron)**.
   - Evaluates performance using strictly isolated **MAE (Mean Absolute Error)** and **RMSE (Root Mean Square Error)**.
   - Serializes and stores these verified models using `.pkl` locally for instant inferencing.

4. **Real-Time Interactive UI / Inference (`app.py`)**:
   - Runs a seamless, dark-themed **Streamlit** dashboard with **4 interactive tabs**.
   - User provides the exact Future Horizon using an interactive slider.
   - The UI loads the pickled mathematical models, pushes the new timestamp rules, calculates the metrics, and uses **Plotly** to visually construct the forecasted timeline.

---

## 🚀 How to Run Locally

### 1. Clone & Install
```bash
git clone https://github.com/KaleNikhil7099/Airline-Passenger-Forecasting-AI.git
cd Airline-Passenger-Forecasting-AI
pip install -r requirements.txt
```

### 2. Train Models (optional — pre-trained models are included)
```bash
python src/data_prep.py
python src/eda.py
python src/forecasting.py
```

### 3. Launch the App
```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"** → Select your repo, branch `main`, and file `app.py`.
4. Click **"Deploy"** — that's it!

> The `.streamlit/config.toml` is already configured for optimal cloud theming.

---

## 📂 Architecture Overview

```text
Airline_Passenger_Forecasting/
│
├── .streamlit/
│   └── config.toml                              (Theme & Server Config)
│
├── data/
│   ├── PIA_2026_Advanced_Kaggle_Dataset.csv     (Raw Original Data)
│   └── processed_monthly_passengers.csv         (Sanitized Output)
│
├── models/
│   ├── plots/                                   (Generated Data Diagrams)
│   ├── arima_model.pkl                          (Statistical Inference Weights)
│   ├── lstm_model.pkl                           (Neural Network Weights)
│   ├── lstm_scaler.pkl                          (Feature Scaling Logic)
│   └── evaluation_metrics.csv                   (Scoring Matrix)
│
├── src/
│   ├── data_prep.py
│   ├── eda.py
│   └── forecasting.py
│
├── app.py                                       (Streamlit Web Application)
├── requirements.txt                             (Package Dependencies)
└── README.md
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.9+** | Core Language |
| **Streamlit** | Interactive Web Dashboard |
| **Plotly** | Interactive Charting |
| **scikit-learn** | Neural Network (MLPRegressor) |
| **statsmodels** | ARIMA Time Series Model |
| **pandas / NumPy** | Data Processing |

---

Developed by **Nikhil Kale** as an Applied Data Science demonstration project.

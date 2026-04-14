# ✈️ Airline Passenger Forecasting System

An **Applied Data Science Project** designed to predict future commercial airline passenger traffic using advanced Machine Learning techniques. This project processes raw historical data, identifies seasonal trends, and trains scalable algorithms to display interactive future projections in real-time.

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
   - Iterates through **ARIMA (Statistical Autoregression)** and robust **Neural Networks (Multi-Layer Perceptron / LSTM fallbacks)**.
   - Evaluates performance using strictly isolated **MAE (Mean Absolute Error)** and **RMSE (Root Mean Square Error)**.
   - Serializes and stores these verified models using `.pkl` locally for instant inferencing.

4. **Real-Time Interactive UI / Inference (`app.py`)**:
   - Runs a seamless, dark-themed **Streamlit** dashboard.
   - User provides the exact Future Horizon using an interactive slider.
   - The UI loads the pickled mathematical models, pushes the new timestamp rules, calculates the metrics, and uses **Plotly Express** to visually construct the forecasted timeline alongside raw tabular export structures.

---

## 🚀 How to Run Locally

### 1. Requirements
Ensure you are in the project root directory and install exactly what this ecosystem needs via Python Pip.
```powershell
pip install -r requirements.txt
```

### 2. Orchestrate the Model Training
Launch the internal training scripts to produce your serialized models.
```powershell
python src/data_prep.py
python src/eda.py
python src/forecasting.py
```

### 3. Launch the Application!
Spawn the Streamlit instance. Your browser will automatically open a local server instance (typically `localhost:8501`).
```powershell
streamlit run app.py
```

---

## 📂 Architecture Overview

```text
Airline_Passenger_Forecasting\
│
├── data\
│   ├── PIA_2026_Advanced_Kaggle_Dataset.csv     (Raw Original Data)
│   └── processed_monthly_passengers.csv         (Sanitized Output)
│
├── models\
│   ├── plots\                                   (Generated Data Diagrams)
│   ├── arima_model.pkl                          (Statistical Inference Weights)
│   ├── lstm_model.pkl                           (Neural Network Weights)
│   ├── lstm_scaler.pkl                          (Feature Scaling Logic)
│   └── evaluation_metrics.csv                   (Scoring Matrix)
│
├── src\
│   ├── data_prep.py
│   ├── eda.py
│   └── forecasting.py
│
├── app.py                                       (Frontend Web Server)
└── requirements.txt                             (Package Locks)
```

Developed as an Applied Data Science demonstration model.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y).ravel()

def train_arima(train, test):
    print("Training ARIMA Model...")
    # Simple (p,d,q) order for monthly data
    model = ARIMA(train, order=(5, 1, 0))
    fitted_model = model.fit()
    
    predictions = fitted_model.forecast(steps=len(test))
    
    mae, rmse = evaluate_metrics(test, predictions)
    print(f"ARIMA MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return fitted_model, predictions, mae, rmse

def train_lstm(train, test, seq_length=12):
    print("Training Neural Net (LSTM Fallback) Model...")
    scaler = MinMaxScaler()
    
    # Scale data
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
    test_scaled = scaler.transform(test.values.reshape(-1, 1))
    
    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1])) # Flatten for MLP
    
    model = MLPRegressor(hidden_layer_sizes=(50, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
    
    # Train
    model.fit(X_train, y_train)
    
    # Predictions
    inputs = pd.concat([train.iloc[-seq_length:], test]).values.reshape(-1, 1)
    inputs_scaled = scaler.transform(inputs)
    
    X_test, _ = create_sequences(inputs_scaled, seq_length)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))
    pred_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    
    mae, rmse = evaluate_metrics(test, predictions)
    print(f"Neural Net MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return model, scaler, predictions, mae, rmse

def forecast_future_arima(model, steps=12):
    return model.forecast(steps=steps).values

def forecast_future_lstm(model, scaler, last_sequence, steps=12):
    curr_seq = last_sequence.copy()
    predictions = []
    
    for _ in range(steps):
        pred_scaled = model.predict(curr_seq.reshape(1, -1))
        curr_seq = np.append(curr_seq[1:], pred_scaled[0])
        predictions.append(pred_scaled[0])
        
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def run_models_and_save(data_path, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    series = df['Passengers']
    
    # Train-test split (80-20)
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # Train Models
    arima_model, arima_preds, arima_mae, arima_rmse = train_arima(train, test)
    lstm_model, lstm_scaler, lstm_preds, lstm_mae, lstm_rmse = train_lstm(train, test, seq_length=12)
    
    # Evaluate Models on test set
    results_df = pd.DataFrame({
        'Model': ['ARIMA', 'LSTM'],
        'MAE': [arima_mae, lstm_mae],
        'RMSE': [arima_rmse, lstm_rmse]
    })
    results_df.to_csv(os.path.join(models_dir, 'evaluation_metrics.csv'), index=False)
    
    # Future forecasts (12 months ahead)
    future_arima = forecast_future_arima(arima_model, 12)
    
    # LSTM requires last 12 points scaled
    last_12 = series.values[-12:]
    last_12_scaled = lstm_scaler.transform(last_12.reshape(-1, 1)).flatten()
    future_lstm = forecast_future_lstm(lstm_model, lstm_scaler, last_12_scaled, 12)
    
    # Save the futures to CSV for the app
    future_dates = pd.date_range(start=pd.to_datetime(df['Date'].iloc[-1]) + pd.DateOffset(months=1), periods=12, freq='M')
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'ARIMA_Forecast': future_arima,
        'LSTM_Forecast': future_lstm
    })
    forecast_df.to_csv(os.path.join(models_dir, 'forecast_12_months.csv'), index=False)
    
    # Save Models
    arima_path = os.path.join(models_dir, 'arima_model.pkl')
    with open(arima_path, 'wb') as f:
        pickle.dump(arima_model, f)
    
    lstm_path = os.path.join(models_dir, 'lstm_model.pkl')
    with open(lstm_path, 'wb') as f:
        pickle.dump(lstm_model, f)
    
    # Save scaler
    with open(os.path.join(models_dir, 'lstm_scaler.pkl'), 'wb') as f:
        pickle.dump(lstm_scaler, f)
        
    print(f"Models and forecasts saved to {models_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, 'data', 'processed_monthly_passengers.csv')
    models_dir = os.path.join(base_dir, 'models')
    
    if os.path.exists(processed_path):
        run_models_and_save(processed_path, models_dir)
    else:
        print(f"Processed file not found at {processed_path}. Please run data_prep.py first.")

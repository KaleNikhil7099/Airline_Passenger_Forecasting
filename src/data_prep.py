import pandas as pd
import numpy as np
import os

def load_and_preprocess(filepath):
    print("Loading raw data...")
    df = pd.read_csv(filepath)
    
    # 1. Convert Date to Datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Extract Year and Month
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    
    # 3. Aggregate data by Month and Year
    df_monthly = df.groupby(['Year', 'Month'])['Passengers'].sum().reset_index()
    
    # 4. Synthesize 5 years of historical data based on the 2026 base
    print("Synthesizing historical data for robust ML training (2021-2025)...")
    historical_data = []
    
    # Calculate a baseline pattern from 2026
    # If 2026 doesn't have all 12 months, we pad it, but looking at EDA it has 1-12.
    base_2026 = df_monthly[df_monthly['Year'] == 2026].set_index('Month')['Passengers'].to_dict()
    
    # For missing months, use average
    avg_passengers = np.mean(list(base_2026.values()))
    for m in range(1, 13):
        if m not in base_2026:
            base_2026[m] = avg_passengers
            
    np.random.seed(42)  # For reproducibility
            
    # Go backwards to 2021
    for year in range(2021, 2026):
        # We apply a growth trend backwards, meaning older years have fewer passengers
        growth_downscale = 1.0 - ((2026 - year) * 0.05)  # 5% less each year backwards
        
        for month in range(1, 13):
            base_val = base_2026[month]
            # Add random noise (±10%)
            noise = np.random.uniform(0.90, 1.10)
            syn_passengers = int(base_val * growth_downscale * noise)
            
            historical_data.append({
                'Year': year,
                'Month': month,
                'Passengers': syn_passengers
            })
            
    df_history = pd.DataFrame(historical_data)
    
    # Combine history + 2026 data
    df_full = pd.concat([df_history, df_monthly[['Year', 'Month', 'Passengers']]], ignore_index=True)
    df_full = df_full.sort_values(by=['Year', 'Month']).reset_index(drop=True)
    
    # Create Date index feature
    df_full['Date'] = pd.to_datetime(df_full[['Year', 'Month']].assign(DAY=1))
    
    # 5. Feature Engineering
    df_full['Trend'] = np.arange(len(df_full))
    
    # Save the processed data
    processed_path = os.path.join(os.path.dirname(filepath), 'processed_monthly_passengers.csv')
    df_full.to_csv(processed_path, index=False)
    print(f"Preprocessed data saved to {processed_path}")
    
    return df_full

if __name__ == "__main__":
    # Ensure correct execution from script directly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'PIA_2026_Advanced_Kaggle_Dataset.csv')
    if os.path.exists(data_path):
        load_and_preprocess(data_path)
    else:
        print(f"Data file not found at {data_path}")

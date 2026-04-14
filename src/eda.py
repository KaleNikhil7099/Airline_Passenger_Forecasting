import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_processed_data(filepath):
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def generate_eda_plots(df, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print("Generating EDA plots...")
    
    # Setting modern aesthetic styles
    plt.style.use('dark_background')
    sns.set_context("talk")
    
    # 1. Line plot (Historical trend)
    plt.figure(figsize=(14, 6))
    plt.plot(df['Date'], df['Passengers'], color='#00ffcc', linewidth=2, marker='o', markersize=4)
    plt.title('Monthly Airline Passengers (Historical & Synthetic)', color='white')
    plt.xlabel('Date', color='white')
    plt.ylabel('Passengers', color='white')
    plt.grid(color='#444444', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'historical_trend.png'), facecolor='#111111')
    plt.close()
    
    # 2. Seasonal trend (Month-over-Month)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='Month', y='Passengers', hue='Year', palette='viridis', marker='o', linewidth=2)
    plt.title('Seasonal Trends (Year-over-Year comparison)', color='white')
    plt.xlabel('Month', color='white')
    plt.ylabel('Passengers', color='white')
    plt.xticks(range(1, 13))
    plt.grid(color='#444444', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'seasonal_trend.png'), facecolor='#111111')
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(8, 6))
    corr = df[['Year', 'Month', 'Passengers', 'Trend']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, annot_kws={"size": 12})
    plt.title('Feature Correlation Heatmap', color='white')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_heatmap.png'), facecolor='#111111')
    plt.close()
    
    print(f"EDA plots saved to {save_dir}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, 'data', 'processed_monthly_passengers.csv')
    
    if os.path.exists(processed_path):
        df = load_processed_data(processed_path)
        generate_eda_plots(df, os.path.join(base_dir, 'models', 'plots'))
    else:
        print(f"Processed file not found at {processed_path}. Please run data_prep.py first.")

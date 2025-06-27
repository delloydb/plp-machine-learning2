"""
utils.py
Utility functions for preprocessing and feature engineering.
"""

import pandas as pd

def summarize_dataset(df):
    """
    Prints basic info and statistics about the dataset.

    Args:
        df (pd.DataFrame): Input dataset
    """
    print("\n[INFO] Dataset Summary:")
    print(df.info())
    print("\n[INFO] First 5 Rows:")
    print(df.head())
    print("\n[INFO] Descriptive Statistics:")
    print(df.describe())

def simulate_future_inputs(base_df, future_years=[2025, 2030]):
    """
    Creates simulated future data based on trends or assumptions.

    Args:
        base_df (pd.DataFrame): Base input data
        future_years (list): Years to forecast

    Returns:
        pd.DataFrame: Simulated future input records
    """
    future_data = []
    for year in future_years:
        for _, row in base_df.iterrows():
            future_row = row.copy()
            future_row['year'] = year

            # Simulate policy impact (e.g., increase renewables, reduce fossil fuel use)
            if 'renewable_share' in future_row:
                future_row['renewable_share'] = min(future_row['renewable_share'] * 1.1, 100)
            if 'fossil_fuel_pct' in future_row:
                future_row['fossil_fuel_pct'] *= 0.9
            if 'energy_use' in future_row:
                future_row['energy_use'] *= 1.05  # assume more energy needed
            future_data.append(future_row)

    return pd.DataFrame(future_data)

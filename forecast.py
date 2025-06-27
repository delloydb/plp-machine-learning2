"""
forecast.py
Uses the trained model to make future carbon emission predictions.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_model(model_path='trained_model.pkl'):
    """
    Loads a pre-trained model from disk.

    Args:
        model_path (str): Path to the saved model file

    Returns:
        model: Loaded model
    """
    try:
        model = joblib.load(model_path)
        print(f"[INFO] Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load model: {e}")
        return None

def prepare_future_data(future_df, reference_df):
    """
    Prepares simulated future data for prediction by aligning features and scaling.

    Args:
        future_df (pd.DataFrame): Simulated future input data
        reference_df (pd.DataFrame): Original reference data used in training

    Returns:
        np.ndarray: Scaled feature matrix for prediction
    """
    # Match columns using training set reference
    future_processed = pd.get_dummies(future_df)
    reference_processed = pd.get_dummies(reference_df.drop(columns=['co2_emissions']))

    # Align columns (add missing, reorder)
    future_aligned = future_processed.reindex(columns=reference_processed.columns, fill_value=0)

    # Scale features using reference scaler
    scaler = StandardScaler()
    scaler.fit(reference_processed)
    future_scaled = scaler.transform(future_aligned)

    return future_scaled

def forecast_emissions(model, future_data_scaled, original_future_df):
    """
    Predict carbon emissions based on future data.

    Args:
        model: Trained model
        future_data_scaled: Preprocessed input features
        original_future_df: Raw input with year and country info

    Returns:
        pd.DataFrame: Output with predictions
    """
    predictions = model.predict(future_data_scaled)
    results_df = original_future_df.copy()
    results_df['predicted_co2_emissions'] = predictions
    return results_df

"""
data_loader.py
Loads, cleans, and preprocesses carbon emissions dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """
    Loads the dataset from the given file path.

    Args:
        filepath (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        print(f"[INFO] Loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        return None

def preprocess_data(df, target_col='co2_emissions'):
    """
    Cleans and prepares the data for modeling.

    Args:
        df (pd.DataFrame): Raw DataFrame
        target_col (str): Column to predict

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = df.copy()

    # Drop missing values (can later enhance with imputation)
    df = df.dropna()

    # Optional: Filter or normalize country names, if needed
    if 'country' in df.columns:
        df = pd.get_dummies(df, columns=['country'], drop_first=True)

    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, X.columns.tolist()

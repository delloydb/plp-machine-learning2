"""
model.py
Train and save a regression model for carbon emissions forecasting.
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

def train_model(X_train, y_train, use_grid_search=False):
    """
    Trains a Random Forest regression model.

    Args:
        X_train: Feature training data
        y_train: Target training data
        use_grid_search (bool): Whether to perform hyperparameter tuning

    Returns:
        model: Trained model
    """
    if use_grid_search:
        print("[INFO] Performing hyperparameter tuning...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='r2')
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"[INFO] Best parameters: {grid_search.best_params_}")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        print("[INFO] Model trained using default RandomForestRegressor.")

    return model

def save_model(model, filename='trained_model.pkl'):
    """
    Saves the trained model to disk using joblib.

    Args:
        model: Trained model
        filename (str): Output file path
    """
    joblib.dump(model, filename)
    print(f"[INFO] Model saved to {filename}")

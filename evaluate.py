"""
evaluate.py
Evaluate and visualize the performance of the trained model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and print performance metrics.

    Args:
        model: Trained regression model
        X_test: Feature test data
        y_test: Target test data

    Returns:
        dict: Dictionary of evaluation metrics
    """
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n[INFO] Model Evaluation Results:")
    print(f"MAE  (Mean Absolute Error) : {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"R²   (Coefficient of Determination): {r2:.4f}")

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def plot_predictions(y_test, predictions):
    """
    Plot actual vs. predicted emissions.

    Args:
        y_test: True values
        predictions: Predicted values
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    plt.xlabel("Actual Emissions")
    plt.ylabel("Predicted Emissions")
    plt.title("Actual vs Predicted Carbon Emissions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance from the trained model.

    Args:
        model: Trained Random Forest model
        feature_names (list): List of feature names
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[sorted_idx], y=[feature_names[i] for i in sorted_idx], palette="coolwarm")
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
    else:
        print("[WARNING] Model does not support feature importance.")

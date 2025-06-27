"""
main.py
End-to-end pipeline to train, evaluate, and forecast carbon emissions using ML.
"""

from data_loader import load_data, preprocess_data
from utils import summarize_dataset, simulate_future_inputs
from model import train_model, save_model
from evaluate import evaluate_model, plot_predictions, plot_feature_importance
from forecast import load_model, prepare_future_data, forecast_emissions

# === Step 1: Load Dataset ===
data_path = "data/emissions.csv"  # Update this to your actual dataset path
df = load_data(data_path)
summarize_dataset(df)

# === Step 2: Preprocess Data ===
X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

# === Step 3: Train Model ===
model = train_model(X_train, y_train, use_grid_search=False)
save_model(model, "trained_model.pkl")

# === Step 4: Evaluate Model ===
metrics = evaluate_model(model, X_test, y_test)
plot_predictions(y_test, model.predict(X_test))
plot_feature_importance(model, feature_names)

# === Step 5: Simulate Future Scenarios ===
print("\n[INFO] Simulating future scenarios for forecasting...")
future_inputs = simulate_future_inputs(df[df['year'] == df['year'].max()].head(5), future_years=[2025, 2030])
future_scaled = prepare_future_data(future_inputs, df)
model_loaded = load_model("trained_model.pkl")
future_predictions = forecast_emissions(model_loaded, future_scaled, future_inputs)

print("\n[INFO] Future Emissions Forecast:")
print(future_predictions[['country', 'year', 'predicted_co2_emissions']])

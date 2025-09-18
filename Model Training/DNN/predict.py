# predict.py

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import argparse

# --- Configuration ---
MODEL_PATH = 'best_ids_model.keras'
SCALER_PATH = 'scaler.joblib'
ENCODER_PATH = 'label_encoder.joblib'

# --- 1. Load Artifacts ---
print("Loading model and preprocessing objects...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    print(f"Error loading files: {e}")
    print("Please make sure you have run train.py first to generate the necessary files.")
    exit()

# --- 2. Create a command-line interface to get the input file ---
parser = argparse.ArgumentParser(description="Predict network intrusion from a CSV file.")
parser.add_argument(
    '--file',
    type=str,
    required=True,
    help='Path to the CSV file with data for prediction.'
)
args = parser.parse_args()

# --- 3. Load and Preprocess New Data ---
print(f"Loading data from {args.file}...")
try:
    new_data = pd.read_csv(args.file)
except FileNotFoundError:
    print(f"Error: The file '{args.file}' was not found.")
    exit()

# Store original labels if they exist for comparison
original_labels = None
if 'Label' in new_data.columns:
    original_labels = new_data['Label']
    new_data = new_data.drop('Label', axis=1)

# Clean column names to match training
new_data.columns = new_data.columns.str.strip().str.replace(' ', '_')

# Apply the loaded scaler
print("Scaling new data...")
new_data_scaled = scaler.transform(new_data)

# --- 4. Make Predictions ---
print("Making predictions...")
predictions_prob = model.predict(new_data_scaled)
predictions_encoded = np.argmax(predictions_prob, axis=1)

# Decode the predictions to get original labels
predicted_labels = label_encoder.inverse_transform(predictions_encoded)

# --- 5. Display Results ---
print("\n--- Prediction Results ---")
results_df = pd.DataFrame({'Predicted_Label': predicted_labels})

# If we had original labels, we can show a comparison
if original_labels is not None:
    results_df['Original_Label'] = original_labels
    correct_predictions = (results_df['Predicted_Label'] == results_df['Original_Label']).sum()
    accuracy = (correct_predictions / len(results_df)) * 100
    print(f"\nAccuracy on provided file: {accuracy:.2f}%")

print("\nFirst 10 predictions:")
print(results_df.head(10))

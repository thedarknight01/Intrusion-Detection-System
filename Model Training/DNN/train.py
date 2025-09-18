# train.py

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import joblib

# --- Configuration ---
DATA_PATH = "/kaggle/input/csecicids2018-cleaned/cleaned_ids2018_sampled.csv"
MODEL_SAVE_PATH = 'best_ids_model.keras'
SCALER_SAVE_PATH = 'scaler.joblib'
ENCODER_SAVE_PATH = 'label_encoder.joblib'

# --- 1. Load and Prepare Data ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)

print("Preprocessing data...")
# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_')

# Separate features (X) and labels (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Important for imbalanced datasets
)

# --- 2. Scale Features ---
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Encode Labels ---
print("Encoding labels...")
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

num_classes = y_train_categorical.shape[1]
print(f"Found {num_classes} classes.")

# --- 4. Handle Class Imbalance ---
print("Calculating class weights...")
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)
class_weight_dict = dict(enumerate(class_weights))


# --- 5. Build the Model ---
print("Building the model...")
model = Sequential([
    Dense(256, input_shape=(X_train_scaled.shape[1],)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),

    Dense(128),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    Dense(64),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.summary()

# --- 6. Compile the Model ---
print("Compiling the model...")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

# --- 7. Set up Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

model_checkpoint = ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_loss',
    save_best_only=True
)

# --- 8. Train the Model ---
print("Starting training...")
history = model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=100,
    batch_size=256,
    validation_data=(X_test_scaled, y_test_categorical),
    class_weight=class_weight_dict,
    callbacks=[early_stopping, model_checkpoint]
)
print("Training finished.")

# --- 9. Evaluate the Best Model ---
print("\nEvaluating the best model on the test set...")
# The best model is already loaded thanks to restore_best_weights=True
results = model.evaluate(X_test_scaled, y_test_categorical, verbose=1)

print("\nFinal Test Results:")
print(f"Loss: {results[0]:.4f}")
print(f"Accuracy: {results[1]:.4f}")
print(f"Precision: {results[2]:.4f}")
print(f"Recall: {results[3]:.4f}")


# --- 10. Save Preprocessing Objects ---
print(f"Saving scaler to {SCALER_SAVE_PATH}")
joblib.dump(scaler, SCALER_SAVE_PATH)

print(f"Saving label encoder to {ENCODER_SAVE_PATH}")
joblib.dump(label_encoder, ENCODER_SAVE_PATH)

print("\nTraining script finished successfully!")

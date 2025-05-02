import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# === Config ===
CSV_PATH = "data/gestures.csv"
MODEL_DIR = "model"
MODEL_NAME = "gestures"
INPUT_SIZE = 63 * 2  # 21 landmarks per hand, 3 coords each, 2 hands

# === Load Data ===
print("Loading data...")
df = pd.read_csv(CSV_PATH)

# Assume columns: x1, y1, z1, ..., x42, y42, z42, label
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

# Save label classes for inference
os.makedirs(MODEL_DIR, exist_ok=True)
np.save(os.path.join(MODEL_DIR, "labels.npy"), le.classes_)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# === Build Model ===
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# === Train ===
history = model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), batch_size=32)

# === Save Models ===
model.save(os.path.join(MODEL_DIR, f"{MODEL_NAME}.h5"))

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(os.path.join(MODEL_DIR, f"{MODEL_NAME}.tflite"), 'wb') as f:
    f.write(tflite_model)

print("✅ Training complete. Model saved to:", MODEL_DIR)

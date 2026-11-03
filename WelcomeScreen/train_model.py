
import numpy as np
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split

# Load prepared data
X = np.load("signs_X.npy")
y = np.load("signs_y.npy")

with open("labels.json", "r", encoding="utf-8") as file:
    labels = json.load(file)

number_of_classes = len(labels)

if number_of_classes < 2:
    raise ValueError("You need at least two different signs to train!")

if len(X) < 10:
    raise ValueError("Collect more recordings before training.")

print("Training data:", X.shape)
print("Number of signs:", number_of_classes)
print("Signs:", labels)

# Show recordings per sign
print("Total recordings:", len(X))
print(
    "Recordings per sign:",
    {
        label: int(np.sum(y == i))
        for i, label in enumerate(labels)
    }
)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100, 63)),

    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(number_of_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Stop training when validation stops improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

# Train model
model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_val, y_val),
    shuffle=True,
    callbacks=[early_stop]
)

# Save model
model.save("sign_language_model.keras")

print("\nTraining complete!")
print("Model saved as sign_language_model.keras")

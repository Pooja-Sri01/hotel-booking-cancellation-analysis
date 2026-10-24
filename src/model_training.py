"""
Deep Learning Model Architecture and Training Module.
Course: Exploratory Data Analysis and Visualization (U21ADP05)
Institution: KPR Institute of Engineering and Technology
"""

import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def build_mlp_model(input_dim: int, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Constructs the Deep Learning Multilayer Perceptron (MLP) architecture:
    - Input layer: input_dim features
    - Dense: 128 neurons, ReLU activation
    - Dropout: 0.30
    - Dense: 64 neurons, ReLU activation
    - Dropout: 0.20
    - Dense: 32 neurons, ReLU activation
    - Output: 1 neuron, Sigmoid activation (Binary Classification)
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="input_layer"),
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.Dropout(0.30, name="dropout_1"),
        layers.Dense(64, activation="relu", name="dense_64"),
        layers.Dropout(0.20, name="dropout_2"),
        layers.Dense(32, activation="relu", name="dense_32"),
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ], name="Hotel_Cancellation_MLP")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int = 256,
    max_epochs: int = 30,
    learning_rate: float = 0.001,
    use_class_weights: bool = True,
    output_dir: str = OUTPUTS_DIR,
) -> tuple[tf.keras.Model, dict, dict]:
    """
    Trains the MLP model with early stopping, restores best weights,
    and handles class imbalance using balanced class weights.
    """
    input_dim = X_train.shape[1]
    model = build_mlp_model(input_dim, learning_rate=learning_rate)

    print("\n" + "=" * 60)
    print("      DEEP LEARNING MODEL ARCHITECTURE (MLP)")
    print("=" * 60)
    model.summary()
    print("=" * 60 + "\n")

    # Class Weight Computation
    if use_class_weights:
        weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight_dict = {int(cls): float(w) for cls, w in zip(np.unique(y_train), weights)}
        print(f"[*] Computed Balanced Class Weights: {class_weight_dict}")
    else:
        class_weight_dict = None

    # Callbacks: Early Stopping on validation loss with weight restoration
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    print(f"[*] Training MLP model (max {max_epochs} epochs, batch size {batch_size})...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1,
    )

    # Save model in Keras native format
    model_path = os.path.join(output_dir, "hotel_cancellation_model.keras")
    model.save(model_path)
    print(f"[+] Model weights saved to '{model_path}'.")

    # Save history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    hist_dict["epochs_trained"] = len(history.history["loss"])
    hist_dict["stopped_epoch"] = int(early_stop.stopped_epoch) if early_stop.stopped_epoch > 0 else len(history.history["loss"])

    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=4)

    training_summary = {
        "architecture": "MLP (128-Dropout(0.3)-64-Dropout(0.2)-32-Sigmoid)",
        "input_features": input_dim,
        "optimizer": "Adam",
        "initial_learning_rate": learning_rate,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "epochs_trained": hist_dict["epochs_trained"],
        "class_weights": class_weight_dict,
        "best_val_loss": round(float(min(hist_dict["val_loss"])), 4),
        "best_val_accuracy": round(float(max(hist_dict["val_accuracy"])), 4),
        "best_val_auc": round(float(max(hist_dict["val_auc"])), 4),
        "model_file": model_path,
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(training_summary, f, indent=4)

    return model, hist_dict, training_summary


if __name__ == "__main__":
    try:
        from src.data_loading import load_and_inspect_data
        from src.preprocessing import split_and_preprocess
    except ModuleNotFoundError:
        from data_loading import load_and_inspect_data
        from preprocessing import split_and_preprocess

    df, _ = load_and_inspect_data()
    data = split_and_preprocess(df)
    model, history, summary = train_model(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        max_epochs=15,  # quick test
    )
    print("Model training completed successfully!")

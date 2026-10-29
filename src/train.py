from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

def train_model(
    model,
    X_train,
    y_train,
    w_train,
    X_dev,
    y_dev,
    w_dev,
    epochs=20,
    batch_size=32,
    patience=3,
    model_path="models/best_pos_model.keras"
):
    """
    Train a POS tagging model and save the best model.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled POS tagging model.
    X_train, y_train : np.ndarray
        Training sequences and POS labels.
    w_train : np.ndarray
        Training token weights (1=token, 0=padding).
    X_dev, y_dev : np.ndarray
        Validation sequences and POS labels.
    w_dev : np.ndarray
        Validation token weights.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Number of sentences per training batch.
    patience : int
        Epochs without improvement before stopping.
    model_path : str
        Path where the best model will be saved.

    Returns
    -------
    tf.keras.callbacks.History
        Training history.
    """

    # Check input shapes
    assert X_train.shape == y_train.shape == w_train.shape
    assert X_dev.shape == y_dev.shape == w_dev.shape

    # Create directory for saved models
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    history = model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        validation_data=(X_dev, y_dev, w_dev),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return history



def plot_training_history(history):
    """
    Plot training and validation loss and accuracy.
    """

    metrics = history.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(metrics["loss"], label="Training loss")
    axes[0].plot(metrics["val_loss"], label="Validation loss")
    axes[0].set_title("Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(
        metrics["accuracy"],
        label="Training accuracy"
    )
    axes[1].plot(
        metrics["val_accuracy"],
        label="Validation accuracy"
    )
    axes[1].set_title("Model Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
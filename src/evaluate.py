
import numpy as np


def evaluate_model(model, X, y, weights, batch_size=32):
    """
    Evaluate POS tagging accuracy, excluding padding.

    Parameters
    ----------
    model : tf.keras.Model
        Trained POS tagging model.
    X : np.ndarray
        Input token IDs, shape (samples, seq_length).
    y : np.ndarray
        True POS tag IDs.
    weights : np.ndarray
        Token mask (1=real token, 0=padding).
    batch_size : int
        Prediction batch size.

    Returns
    -------
    dict
        Accuracy and token counts.
    """

    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)

    if X.shape != y.shape or X.shape != weights.shape:
        raise ValueError("X, y and weights must have the same shape.")

    # Predict probability distribution for each token
    probabilities = model.predict(
        X,
        batch_size=batch_size,
        verbose=0
    )

    # Select most probable POS tag
    predictions = np.argmax(probabilities, axis=-1)

    # Ignore padding positions
    mask = weights > 0

    total_tokens = int(np.sum(mask))

    if total_tokens == 0:
        raise ValueError("No valid tokens found.")

    correct_tokens = int(
        np.sum((predictions == y) & mask)
    )

    accuracy = correct_tokens / total_tokens

    results = {
        "accuracy": accuracy,
        "correct_tokens": correct_tokens,
        "total_tokens": total_tokens
    }

    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Correct tokens: {correct_tokens}")
    print(f"Total tokens: {total_tokens}")

    return results

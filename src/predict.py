
import numpy as np
import tensorflow as tf


def predict_pos(model, tokens, vectorizer, id2tag):
    """
    Predict POS tags for a tokenized sentence.

    Parameters
    ----------
    model : tf.keras.Model
        Trained POS tagging model.
    tokens : list[str]
        Tokenized input sentence.
    vectorizer : tf.keras.layers.TextVectorization
        Fitted training vectorizer.
    id2tag : dict
        Mapping from tag IDs to UPOS labels.

    Returns
    -------
    list[tuple[str, str]]
        Pairs of (token, predicted POS tag).
    """

    if not tokens:
        raise ValueError("The input sentence cannot be empty.")

    if not all(isinstance(token, str) and token for token in tokens):
        raise ValueError("Tokens must be non-empty strings.")

    if any(any(char.isspace() for char in token) for token in tokens):
        raise ValueError("Tokens cannot contain whitespace.")

    # Reconstruct text using the training convention
    text = " ".join(tokens)

    # Reuse fitted vectorizer
    X = vectorizer(tf.constant([text])).numpy()

    # Check for truncation
    if len(tokens) > X.shape[1]:
        raise ValueError(
            f"Sentence is too long. Maximum: {X.shape[1]} tokens."
        )

    # Predict POS probabilities
    probabilities = model.predict(X, verbose=0)

    # Select most probable label for each position
    predicted_ids = np.argmax(probabilities, axis=-1)[0]

    # Remove padding predictions
    predicted_ids = predicted_ids[:len(tokens)]

    # Convert numerical IDs to UPOS labels
    predicted_tags = [
        id2tag[int(tag_id)]
        for tag_id in predicted_ids
    ]

    return list(zip(tokens, predicted_tags))

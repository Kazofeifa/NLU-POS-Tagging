
from pathlib import Path
import re

import tensorflow as tf

from predict import predict_pos
from pre_save import load_preprocessing


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = "en_ewt"

MODEL_PATH = BASE_DIR / "models" / "en_ewt_bilstm.keras"
PREPROCESSING_PATH = (
    BASE_DIR / "models" / DATASET / "preprocessing"
)


def tokenize_sentence(sentence):
    """
    Simple tokenizer that separates punctuation.
    """
    return re.findall(
        r"\w+(?:['’]\w+)*|[^\w\s]",
        sentence
    )


def main():
    print("\n=== POS TAGGING ===")

    # Load model and preprocessing
    print("Loading model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    vectorizer, tag2id, id2tag = load_preprocessing(
        PREPROCESSING_PATH
    )

    print("Model loaded successfully!")

    # Interactive prediction loop
    while True:
        sentence = input(
            "\nEnter a sentence (or 'exit' to quit): "
        ).strip()

        if sentence.lower() == "exit":
            break

        if not sentence:
            print("Please enter a sentence.")
            continue

        # Tokenize sentence
        tokens = tokenize_sentence(sentence)

        # Predict POS tags
        try:
            predictions = predict_pos(
                model=model,
                tokens=tokens,
                vectorizer=vectorizer,
                id2tag=id2tag
            )
        except ValueError as error:
            print(f"Prediction error: {error}")
            continue

        # Display results
        print("\nWORD            POS TAG")
        print("-" * 25)

        for word, tag in predictions:
            print(f"{word:<15} {tag}")


if __name__ == "__main__":
    main()

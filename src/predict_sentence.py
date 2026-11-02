
from pathlib import Path
import re

import tensorflow as tf

from predict import predict_pos
from pre_save import load_preprocessing


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def get_available_datasets():
    """
    Find available datasets from saved BiLSTM models.
    """
    datasets = {}

    for model_path in sorted(MODELS_DIR.glob("*_bilstm_dropout.keras")):
        dataset = model_path.name.removesuffix("_bilstm_dropout.keras")

        preprocessing_path = (
            MODELS_DIR / dataset / "preprocessing"
        )

        required_files = [
            "vocabulary.json",
            "tags.json",
            "vectorizer_config.json"
        ]

        if all(
            (preprocessing_path / filename).exists()
            for filename in required_files
        ):
            datasets[dataset] = model_path

    return datasets


def select_dataset(datasets):
    """
    Let the user select a dataset.
    """
    names = list(datasets.keys())

    print("\n=== SELECT LANGUAGE ===")

    for i, name in enumerate(names, start=1):
        print(f"{i}. {name}")

    while True:
        choice = input("\nChoose an option: ").strip()

        if choice.isdigit():
            index = int(choice) - 1

            if 0 <= index < len(names):
                return names[index]

        print("Invalid option. Try again.")


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

    # Discover available datasets
    datasets = get_available_datasets()

    if not datasets:
        print("No complete saved models were found.")
        print(f"Check directory: {MODELS_DIR}")
        return

    # Select language
    dataset = select_dataset(datasets)

    model_path = datasets[dataset]
    preprocessing_path = (
        MODELS_DIR / dataset / "preprocessing"
    )

    # Load model
    print(f"\nLoading model: {model_path.name}")

    model = tf.keras.models.load_model(model_path)

    vectorizer, tag2id, id2tag = load_preprocessing(
        preprocessing_path
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

        tokens = tokenize_sentence(sentence)

        try:
            predictions = predict_pos(
                model=model,
                tokens=tokens,
                vectorizer=vectorizer,
                id2tag=id2tag
            )

        except (ValueError, KeyError) as error:
            print(f"Prediction error: {error}")
            continue

        print("\nWORD            POS TAG")
        print("-" * 25)

        for word, tag in predictions:
            print(f"{word:<15} {tag}")


if __name__ == "__main__":
    main()


import json
from pathlib import Path

import tensorflow as tf

# Save the fitted vocabulary, POS tags and TextVectorization configuration
def save_preprocessing(preprocessor, save_dir):
    """
    Save the fitted vocabulary, POS tags and
    TextVectorization configuration.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = preprocessor.vectorizer

    # Save vocabulary
    vocabulary = vectorizer.get_vocabulary()

    with open(save_dir / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)

    # Save UPOS mapping
    with open(save_dir / "tags.json", "w", encoding="utf-8") as f:
        json.dump(preprocessor.tag2id, f, ensure_ascii=False, indent=2)

    # Save vectorizer configuration
    with open(save_dir / "vectorizer_config.json", "w", encoding="utf-8") as f:
        json.dump(vectorizer.get_config(), f, ensure_ascii=False, indent=2)

    print(f"Preprocessing saved in: {save_dir}")


#Loads the fitted vocabulary, POS tags and TextVectorization configuration
def load_preprocessing(save_dir):
    
    save_dir = Path(save_dir)

    with open(save_dir / "vocabulary.json", encoding="utf-8") as f:
        vocabulary = json.load(f)

    with open(save_dir / "tags.json", encoding="utf-8") as f:
        tag2id = json.load(f)

    with open(save_dir / "vectorizer_config.json", encoding="utf-8") as f:
        config = json.load(f)

    # Restore vectorizer
    vectorizer = tf.keras.layers.TextVectorization.from_config(config)

    # The first two IDs are reserved for padding and OOV
    vectorizer.set_vocabulary(vocabulary[2:])

    if vectorizer.get_vocabulary() != vocabulary:
        raise ValueError("Restored vocabulary does not match saved vocabulary.")

    # Convert label IDs back to integers
    tag2id = {tag: int(idx) for tag, idx in tag2id.items()}
    id2tag = {idx: tag for tag, idx in tag2id.items()}

    return vectorizer, tag2id, id2tag

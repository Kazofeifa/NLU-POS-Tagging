import numpy as np


def validate_data(X, y, sentences, tags):

    # Check that inputs and labels have the same shape
    assert X.shape == y.shape
    assert len(X) == len(sentences)
    assert len(y) == len(tags)

    # Check that every word has a POS tag
    for words, labels in zip(sentences, tags):
        assert len(words) == len(labels)

    # Create a mask to ignore padding
    mask = (X != 0).astype("float32")

    # Check that the number of words is correct
    total_words = sum(len(sentence) for sentence in sentences)

    assert mask.sum() == total_words

    print("Validation successful!")

    return mask
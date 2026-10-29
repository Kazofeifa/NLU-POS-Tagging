
import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import pad_sequences


# Convert words into numbers
def preprocess_text(train_sentences, dev_sentences, test_sentences):

    # Convert word lists into sentences
    train_texts = [" ".join(sentence) for sentence in train_sentences]
    dev_texts = [" ".join(sentence) for sentence in dev_sentences]
    test_texts = [" ".join(sentence) for sentence in test_sentences]

    # Making sure that words are not split incorrectly
    for sentences in [train_sentences, dev_sentences, test_sentences]:
        for sentence in sentences:
            if len(sentence) != len(" ".join(sentence).split()):
                raise ValueError("A sentence contains tokens with spaces")

    # Finding the longest sentence
    max_length = max(
        len(sentence)
        for dataset in [train_sentences, dev_sentences, test_sentences]
        for sentence in dataset
    )

    # TextVectorization
    vectorizer = TextVectorization(
        max_tokens=20000,
        standardize=None,
        output_mode="int",
        output_sequence_length=max_length
    )

    # Learning the vocabulary from training data
    vectorizer.adapt(train_texts)

    # Converting sentences into numbers
    X_train = vectorizer(np.array(train_texts)).numpy()
    X_dev = vectorizer(np.array(dev_texts)).numpy()
    X_test = vectorizer(np.array(test_texts)).numpy()

    return X_train, X_dev, X_test, vectorizer, max_length


# Convert POS tags into numbers
def preprocess_tags(train_tags, dev_tags, test_tags, max_length):

    # Get all POS categories
    unique_tags = sorted(set(
        tag for sentence in train_tags for tag in sentence
    ))

    # Create dictionaries
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}

    # Convert tags to numbers
    y_train = [[tag2id[tag] for tag in sentence] for sentence in train_tags]
    y_dev = [[tag2id[tag] for tag in sentence] for sentence in dev_tags]
    y_test = [[tag2id[tag] for tag in sentence] for sentence in test_tags]

    # Use of padding to make all sequences the same length
    y_train = pad_sequences(y_train, maxlen=max_length, padding="post")
    y_dev = pad_sequences(y_dev, maxlen=max_length, padding="post")
    y_test = pad_sequences(y_test, maxlen=max_length, padding="post")

    return y_train, y_dev, y_test, tag2id, id2tag


import numpy as np
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.utils import pad_sequences


class POSPreprocessor:

    def __init__(self, max_tokens=20000):
        self.max_tokens = max_tokens
        self.vectorizer = None
        self.max_length = None
        self.tag2id = {}
        self.id2tag = {}

    # Learn vocabulary and POS categories
    def fit(self, train_sentences, train_tags):

        train_texts = [" ".join(s) for s in train_sentences]

        self.max_length = max(len(s) for s in train_sentences)

        self.vectorizer = TextVectorization(
            max_tokens=self.max_tokens,
            standardize=None,
            split="whitespace",
            output_mode="int",
            output_sequence_length=self.max_length
        )

        self.vectorizer.adapt(train_texts)

        unique_tags = sorted(set(
            tag for sentence in train_tags for tag in sentence
        ))

        self.tag2id = {tag: i for i, tag in enumerate(unique_tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

    # Convert sentences to numerical IDs
    def transform_text(self, sentences):

        texts = [" ".join(s) for s in sentences]

        for sentence, text in zip(sentences, texts):
            if len(text.split()) != len(sentence):
                raise ValueError("Tokenization changed the sentence length")

        return self.vectorizer(np.array(texts)).numpy()

    # Convert POS tags to numerical IDs
    def transform_tags(self, tags):

        encoded = [
            [self.tag2id[tag] for tag in sentence]
            for sentence in tags
        ]

        return pad_sequences(
            encoded,
            maxlen=self.max_length,
            padding="post",
            truncating="post"
        )

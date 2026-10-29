
import os
import tensorflow as tf
from conllu import parse_incr


# Download and load the dataset.
# parse_incr interprets the conllu format so delimiter="\t not requiered.

from pathlib import Path
from conllu import parse_incr


# Load dataset from the data folder
def load_dataset(filename):

    # Find the data folder
    project_path = Path(__file__).resolve().parent.parent
    path = project_path / "data" / filename

    sentences = []
    pos_tags = []

    with open(path, "r", encoding="utf-8") as file:

        # Extract sentences and POS tags
        for sentence in parse_incr(file):
            words = []
            tags = []

            for token in sentence:

                if not isinstance(token["id"], int):
                    continue

                words.append(token["form"])
                tags.append(token["upos"])

            if words:
                sentences.append(words)
                pos_tags.append(tags)

    return sentences, pos_tags

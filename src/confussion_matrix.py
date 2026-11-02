import numpy as np
import csv
from pathlib import Path

def generate_confusion_matrix(model, X, y, weights, id2tag, batch_size=32):
    """Compute a POS-tag confusion matrix, excluding padding.

    Rows are true tags, columns are predicted tags.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)

    if X.shape != y.shape or X.shape != weights.shape:
        raise ValueError("X, y and weights must have the same shape.")
    if not id2tag:
        raise ValueError("id2tag must contain at least one POS tag.")

    tag_ids = sorted(id2tag)
    tag_id_to_index = {tag_id: index for index, tag_id in enumerate(tag_ids)}
    tag_names = [id2tag[tag_id] for tag_id in tag_ids]

    probabilities = model.predict(X, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=-1)

    if predictions.shape != y.shape:
        raise ValueError("Model predictions must have the same shape as y.")

    not_padding = weights > 0
    if not np.any(not_padding):
        raise ValueError("No valid tokens found.")
    true_ids = y[not_padding]
    predicted_ids = predictions[not_padding]

    true_indices = np.array([tag_id_to_index[t] for t in true_ids], dtype=np.intp)
    predicted_indices = np.array([tag_id_to_index[t] for t in predicted_ids], dtype=np.intp)

    matrix = np.zeros((len(tag_ids), len(tag_ids)), dtype=np.int64)
    np.add.at(matrix, (true_indices, predicted_indices), 1)

    return matrix, tag_names

def print_confusion_matrix(matrix, tag_names):
    """Print the matrix with true tags as rows and predicted tags as columns."""
    label_width = max(
        [len("REAL / PRED")] + [len(name) for name in tag_names]
    )
    value_width = max(
        [1] + [len(str(int(value))) for value in matrix.flat]
    )
    cell_width = max(label_width, value_width)

    print(f"{'REAL / PRED':>{cell_width}}", end="  ")
    for name in tag_names:
        print(f"{name:>{cell_width}}", end="  ")
    print()

    for name, row in zip(tag_names, matrix):
        print(f"{name:>{cell_width}}", end="  ")
        for value in row:
            print(f"{int(value):>{cell_width}}", end="  ")
        print()

def save_confusion_matrix(matrix, tag_names, output_path):
    """Save a confusion matrix as a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["REAL / PRED"] + tag_names)

        for tag_name, row in zip(tag_names, matrix):
            csv_writer.writerow([tag_name] + [int(value) for value in row])

    return output_path
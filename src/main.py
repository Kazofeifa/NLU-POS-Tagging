from loader import load_dataset
from preprocessing_class import POSPreprocessor
from validation import validate_data
from model import build_lstm_model
from train import train_model
from evaluate import evaluate_model



# 1. Select dataset
dataset = "en_ewt"


# 2. Load datasets
train_sentences, train_tags = load_dataset(
    f"{dataset}-ud-train.conllu"
)

dev_sentences, dev_tags = load_dataset(
    f"{dataset}-ud-dev.conllu"
)

test_sentences, test_tags = load_dataset(
    f"{dataset}-ud-test.conllu"
)


# 3. Create preprocessing object
preprocessor = POSPreprocessor(max_tokens=20000)


# 4. Learn vocabulary and POS tags
preprocessor.fit(train_sentences, train_tags)


# 5. Preprocess input sentences
X_train = preprocessor.transform_text(train_sentences)
X_dev = preprocessor.transform_text(dev_sentences)
X_test = preprocessor.transform_text(test_sentences)


# 6. Preprocess POS labels
y_train = preprocessor.transform_tags(train_tags)
y_dev = preprocessor.transform_tags(dev_tags)
y_test = preprocessor.transform_tags(test_tags)


# 7. Validate data
w_train = validate_data(
    X_train, y_train, train_sentences, train_tags
)

w_dev = validate_data(
    X_dev, y_dev, dev_sentences, dev_tags
)

w_test = validate_data(
    X_test, y_test, test_sentences, test_tags
)


# 8. Show results
print("\nPreprocessing completed!")

print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_dev.shape, y_dev.shape)
print("Test:", X_test.shape, y_test.shape)

print("Vocabulary size:",
      len(preprocessor.vectorizer.get_vocabulary()))

print("POS categories:", preprocessor.tag2id)


#9. Models configuration
vocab_size = len(preprocessor.vectorizer.get_vocabulary())
num_tags = max(int(tag_id) for tag_id in preprocessor.tag2id.values()) + 1
experiments = [
    {"name": "lstm", "bidirectional": False, "dropout": 0.0},
    {"name": "bilstm", "bidirectional": True, "dropout": 0.0},
    {"name": "bilstm_dropout", "bidirectional": True, "dropout": 0.3},
]
results = {}

# 10. Build, train and save each model
for experiment in experiments:
    name = experiment["name"]
    print(f"\n{'=' * 60}")
    print(f"TRAINING: {name}")
    print(f"{'=' * 60}")

    model = build_lstm_model(
        vocab_size=vocab_size,
        num_tags=num_tags,
        embedding_dim=128,
        lstm_units=128,
        bidirectional=experiment["bidirectional"],
        dropout=experiment["dropout"],
        learning_rate=0.001,
    )
    model.summary()

    model_path = f"models/{dataset}_{name}.keras"
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        X_dev=X_dev,
        y_dev=y_dev,
        w_dev=w_dev,
        epochs=20,
        batch_size=32,
        patience=3,
        model_path=model_path,
    )

    # 11. Evaluate on development only (test remains untouched)
    print(f"\nTRAINING EVALUATION: {name}")
    train_metrics = evaluate_model(
            model=model,
            X=X_train,
            y=y_train,
            weights=w_train,
        )

    print(f"\nDEVELOPMENT EVALUATION: {name}")
    dev_metrics = evaluate_model(
        model=model,
        X=X_dev,
        y=y_dev,
        weights=w_dev,
    )
    results[name] = {
        "dev_accuracy": dev_metrics["accuracy"],
        "best_val_loss": min(history.history["val_loss"]),
        "model_path": model_path,
    }

print("\n=== DEVELOPMENT RESULTS ===")
for name, metrics in results.items():
    print(
        f"{name:<18} "
        f"val_accuracy={metrics['dev_accuracy']:.2%}  "
        f"best_val_loss={metrics['best_val_loss']:.4f}  "
        f"saved={metrics['model_path']}"
    )

# Select by validation loss, matching ModelCheckpoint monitoring.
best_name = min(results, key=lambda name: results[name]["best_val_loss"])
print(f"\nBest model according to val_loss: {best_name}")
print(f"Saved at: {results[best_name]['model_path']}")
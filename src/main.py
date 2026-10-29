
from src.loader import load_dataset
from src.preprocessing import preprocess_text, preprocess_tags
from src.validation import validate_data


# 1. Load datasets
train_sentences, train_tags = load_dataset("en_ewt-ud-train.conllu")
dev_sentences, dev_tags = load_dataset("en_ewt-ud-dev.conllu")
test_sentences, test_tags = load_dataset("en_ewt-ud-test.conllu")


# 2. Preprocess input sentences
X_train, X_dev, X_test, vectorizer, max_length = preprocess_text(
    train_sentences, dev_sentences, test_sentences
)


# 3. Preprocess POS labels
y_train, y_dev, y_test, tag2id, id2tag = preprocess_tags(
    train_tags, dev_tags, test_tags, max_length
)


# 4. Validate data
w_train = validate_data(X_train, y_train, train_sentences, train_tags)
w_dev = validate_data(X_dev, y_dev, dev_sentences, dev_tags)
w_test = validate_data(X_test, y_test, test_sentences, test_tags)


# 5. Show results
print("Training:", X_train.shape, y_train.shape)
print("Validation:", X_dev.shape, y_dev.shape)
print("Test:", X_test.shape, y_test.shape)

print("Vocabulary size:", len(vectorizer.get_vocabulary()))
print("POS categories:", tag2id)

from loader import load_dataset
from preprocessing_class import POSPreprocessor
from validation import validate_data




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
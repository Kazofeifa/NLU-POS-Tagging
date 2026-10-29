
import numpy as np
from src.model import build_lstm_model

# Example configuration
VOCAB_SIZE = 10000
NUM_TAGS = 17
MAX_LENGTH = 128

# 1. Create the model
model = build_lstm_model(
    vocab_size=VOCAB_SIZE,
    num_tags=NUM_TAGS,
    embedding_dim=128,
    lstm_units=128,
    bidirectional=False
)

print("\n=== MODEL ARCHITECTURE ===")
model.summary()

# 2. Create two example sentencesg
# 0 represents padding
X_test = np.array([
    [12, 45, 78, 92, 3] + [0] * (MAX_LENGTH - 5),
    [21, 34, 56, 0, 0] + [0] * (MAX_LENGTH - 5)
], dtype=np.int32)

# 3. Forward pass
predictions = model.predict(X_test, verbose=0)

print("\n=== OUTPUT SHAPE ===")
print("Input shape:", X_test.shape)
print("Output shape:", predictions.shape)

assert predictions.shape == (2, MAX_LENGTH, NUM_TAGS)

# 4. Check probabilities for real tokens
probabilities = predictions[0, 0]

print("\n=== FIRST TOKEN ===")
print("Predicted probabilities:", probabilities)
print("Sum of probabilities:", probabilities.sum())
print("Predicted tag ID:", np.argmax(probabilities))

assert np.allclose(probabilities.sum(), 1.0, atol=1e-5)

print("\nAll tests passed!")

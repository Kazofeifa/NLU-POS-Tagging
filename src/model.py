

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, TimeDistributed


def build_lstm_model(
    vocab_size,
    num_tags,
    embedding_dim=128,
    lstm_units=128,
    bidirectional=False,
    dropout=0.0,
    learning_rate=0.001
):
    #Input layer
    inputs = Input(
        shape=(None,),
        dtype="int32",
        name="input_tokens"
    )

    #Embaddings layer
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="embedding"
    )(inputs)

    # Recurrent layer
    lstm_layer = LSTM(
        units=lstm_units,
        return_sequences=True,
        dropout=dropout
    )

    if bidirectional:
        x = Bidirectional(
            lstm_layer,
            name="bilstm"
        )(x)
    else:
        x = lstm_layer(x)

    # Output layer
    outputs = Dense(
        units=num_tags,
        activation="softmax",
        name="tag_classifier"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="bilstm_pos" if bidirectional else "lstm_pos"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]
    )

    return model

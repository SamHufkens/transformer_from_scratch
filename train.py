import tensorflow as tf
from model_builder import create_model
import numpy as np
from tensorflow.keras.layers import TextVectorization

SEQUENCES_DATA_PATH = './data/movie_reviews.npy'
BATCH_SIZE = 64
EPOCHS = 5
VOCAB_SIZE = 50000
SEQ_LENGTH = 200

# Load sequences
sequences = np.load(SEQUENCES_DATA_PATH)

# Turn sequences into a tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices(sequences).batch(BATCH_SIZE).shuffle(1000)

# Create vocab
vectorize_layer = TextVectorization(
    standardize='lower',
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=SEQ_LENGTH + 1
)
vectorize_layer.adapt(dataset)
vocab = vectorize_layer.get_vocabulary()


# Create the train dataset with x and y
def create_train_dataset(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y
train_dataset = dataset.map(create_train_dataset)

# Build model
transformer_model = create_model()
transformer_model.compile("adam", loss=[tf.keras.losses.SparseCategoricalCrossentropy(), None])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.weights.h5",
    save_weights_only=True,
    save_freq="epoch",
    verbose=0,
)

# Train model
transformer_model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[model_checkpoint_callback]
)


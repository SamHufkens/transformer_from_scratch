from layers.layers import TransformerBlock, PositionalEmbedding
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow import int32

# Constants
VOCAB_SIZE = 50000
EMBEDDING_DIM = 768
MAX_LEN_UNIQUE_POS_EMBED = 1000
DENSE_1_NEURONS = 3072
HEADS = 12
DROP_RATE = 0.1

def create_model():

    # Input layer
    inputs = Input(shape=(None,), dtype=int32)

    # Positional Embedding
    out_pos = PositionalEmbedding(
                vocab_size=VOCAB_SIZE,
                embedding_dim=EMBEDDING_DIM,
                max_unique_pos_embed=MAX_LEN_UNIQUE_POS_EMBED
            )(inputs)

    # Transformer block
    out_tf = TransformerBlock(
                heads=HEADS,
                embedding_dim=EMBEDDING_DIM,
                dense_1_neurons=DENSE_1_NEURONS,
                dropout_rate=DROP_RATE
            )(out_pos)

    # Output layer
    outputs = Dense(VOCAB_SIZE, activation="softmax")(out_tf)

    transformer_model = Model(inputs=inputs, outputs=outputs)

    return transformer_model


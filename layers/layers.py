from tensorflow import shape, range, matmul, float32, cast, math, expand_dims, nn, reshape, transpose, bool, concat, tile, int32, constant
from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization, Dropout



class PositionalEmbedding(Layer):
  def __init__(self, vocab_size, embedding_dim, max_unique_pos_embed, **kwargs):
    super(PositionalEmbedding, self).__init__(**kwargs)
    self.vocab_size = vocab_size
    self.max_unique_pos_embed = max_unique_pos_embed
    self.embedding_dim = embedding_dim
    self.token_embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
    self.position_embedding_layer = Embedding(input_dim=max_unique_pos_embed, output_dim=embedding_dim)

  def call(self, x):

    maxlen = shape(x)[-1]

    positions = range(start=0, limit=maxlen, delta=1)
    positions = self.position_embedding_layer(positions)

    x = self.token_embedding_layer(x)

    output = x + positions

    return output

class SelfAttention(Layer):
  def __init__(self, **kwargs):
    super(SelfAttention, self).__init__(**kwargs)

  def call(self, query, keys, values, keys_dim, mask=None):

    # Calculate scores
    scores = matmul(query, keys, transpose_b=True) / math.sqrt(cast(keys_dim, float32))

    # Prevent attention to future tokens
    if mask is not None:
      mask = cast(mask, dtype=float32)

      # Expand mask shape to [batch_size, 1, seq_len, seq_len]
      mask = expand_dims(mask, axis=1)
      scores += (mask * -1e9)

    # Calculate weights with softmax
    attention_weights = nn.softmax(scores, axis=-1)

    # Calculate the output
    output = matmul(attention_weights, values)

    return output

class MultiHeadAttention(Layer):
  def __init__(self, heads, embedding_dim, **kwargs):
    super(MultiHeadAttention, self).__init__(**kwargs)
    self.heads = heads
    self.embedding_dim = embedding_dim
    self.self_attention = SelfAttention()
    self.W_q = Dense(embedding_dim)
    self.W_k = Dense(embedding_dim)
    self.W_v = Dense(embedding_dim)
    self.W_o = Dense(embedding_dim)

  def split_heads(self, x, batch_size):
    x = reshape(x, (batch_size, -1, self.heads, self.embedding_dim // self.heads))
    return transpose(x, perm=[0, 2, 1, 3])

  def concatenate_heads(self, attention, batch_size):
    attention = transpose(attention, perm=[0, 2, 1, 3])
    return reshape(attention, (batch_size, -1, self.heads * (self.embedding_dim // self.heads)))


  def call(self, queries, keys, values, mask=None):

    # Extract batch size (32)
    batch_size = shape(queries)[0]

    # Split the queries, keys, values from
    # (32, 200, 768) -> (32, 12, 200, 64)
    Q = self.split_heads(self.W_q(queries), batch_size)
    K = self.split_heads(self.W_k(keys), batch_size)
    V = self.split_heads(self.W_v(values), batch_size)

    # Apply attention to all 12 heads
    attention = self.self_attention(Q, K, V, keys_dim=self.embedding_dim // self.heads, mask=mask)

    # Concatenate all heads together
    # (32, 12, 200, 64) -> (32, 200, 768)
    concatenated_attention = self.concatenate_heads(attention, batch_size)

    # Calculate a last linear transformation to get the output
    output = self.W_o(concatenated_attention)

    return output


class TransformerBlock(Layer):
  def __init__(self, heads, embedding_dim, dense_1_neurons, dropout_rate, **kwargs):
    super(TransformerBlock, self).__init__(**kwargs)

    self.dropout_rate = dropout_rate

    # Multi-head attention layer
    self.multi_head_attention = MultiHeadAttention(heads, embedding_dim)

    # FFN layers
    self.dense_1 = Dense(dense_1_neurons, activation='relu')
    self.dense_2 = Dense(embedding_dim)

    # Normalization layers
    self.layer_norm_1 = LayerNormalization(epsilon=1e-6)
    self.layer_norm_2 = LayerNormalization(epsilon=1e-6)

    # Dropout layers
    self.dropout_1 = Dropout(self.dropout_rate)
    self.dropout_2 = Dropout(self.dropout_rate)

  def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
    i = range(n_dest)[:, None]
    j = range(n_src)
    m = i >= j - n_src + n_dest
    mask = cast(m, dtype)
    mask = reshape(mask, [1, n_dest, n_src])
    mult = concat(
        [expand_dims(batch_size, -1), constant([1, 1], dtype=int32)], 0
    )
    return tile(mask, mult)


  def call(self, x):

    input_shape = shape(x)
    batch_size = input_shape[0]
    seq_len = input_shape[1]
    causal_mask = self.causal_attention_mask(
        batch_size, seq_len, seq_len, bool
    )

    # Multi-Head Attention
    attention = self.multi_head_attention(x, x, x, mask=causal_mask)

    # Dropout
    out_drop_1 = self.dropout_1(attention)

    # Residual connection + Layer normalization
    res_1 = x + out_drop_1
    out_ln_1 = self.layer_norm_1(res_1)

    # FFN layers
    out_dense_1 = self.dense_1(out_ln_1)
    out_dense_2 = self.dense_2(out_dense_1)

    # Dropout
    out_drop_2 = self.dropout_2(out_dense_2)

    # Residual connection + Layer normalization
    res_2 = out_ln_1 + out_drop_2
    output = self.layer_norm_2(res_2)

    return output


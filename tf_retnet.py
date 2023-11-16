import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, LayerNormalization, GroupNormalization, Dense, SimpleRNNCell, RNN, LSTM, Bidirectional, LSTMCell
import torch
from itertools import repeat

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, SimpleRNNCell, RNN, LSTM, Bidirectional, LSTMCell


import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, LayerNormalization, GroupNormalization, Dense, SimpleRNNCell, RNN, LSTM, Bidirectional, LSTMCell
import torch
from itertools import repeat

import tensorflow as tf
from tensorflow.keras.layers import Layer, LayerNormalization, Dense, SimpleRNNCell, RNN, LSTM, Bidirectional, LSTMCell

class RecurrentRetention(tf.compat.v1.nn.rnn_cell.RNNCell):
    def __init__(self,
                 input_size,
                 hidden_size=32,
                 gamma=0.984375,
                 trainable=True,
                 dtype=None,
                 **kwargs):
        super(RecurrentRetention, self).__init__(trainable=trainable,
                                        dtype=dtype,
                                        **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r_layers = {
            'Q' : Dense(hidden_size),
            'K' : Dense(hidden_size),
            'V' : Dense(hidden_size)
        }

        self.gamma = tf.Variable(gamma, trainable=True)

    @property
    def state_size(self):
        return tf.TensorShape([self.hidden_size, self.hidden_size])

    @property
    def output_size(self):
        return self.hidden_size

    def call(self, inputs, state):
      q, k, v = [tf.cast(f(inputs), tf.float32) for f in self.r_layers.values()]
      s = self.gamma*state + tf.linalg.matmul(k, v, transpose_a=True)#tf.transpose(k, perm=[1, 0])@v
      x = tf.einsum('bi, bzk -> bk', q, s)
      return x, s


class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1, activation='gelu'):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation=activation),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])

  def call(self, x):
    return self.seq(x)


class Retention(Layer):
    def __init__(self, dim=128, gamma = 0.9865, **kwargs):
        super().__init__()
        _dense_kwargs = {
                "use_bias" : True,
                "dtype" : "float32"
                }
        self._qkv_layers = [Dense(dim, **_dense_kwargs),
                            Dense(dim, **_dense_kwargs),
                            Dense(dim, **_dense_kwargs)]
        self.gamma = gamma

    def call(self, x, training=False):
        Q, K, V = [f(z) for f, z in zip(self._qkv_layers, x)]
        _, s, d = Q.shape
        #b = b if b is not None else 1
        x = Q@tf.transpose(K, perm=[0, 2, 1])
        x /= d**0.5 #Normalization Trick 1
        D = self._compute_decay(s, self.gamma)
        D /= tf.reduce_sum(D, -1)**0.5 #Normalization Trick 2
        #D = tf.stack([*repeat(D, b)])
        x = x*D
        _norm_3 = lambda xs: xs/tf.maximum(tf.abs(tf.math.reduce_sum(xs, -1)), 1)
        x = tf.vectorized_map(_norm_3, x) #Normalization Trick 3
        x = x@V
        return x

    def _compute_decay(self, seq_len, gamma = 0.96875):
        _indices = list(range(0, seq_len))
        _decay_factors = [gamma**(i-j) if i>=j else 0 for i in _indices for j in _indices]
        D = tf.reshape(tf.convert_to_tensor(_decay_factors, dtype='float32'), (seq_len, seq_len))
        #mask = tf.eye(seq_len, dtype=tf.float32)
        #mask += tf.roll(mask, shift=-1, axis=-1)+tf.roll(mask, shift=-2, axis=-1)
        return D#*mask

class MultiScaleRetention(Layer):
    def __init__(self, dim, hdim=100, retention_layer=Retention, **kwargs):
      super(MultiScaleRetention, self).__init__()
      gamma = 1 - (2 ** (-5 - torch.arange(0, dim//hdim).float()))
      gamma = gamma.numpy().tolist()
      self.dim = dim
      self.hdim = hdim
      self.heads = [ChunkwiseRetention(hdim, gamma=gamma[head], **kwargs) for head in range(dim // hdim)]
      self.gn = GroupNormalization(dim//hdim, scale=False)
      self.wg = Sequential([
            Dense(dim, use_bias=True, activation = 'swish', **kwargs),
        ])
      self.wo = Dense(dim, use_bias=True, **kwargs)

    def call(self, q, k, v):
      W = self.wg(q)
      #q, k, v = list(map(lambda val: tf.split(val, self.dim//self.hdim, 2), x))
      x = [headi(q) for headi in self.heads]
      x = tf.concat(x, -1)
      Y = self.gn(x)
      x = self.wo(W*Y)
      return x

class RetentionEncoder(Layer):
    def __init__(self, dim=540, hdim=100, retention_layer=Retention, **kwargs):
        super().__init__()
        self.layer_norm = LayerNormalization()
        self.msr = MultiScaleRetention(dim, hdim, retention_layer=retention_layer)
        self.layer_norm1 = LayerNormalization()
        self.ffn = FeedForward(dim, dim)

    def call(self, x, training=False):
      xn = self.layer_norm(x)
      msr_x = self.msr(xn, xn, xn) + xn
      x = self.ffn(self.layer_norm1(msr_x)) + msr_x
      return x


class ChunkwiseRetention(Layer):
  def __init__(self, hidden_dim, gamma):
    super().__init__()
    self.rnn = RNN(ChunkwiseRecurrentRetention(hidden_dim, hidden_dim, gamma), return_sequences=True)
    self.gamma = gamma
    self.hdim = hidden_dim

  def _compute_decay(self, seq_len, gamma = 0.96875):
      _indices = list(range(0, seq_len))
      _decay_factors = [gamma**(i-j) if i>=j else 0 for i in _indices for j in _indices]
      D = tf.reshape(tf.convert_to_tensor(_decay_factors, dtype='float32'), (seq_len, seq_len))
      #mask = tf.eye(seq_len, dtype=tf.float32)
      #mask += tf.roll(mask, shift=-1, axis=-1)
      return D#*mask

  def call(self, x):
    b, s, d = x.shape
    D = self._compute_decay(s, self.gamma)
    D /= tf.reduce_sum(D, -1)**0.5
    _num_chunks = s//2
    chunks = tf.split(x, _num_chunks, -2)
    chunks = tf.stack(chunks, 1)
    self.rnn.cell.D = tf.cast(D, tf.float32)
    self.rnn.cell.chunk_size = s//_num_chunks
    return tf.reshape(self.rnn(chunks), (tf.shape(x)[0], s, self.hdim))

class ChunkwiseRecurrentRetention(tf.compat.v1.nn.rnn_cell.RNNCell):
    def __init__(self,
                 input_size,
                 hidden_size=32,
                 gamma=0.984375,
                 trainable=True,
                 dtype=None,
                 **kwargs):
        super(ChunkwiseRecurrentRetention, self).__init__(trainable=trainable,
                                        dtype=dtype,
                                        **kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.qkv = Dense(hidden_size*3)
        self.gamma = gamma
        self.counter = 1

    @property
    def state_size(self):
        return tf.TensorShape([self.hidden_size, self.hidden_size])

    @property
    def output_size(self):
        return self.hidden_size

    def call(self, inputs, state):
      q, k, v = tf.split(self.qkv(inputs), 3, -1)
      s = state*self.gamma**self.chunk_size + tf.linalg.matmul(k, v*(self.gamma**(self.chunk_size-1-self.counter)), transpose_a=True)
      x = ((tf.linalg.matmul(q, k, transpose_b=True)/self.hidden_size**0.5)*self.D[self.counter:self.counter+self.chunk_size, self.counter:self.counter+self.chunk_size])
      _norm_3 = lambda xs: tf.math.divide(xs, tf.maximum(tf.abs(tf.math.reduce_sum(xs, 1)), 1))
      x = tf.vectorized_map(_norm_3, x) #Normalization Trick 3
      x = x@v + (q@state)*tf.expand_dims(tf.convert_to_tensor([*repeat(self.gamma**(self.counter+1), self.chunk_size)]), -1)
      self.counter += 1
      return x, s


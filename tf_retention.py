# Imports
import tensorflow as tf
from pandas import read_csv, DataFrame, concat
from tensorflow.keras import Model, Sequential
from tensorboard.plugins.projector import ProjectorConfig, visualize_embeddings
from tensorflow.keras.layers import TextVectorization, Input, Embedding, Conv1D,\
 MultiHeadAttention, LayerNormalization, Add, Dense, Flatten, BatchNormalization,\
  DepthwiseConv1D, MaxPooling1D,\
   GlobalAveragePooling1D, Concatenate, GroupNormalization, LSTM, GlobalMaxPooling1D, Activation,\
    Dropout, Attention, Dot, Bidirectional, GRU
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.metrics import AUC
import numpy as np
import torch
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, ReLU, LayerNormalization, RNN, SimpleRNNCell
from tensorflow.keras.layers import Layer, Dense, LayerNormalization

class ParallelRetNetLayer(Layer):
  def __init__(self, dim, seq_len=50, gamma=0.96875, num_heads=4, gn_huh=True, **kwargs):
    super(ParallelRetNetLayer, self).__init__()
    self.dim = dim
    self.seq_len = seq_len
    self.gamma = gamma
    self.num_heads = num_heads
    self.gn_huh = gn_huh
    self.retention = {
        "query" : Dense(dim, use_bias=False, dtype='float32'),
        "key" : Dense(dim, use_bias=False, dtype='float32'),
        "value" : Dense(dim, use_bias=False, dtype='float32'),
    }
    indices = tf.range(seq_len, dtype=tf.float32)
    decay_factors = gamma ** (tf.expand_dims(indices, 1) - indices)
    self.decay = tf.ones((seq_len, seq_len), dtype=tf.float32) * decay_factors
    if self.gn_huh:
      self.gn = GroupNormalization(dim)

  def call(self, x):
    tf.cast(x, tf.float32)
    Q, K, V = [fn(x) for fn in self.retention.values()]
    D = self.decay
    x = Q@tf.transpose(K, perm=[0, 2, 1])
    x = x*D
    x = x@V

    return self.gn(x) if self.gn_huh else x

class RecurrentRetention(Layer):
    def __init__(self, dim, gamma, seq_len=50, **kwargs):
        super(RecurrentRetention, self).__init__()
        dims = dim
        self.retention = {
            "query": Dense(units=dim, use_bias=False, **kwargs),
            "key": Dense(units=dim, use_bias=False, **kwargs),
            "value": Dense(units=dim, use_bias=False, **kwargs),
        }
        self.gamma = tf.cast(gamma, tf.float32)
        self.s = RNN(SimpleRNNCell(dim))
        self.seq_len=seq_len

    def call(self, x):
        Q, K, V = [fn(x) for fn in self.retention.values()]
        state_rnn = self.s
        bias = tf.reduce_sum(tf.math.multiply(K, V), -1)

        s = [0 for i in range(self.seq_len)]
        for t in range(1, self.seq_len):
          s[t] = (s[t-1]*self.gamma) + tf.transpose(K[:, t, :], perm=[1, 0])@V[:, t , :]
        s[0] = s[1]
        S = tf.convert_to_tensor(s)
        S = tf.reshape(tf.math.reduce_sum(S, -1), [-1, self.seq_len])
        x = tf.multiply(tf.transpose(S), Q)
        return x


class MultiScaleRetention(Layer):
    def __init__(self, dim, hdim=128, seq_len=50, retention_layer=ParallelRetNetLayer, **kwargs):
        super(MultiScaleRetention, self).__init__()
        dims = dim
        gamma = 1 - (2 ** (-5 - torch.arange(0, hdim)))

        gamma = gamma.numpy().tolist()
        self.dim = dim
        self.hdim = hdim
        self.heads = [ParallelRetNetLayer(hdim, gamma=gamma[head], num_heads=dim//hdim, seq_len=seq_len, gn_huh=False) for head in range(dim // hdim)]
        self.gn = GroupNormalization(hdim)
        self.wg = Sequential([
            Dense(dims, use_bias=False, **kwargs),
            ReLU()
        ])
        self.wo = Dense(dims, use_bias=False, **kwargs)
    
    def call(self, x):
        W = self.wg(x)
        x = tf.split(x, self.dim//self.hdim, 2)
        x = tf.concat([headi(xi) for headi, xi in zip(self.heads, x)], -1)
        Y = self.gn(x)
        x = self.wo(W * Y)
        return x

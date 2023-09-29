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






import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from torch.nn import RNN
import torch

class Retention(Layer):
    def __init__(self, dim = 32, nheads = 2, seq_len = 50, gamma = 0.9865):
        super().__init__()

        _dense_kwargs = {
                "use_bias" : False,
                "dtype" : 'float32'
                }
        _layer_names = ['Q', 'K', 'V']
        _layer = Dense(dim, **_dense_kwargs)
        self.layers = dict.fromkeys(_layer_names, _layer)

        _indices = torch.arange(seq_len, dtype=torch.float)
        _decay_factors = gamma ** (_indices.unsqueeze(1) - _indices)
        D = tf.ones((seq_len, seq_len), dtype='float32') * _decay_factors.numpy()
        self.D = tf.transpose(tf.linalg.band_part(D, 0, -1), perm=[1, 0])

    def call(self, x):
        Q, K, V = [f(z) for f, z in zip(self.layers.values(), x)]
        _, _, d = Q.shape
        x = Q@tf.transpose(K, perm=[0, 2, 1])
        x /= d**0.5
        D = self.D
        D /= tf.reduce_sum(D, 1)**0.5

        x = x*D
        x = tf.maximum(tf.abs(tf.math.reduce_sum(x, 0)), 1)
        x = x@V
        return x



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
        self.seq_len=seq_len

    def call(self, x):
        Q, K, V = [fn(x) for fn in self.retention.values()]
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
    def __init__(self, dim, hdim=128, seq_len=50, **kwargs):
        super(MultiScaleRetention, self).__init__()
        dims = dim
        gamma = 1 - (2 ** (-5 - torch.arange(0, hdim)))
        gamma = gamma.numpy().tolist()
        self.dim = dim
        self.hdim = hdim
        self.heads = [Retention(hdim, gamma=gamma[head], seq_len=seq_len) for head in range(dim // hdim)]
        self.gn = GroupNormalization(1)
        self.wg = Sequential([
            Dense(dims, use_bias=False, **kwargs),
            ReLU()
        ])
        self.wo = Dense(dims, use_bias=False, **kwargs)

    def call(self, x, k, v):
        W = self.wg(x)
        q = tf.split(x, self.dim//self.hdim, 2)
        k = tf.split(k, self.dim//self.hdim, 2)
        v = tf.split(v, self.dim//self.hdim, 2)
        x = tf.concat([headi([qi, ki, vi]) for headi, qi, ki, vi in zip(self.heads, q, k, v)], -1)
        Y = self.gn(x)
        x = self.wo(W * Y)
        return x


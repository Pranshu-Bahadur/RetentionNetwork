from torch.nn import Module, ModuleDict, Linear, GroupNorm

#@TODO: this might be wrong (no heads and that)
class ParallelRetNetLayer(Module):
  def __init__(self, dim : int, seq_len=50, gamma=0.96875, num_heads=50, gn_huh=True):
    super().__init__()
    dims = (dim, dim)
    self.retention = {
        "query" : Linear(*dims, bias=False),
        "key" : Linear(*dims, bias=False),
        "value" : Linear(*dims, bias=False),
    }
    dtype=torch.float
    indices = torch.arange(seq_len, dtype=dtype)
    decay_factors = gamma ** (indices.unsqueeze(1) - indices)
    self.decay = torch.ones((seq_len, seq_len), dtype=dtype) * decay_factors
    self.gn_huh = gn_huh
    if self.gn_huh:
      self.gn = GroupNorm(num_heads, seq_len)

  def forward(self, x):
    Q, K, V = list(map(lambda fn: fn(x), self.retention.values()))
    D = self.decay
    x = Q@K.transpose(1, 2)
    x = x*D
    x = x@V
    return self.gn(x) if self.gn_huh else x

from torch.nn import Module, RNN

class RecurrentRetention(Module):
  def __init__(self, dim : int, seq_len=50, gamma=0.96875, num_heads=50, gn_huh=True):
    super(

    ).__init__()
    dims = (dim, dim)
    self.retention = {
        "query" : Linear(*dims, bias=False),
        "key" : Linear(*dims, bias=False),
        "value" : Linear(*dims, bias=False),
    }


    self.gn_huh = gn_huh
    if self.gn_huh:
      self.gn = GroupNorm(num_heads, seq_len)
    self.s = RNN(seq_len, seq_len, nonlinearity='tanh', batch_first=True)

    self.s.requires_grad=False

    self.gamma = torch.full((seq_len, ), gamma)

  def forward(self, x):
    Q, K, V = list(map(lambda fn: fn(x), self.retention.values()))
    state_rnn = self.s
    bias = torch.sum(K*V, -1)
    state_rnn._parameters.bias_hh_l0 =  bias
    gamma = torch.stack([self.gamma for _ in range(Q.size(0))])
    S, _ = state_rnn(gamma.float())
    return torch.vmap(torch.mul)(S, Q.transpose(1, 2)).transpose(2, 1)


import torch
from torch.nn import Module, ModuleList, SiLU, Sequential, Linear, SiLU, GELU

class MultiScaleRetention(Module): #Without Multi lol
  def __init__(self, dim : int, hdim=32, seq_len=50):
    super().__init__()

    dims = (dim, dim)
    gamma = 1 - (2 ** (-5 - torch.arange(0, hdim)))


    self.hdim = hdim

    self.heads = ModuleList([
        RecurrentRetention(hdim, gamma=gamma[head], gn_huh=True)\
        for head in range(dim//hdim)
    ])

    self.gn = GroupNorm(seq_len, seq_len)

    self.wg = Sequential(
        Linear(*dims, bias=False),
        SiLU()
    )
    self.wo = Linear(*dims, bias=False)


  def forward(self, x):
    W = self.wg(x)
    x = torch.split(x, self.hdim, 2)
    x = [headi(xi) for headi, xi in zip(self.heads, x)]
    Y = self.gn(torch.cat(x, 2))
    return self.wo(W*Y)



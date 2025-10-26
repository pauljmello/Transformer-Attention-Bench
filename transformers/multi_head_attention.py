import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_attention import BaseAttention


class MultiHeadLatentAttention(BaseAttention):
    def __init__(self, d_model, num_heads):
        super(MultiHeadLatentAttention, self).__init__(d_model=d_model, num_classes=10)

        # Multi-Head Latent Attention: Reduce KV cache using low-rank compression
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Shared latent compression
        self.d_c = d_model // 8

        self.W_DQ = nn.Linear(d_model, self.d_c, bias=False)  # Down-projection
        self.W_UQ = nn.Linear(self.d_c, d_model, bias=False)  # Up-projection

        self.W_DKV = nn.Linear(d_model, self.d_c, bias=False)  # Shared down projection
        self.W_UK = nn.Linear(self.d_c, d_model, bias=False)  # Key up projection
        self.W_UV = nn.Linear(self.d_c, d_model, bias=False)  # Value up projection

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        batch_size, seq_len_actual, d_model = X.shape

        C_Q = self.norm_cq(self.W_DQ(X)) if hasattr(self, 'norm_cq') else self.W_DQ(X)
        Q = self.W_UQ(C_Q)

        C_KV = self.norm_ckv(self.W_DKV(X)) if hasattr(self, 'norm_ckv') else self.W_DKV(X)
        K = self.W_UK(C_KV)
        V = self.W_UV(C_KV)

        Q = Q.unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = K.unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = V.unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        Q = self._apply_rope(Q, seq_len_actual)
        K = self._apply_rope(K, seq_len_actual)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output

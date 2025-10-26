import math

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

from base_attention import BaseAttention


def _causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class FlexAttention(BaseAttention):
    def __init__(self, d_model=512, num_heads=8, num_classes=10, max_seq_len=1024):
        super(FlexAttention, self).__init__(d_model, num_classes)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self._block_mask_cache = {}

    def _get_block_mask(self, seq_len, device):
        if seq_len not in self._block_mask_cache:
            block_mask = create_block_mask(_causal_mask_mod, 1, None, seq_len, seq_len, device=device)
            self._block_mask_cache[seq_len] = block_mask
        return self._block_mask_cache[seq_len]

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        Q = self.W_q(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        block_mask = self._get_block_mask(seq_len, Q.device)

        output = flex_attention(Q, K, V, block_mask=block_mask, scale=self.scale)

        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        output = self.W_o(output)

        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits, None

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_attention import BaseAttention


class LocalAttention(BaseAttention):
    def __init__(self, d_model=512, num_heads=8, num_classes=10, max_seq_len=1024, window_size=128):
        super(LocalAttention, self).__init__(d_model, num_classes)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.window_size = window_size

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Linear projections: Q = XW_q, K = XW_k, V = XW_v
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def create_local_mask(self, seq_len, device):
        positions = torch.arange(seq_len, device=device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        half_window = self.window_size // 2
        mask = distance.abs() <= half_window
        return mask

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        local_mask = self.create_local_mask(seq_len, X.device)
        local_mask = local_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~local_mask, -float('inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits, attn_weights

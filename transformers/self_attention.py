import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_attention import BaseAttention


class SelfAttention(BaseAttention):
    def __init__(self, d_model=512, num_classes=10, max_seq_len=1024):
        super(SelfAttention, self).__init__(d_model, num_classes)
        self.max_seq_len = max_seq_len
        self.scale = 1.0 / math.sqrt(d_model)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward_torch(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits, attn_weights

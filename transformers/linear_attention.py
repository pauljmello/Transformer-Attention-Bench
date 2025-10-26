import torch
import torch.nn as nn
import torch.nn.functional as F

from base_attention import BaseAttention


class LinearAttention(BaseAttention):
    def __init__(self, d_model=512, num_heads=8, num_classes=10, max_seq_len=1024, feature_dim=64):
        super(LinearAttention, self).__init__(d_model, num_classes)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.feature_map = nn.Linear(self.d_k, feature_dim, bias=False)

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        phi_Q = F.elu(self.feature_map(Q), inplace=False).add_(1.0)
        phi_K = F.elu(self.feature_map(K), inplace=False).add_(1.0)

        KV = torch.matmul(phi_K.transpose(-2, -1), V)
        numerator = torch.matmul(phi_Q, KV)

        ones = torch.ones(batch_size, self.num_heads, seq_len, 1, device=Q.device)
        K_sum = torch.matmul(phi_K.transpose(-2, -1), ones)
        denominator = torch.matmul(phi_Q, K_sum)
        output = numerator / (denominator + 1e-8)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits, phi_Q

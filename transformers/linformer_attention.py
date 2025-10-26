import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_attention import BaseAttention


class Linformer(BaseAttention):
    def __init__(self, d_model=512, num_heads=8, num_classes=10, max_seq_len=1024, projection_dim=None, error_tolerance=0.1, share_projections=True):
        super(Linformer, self).__init__(d_model, num_classes)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.error_tolerance = error_tolerance
        self.share_projections = share_projections

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        # Johnson-Lindenstrauss projection dimension
        if projection_dim is None:
            self.projection_dim = self._compute_jl_dimension(max_seq_len, error_tolerance)
        else:
            self.projection_dim = projection_dim

        # Linear projections: Q = XW_q, K = XW_k, V = XW_v
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Low-rank projections
        if share_projections:
            self.E = nn.Linear(max_seq_len, self.projection_dim, bias=False)
            self.F = self.E
        else:
            self.E = nn.Linear(max_seq_len, self.projection_dim, bias=False)
            self.F = nn.Linear(max_seq_len, self.projection_dim, bias=False)

    def _compute_jl_dimension(self, n, epsilon):
        # Johnson-Lindenstrauss lemma: k ≥ 8 * log(n) / epsilon²
        k = math.ceil(8 * math.log(n) / (epsilon ** 2))
        k = max(k, 32)
        k = min(k, n // 2)
        k = 2 ** math.ceil(math.log2(k))
        return k

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        if seq_len > self.projection_dim:
            E_weight = self.E.weight[:, :seq_len]
            F_weight = self.F.weight[:, :seq_len]

            b, h, s, d = K.shape
            K_flat = K.permute(0, 1, 3, 2).reshape(b * h * d, s)
            V_flat = V.permute(0, 1, 3, 2).reshape(b * h * d, s)

            K_proj = torch.matmul(K_flat, E_weight).view(b, h, d, -1).transpose(-1, -2)
            V_proj = torch.matmul(V_flat, F_weight).view(b, h, d, -1).transpose(-1, -2)
        else:
            K_proj, V_proj = K, V

        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V_proj)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        # Classification
        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits, attn_weights

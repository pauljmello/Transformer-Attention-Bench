import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_attention import BaseAttention


class SparseAttention(BaseAttention):
    def __init__(self, d_model=512, num_heads=8, num_classes=10, max_seq_len=1024, num_global_tokens=64, window_size=128, num_random_tokens=32, block_size=64):
        super(SparseAttention, self).__init__(d_model, num_classes)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_global_tokens = num_global_tokens
        self.window_size = window_size
        self.num_random_tokens = num_random_tokens
        self.block_size = block_size

        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def create_bigbird_pattern(self, seq_len, device):
        cache_key = seq_len
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        pattern = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        global_end = min(self.num_global_tokens, seq_len)
        if global_end > 0:
            pattern[:global_end, :] = True
            pattern[:, :global_end] = True

        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            pattern[i, start:end] = True

        torch.manual_seed(42)
        for i in range(global_end, seq_len):
            local_start = max(0, i - self.window_size // 2)
            local_end = min(seq_len, i + self.window_size // 2 + 1)

            candidates = []
            for j in range(seq_len):
                if j < global_end:
                    continue
                if local_start <= j < local_end:
                    continue
                candidates.append(j)

            if candidates and self.num_random_tokens > 0:
                num_random = min(self.num_random_tokens, len(candidates))
                selected = torch.randperm(len(candidates))[:num_random]
                for idx in selected:
                    j = candidates[idx]
                    pattern[i, j] = True
                    pattern[j, i] = True

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        pattern = pattern & causal_mask

        pattern.fill_diagonal_(True)

        self._pattern_cache[cache_key] = pattern
        return pattern

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        sparse_pattern = self.create_bigbird_pattern(seq_len, X.device)
        sparse_mask = sparse_pattern.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~sparse_mask, -float('inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits, None

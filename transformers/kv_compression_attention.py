import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from base_attention import BaseAttention


class KVCompressionAttention(BaseAttention):
    def __init__(self, d_model=512, num_heads=8, num_classes=10, max_seq_len=1024, kv_budget=256, eviction_strategy='h2o'):
        super(KVCompressionAttention, self).__init__(d_model, num_classes)
        self.num_heads = num_heads
        self.kv_budget = kv_budget
        self.eviction_strategy = eviction_strategy
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # H2O state tracking
        self.register_buffer('cum_scores', torch.zeros(max_seq_len))
        self.register_buffer('token_counts', torch.zeros(max_seq_len))

        self.sink_budget = min(4, kv_budget // 4)
        self.recent_budget = min(32, kv_budget // 4)
        self.heavy_budget = kv_budget - self.sink_budget - self.recent_budget
        self.window_size = 32

    def h2o_eviction(self, Q, K, V):
        batch_size, num_heads, seq_len, d_k = Q.shape

        if seq_len <= self.kv_budget:
            return K, V, torch.arange(seq_len, device=Q.device)

        window_start = max(0, seq_len - self.window_size)
        recent_Q = Q[:, :, window_start:, :]

        scores = torch.matmul(recent_Q, K.transpose(-2, -1)) * self.scale

        query_pos = torch.arange(window_start, seq_len, device=Q.device).unsqueeze(1)
        key_pos = torch.arange(seq_len, device=Q.device).unsqueeze(0)
        causal_mask = query_pos >= key_pos
        scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

        # Compute attention weights and update cumulative scores
        attn_weights = F.softmax(scores, dim=-1)
        current_importance = attn_weights.mean(dim=(0, 1, 2))

        # Update scores
        self.cum_scores[:seq_len] += current_importance
        self.token_counts[:seq_len] += 1

        selected_indices = []

        if self.sink_budget > 0:
            sink_indices = torch.arange(min(self.sink_budget, seq_len), device=Q.device)
            selected_indices.append(sink_indices)

        if self.recent_budget > 0:
            recent_start = max(0, seq_len - self.recent_budget)
            recent_indices = torch.arange(recent_start, seq_len, device=Q.device)
            selected_indices.append(recent_indices)

        # Middle tokens with high attention
        if self.heavy_budget > 0:
            middle_start = self.sink_budget
            middle_end = max(middle_start, seq_len - self.recent_budget)

            if middle_end > middle_start:
                # Calculate average attention scores for middle tokens
                middle_scores = self.cum_scores[middle_start:middle_end]
                middle_counts = torch.clamp(self.token_counts[middle_start:middle_end], min=1)
                avg_scores = middle_scores / middle_counts

                num_heavy = min(self.heavy_budget, middle_end - middle_start)
                _, heavy_relative = torch.topk(avg_scores, num_heavy)
                heavy_indices = heavy_relative + middle_start
                selected_indices.append(heavy_indices)

        # Deduplicate indices
        if selected_indices:
            all_indices = torch.cat(selected_indices)
            selected = torch.unique(all_indices, sorted=True)
        else:
            selected = torch.arange(max(0, seq_len - self.kv_budget), seq_len, device=Q.device)

        return K[:, :, selected], V[:, :, selected], selected

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        Q, K, V = [proj(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) for proj in [self.W_q, self.W_k, self.W_v]]

        # Apply KV compression if needed
        if self.eviction_strategy == 'h2o' and seq_len > self.kv_budget:
            K_compressed, V_compressed, selected_indices = self.h2o_eviction(Q, K, V)

            scores = torch.matmul(Q, K_compressed.transpose(-2, -1)) * self.scale

            query_pos = torch.arange(seq_len, device=Q.device).unsqueeze(1)
            key_pos = selected_indices.unsqueeze(0)
            causal_mask = query_pos >= key_pos

            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))
            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V_compressed)
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), -float('inf'))

            attn_weights = F.softmax(scores, dim=-1)
            output = torch.matmul(attn_weights, V)

        # Output projection and classification
        output = self.W_o(output.transpose(1, 2).reshape(batch_size, seq_len, d_model))
        pooled = torch.mean(output, dim=1)
        logits = self.classifier(pooled)

        return logits, attn_weights

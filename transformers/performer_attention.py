import math

import torch
import torch.nn as nn

from base_attention import BaseAttention


class PerformerFAVOR(BaseAttention):
    def __init__(self, d_model=512, num_heads=8, num_classes=10, max_seq_len=1024, num_features=256, use_orthogonal_features=True, redraw_interval=1000):
        super(PerformerFAVOR, self).__init__(d_model, num_classes)
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.use_orthogonal_features = use_orthogonal_features
        self.redraw_interval = redraw_interval

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.random_features = self._create_orthogonal_random_features()
        self.step_count = 0

    def _create_orthogonal_random_features(self):
        if self.use_orthogonal_features:
            if self.num_features <= self.d_k:
                random_matrix = torch.randn(self.d_k, self.num_features, dtype=torch.float32)
                q, r = torch.linalg.qr(random_matrix)
                random_matrix = q[:, :self.num_features].T * math.sqrt(self.d_k)
            else:
                num_blocks = math.ceil(self.num_features / self.d_k)
                orthogonal_blocks = []

                for i in range(num_blocks):
                    start_idx = i * self.d_k
                    end_idx = min((i + 1) * self.d_k, self.num_features)
                    block_size = end_idx - start_idx

                    if block_size > 0:
                        actual_block_size = min(block_size, self.d_k)
                        block = torch.randn(self.d_k, actual_block_size, dtype=torch.float32)
                        q, r = torch.linalg.qr(block)
                        orthogonal_blocks.append(q[:, :actual_block_size].T)

                random_matrix = torch.cat(orthogonal_blocks, dim=0)[:self.num_features]
                random_matrix = random_matrix * math.sqrt(self.d_k)
        else:
            random_matrix = torch.randn(self.num_features, self.d_k, dtype=torch.float32)
            random_matrix = random_matrix * math.sqrt(self.d_k / self.num_features)

        return random_matrix

    def _redraw_features_if_needed(self, device):
        if self.step_count % self.redraw_interval == 0:
            self.random_features = self._create_orthogonal_random_features()
            self.random_features = self.random_features.to(device)
        self.step_count += 1

    def favor_plus_kernel_map(self, x):
        if self.random_features.device != x.device:
            self.random_features = self.random_features.to(x.device)

        x_normalized = x / math.sqrt(math.sqrt(self.d_k))

        x_proj = torch.matmul(x_normalized, self.random_features.T)
        x_norm_sq = torch.sum(x_normalized ** 2, dim=-1, keepdim=True) / 2.0
        features = torch.exp(x_proj - x_norm_sq)

        features = features / math.sqrt(self.num_features)
        return features

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        self._redraw_features_if_needed(X.device)

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        Q = Q / math.sqrt(math.sqrt(self.d_k))
        K = K / math.sqrt(math.sqrt(self.d_k))

        phi_Q = self.favor_plus_kernel_map(Q)
        phi_K = self.favor_plus_kernel_map(K)

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

        return logits, None

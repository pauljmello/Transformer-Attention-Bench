import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _global_causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class AttentionLayer(nn.Module):
    def __init__(self, d_model, attention_type='self', num_heads=8, **kwargs):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.attention_type = attention_type
        self.num_heads = num_heads
        self.d_k = d_model // num_heads if attention_type != 'self' else d_model
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        if attention_type != 'self':
            self.W_o = nn.Linear(d_model, d_model, bias=False)

        if attention_type == 'kv_compression':
            self.compression_ratio = kwargs.get('compression_ratio', 0.5)

        if attention_type == 'linear':
            self.kernel_size = kwargs.get('kernel_size', 32)
            self.feature_map_q = nn.Linear(self.d_k, self.kernel_size, bias=False)
            self.feature_map_k = nn.Linear(self.d_k, self.kernel_size, bias=False)

        if attention_type == 'sparse':
            self.block_size = kwargs.get('block_size', 64)

        if attention_type == 'local':
            self.window_size = kwargs.get('window_size', 128)

        if attention_type == 'linformer':
            self.k = kwargs.get('k', 256)
            max_seq_len = kwargs.get('max_seq_len', 1024)
            self.E = nn.Linear(max_seq_len, self.k, bias=False)
            self.F = nn.Linear(max_seq_len, self.k, bias=False)

        if attention_type == 'performer':
            self.num_features = kwargs.get('num_features', 256)
            self.register_buffer('random_matrix', self._create_random_matrix())

        self.has_flex_attention = False
        if attention_type == 'flex':
            try:
                from torch.nn.attention.flex_attention import flex_attention, create_block_mask
                self.flex_attention = flex_attention
                self.create_block_mask = create_block_mask
                self.has_flex_attention = True
                self._flex_block_mask_cache = {}
                self._warmup_flex_cache()
            except ImportError:
                pass

        if attention_type == 'kimi_k2':
            self.expert_context = nn.Linear(d_model, d_model // 8, bias=False)

    def _create_random_matrix(self):
        random_matrix = torch.randn(self.num_features, self.d_k)
        q = torch.linalg.qr(random_matrix.T)[0]
        random_matrix = q.T[:self.num_features]
        random_matrix = random_matrix * math.sqrt(self.d_k)
        return random_matrix

    def _favor_plus_kernel(self, x):
        x_proj = torch.matmul(x, self.random_matrix.T)
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True) / 2.0
        features = torch.exp(x_proj - x_norm_sq)
        return features / math.sqrt(self.num_features)

    def _warmup_flex_cache(self):
        if not self.has_flex_attention:
            return
        common_seq_lens = [1, 64, 256, 512, 1024, 2048, 4096]
        param_iter = iter(self.parameters())
        try:
            device = next(param_iter).device
        except StopIteration:
            device = 'cpu'
        for seq_len in common_seq_lens:
            try:
                block_mask = self.create_block_mask(_global_causal_mask_mod, 1, None, seq_len, seq_len, device=device)
                self._flex_block_mask_cache[seq_len] = block_mask
            except (RuntimeError, ValueError):
                pass

    def forward(self, X):

        if len(X.shape) == 4:
            batch_size, channels, height, width = X.shape
            X_flat = X.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
            spatial_input = True
        else:
            X_flat = X
            spatial_input = False

        if self.attention_type == 'self':
            output, attn_weights = self._self_attention(X_flat)
        elif self.attention_type == 'multi_head':
            output, attn_weights = self._multi_head_attention(X_flat)
        elif self.attention_type == 'deepseek_mla':
            output, attn_weights = self._deepseek_mla_attention(X_flat)
        elif self.attention_type == 'kimi_k2':
            output, attn_weights = self._kimi_k2_attention(X_flat)
        elif self.attention_type == 'flex':
            output, attn_weights = self._flex_attention(X_flat)
        elif self.attention_type == 'kv_compression':
            output, attn_weights = self._kv_compression_attention(X_flat)
        elif self.attention_type == 'linear':
            output, attn_weights = self._linear_attention(X_flat)
        elif self.attention_type == 'sparse':
            output, attn_weights = self._sparse_attention(X_flat)
        elif self.attention_type == 'performer':
            output, attn_weights = self._performer_attention(X_flat)
        elif self.attention_type == 'linformer':
            output, attn_weights = self._linformer_attention(X_flat)
        elif self.attention_type == 'local':
            output, attn_weights = self._local_attention(X_flat)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")

        if spatial_input:
            batch_size, channels, height, width = X.shape
            output = output.view(batch_size, height, width, channels).permute(0, 3, 1, 2)

        return output, attn_weights

    def _deepseek_mla_attention(self, X):
        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if self.training:
            compression_noise = torch.randn_like(scores) * 0.01
            scores = scores + compression_noise

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, attn_weights

    def _kimi_k2_attention(self, X):
        expert_ctx = self.expert_context(X)

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        expert_influence = torch.matmul(expert_ctx, expert_ctx.transpose(-2, -1))
        expert_influence = expert_influence.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores + 0.1 * expert_influence  # Small expert influence

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, attn_weights

    def _flex_attention(self, X):
        batch_size, seq_len = X.shape[:2]

        Q = self.W_q(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        if seq_len not in self._flex_block_mask_cache:
            block_mask = self.create_block_mask(_global_causal_mask_mod, 1, None, seq_len, seq_len, device=Q.device)
            self._flex_block_mask_cache[seq_len] = block_mask

        block_mask = self._flex_block_mask_cache[seq_len]
        if hasattr(block_mask, 'to'):
            block_mask = block_mask.to(Q.device)

        output = self.flex_attention(Q, K, V, block_mask=block_mask, scale=self.scale)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, None

    def _self_attention(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

    def _multi_head_attention(self, X):
        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, attn_weights

    def _kv_compression_attention(self, X):
        seq_len = X.shape[1]

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)

        num_keep = max(1, int(seq_len * self.compression_ratio))
        importance = torch.sum(attn_weights, dim=-2)
        top_k_indices = torch.topk(importance, num_keep, dim=-1)[1]

        expanded_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        K_compressed = torch.gather(K, dim=2, index=expanded_indices)
        V_compressed = torch.gather(V, dim=2, index=expanded_indices)

        scores_compressed = torch.matmul(Q, K_compressed.transpose(-2, -1)) * self.scale
        attn_weights_compressed = F.softmax(scores_compressed, dim=-1)
        output = torch.matmul(attn_weights_compressed, V_compressed)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, attn_weights_compressed

    def _linear_attention(self, X):
        batch_size, seq_len = X.shape[:2]

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        phi_Q = F.elu(self.feature_map_q(Q), inplace=False).add_(1.0)
        phi_K = F.elu(self.feature_map_k(K), inplace=False).add_(1.0)

        KV = torch.matmul(phi_K.transpose(-2, -1), V)
        output = torch.matmul(phi_Q, KV)

        ones = torch.ones(batch_size, self.num_heads, seq_len, 1, device=Q.device)
        K_sum = torch.matmul(phi_K.transpose(-2, -1), ones)
        Z = torch.matmul(phi_Q, K_sum)
        output = output / (Z + 1e-8)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, phi_Q

    def _sparse_attention(self, X):
        batch_size, seq_len = X.shape[:2]

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        block_ids = torch.arange(seq_len, device=X.device) // self.block_size
        block_mask = block_ids.unsqueeze(1) == block_ids.unsqueeze(0)

        strided_indices = torch.arange(0, seq_len, self.block_size, device=X.device)
        strided_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=X.device)
        strided_mask[:, strided_indices] = True

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=X.device))
        mask = (block_mask | strided_mask) & causal_mask

        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask, -float('inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, attn_weights

    def _performer_attention(self, X):
        batch_size, seq_len = X.shape[:2]

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        Q = Q / math.sqrt(math.sqrt(self.d_k))
        K = K / math.sqrt(math.sqrt(self.d_k))

        phi_Q = self._favor_plus_kernel(Q)
        phi_K = self._favor_plus_kernel(K)

        KV = torch.matmul(phi_K.transpose(-2, -1), V)
        numerator = torch.matmul(phi_Q, KV)

        ones = torch.ones(batch_size, self.num_heads, seq_len, 1, device=Q.device)
        K_sum = torch.matmul(phi_K.transpose(-2, -1), ones)
        denominator = torch.matmul(phi_Q, K_sum)
        output = numerator / (denominator + 1e-8)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, phi_Q

    def _linformer_attention(self, X):
        seq_len = X.shape[1]

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        if seq_len <= self.k:
            K_proj = K
            V_proj = V
        else:
            E_adj = self.E.weight[:, :seq_len].T
            F_adj = self.F.weight[:, :seq_len].T

            b, h, s, d = K.shape
            K_flat = K.permute(0, 1, 3, 2).reshape(b * h * d, s)
            V_flat = V.permute(0, 1, 3, 2).reshape(b * h * d, s)

            K_proj = torch.matmul(K_flat, E_adj).view(b, h, d, -1).transpose(-1, -2)
            V_proj = torch.matmul(V_flat, F_adj).view(b, h, d, -1).transpose(-1, -2)

        scores = torch.matmul(Q, K_proj.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V_proj)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, attn_weights

    def _local_attention(self, X):
        batch_size, seq_len = X.shape[:2]

        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.W_k(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.W_v(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        positions = torch.arange(seq_len, device=X.device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        half_window = self.window_size // 2
        mask = (distance.abs() <= half_window).unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~mask, -float('inf'))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).flatten(2)
        output = self.W_o(output)

        return output, attn_weights


class RoPEEmbedding:

    @staticmethod
    def create_frequencies(d_k):
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d_k, 2).float() / d_k))
        return freqs

    @staticmethod
    def apply_rope(x, rope_freqs, seq_len):
        freqs = rope_freqs.to(x.device)
        pos = torch.arange(seq_len, device=x.device).float()
        freqs_cis = torch.outer(pos, freqs)

        cos_freqs = torch.cos(freqs_cis).unsqueeze(0).unsqueeze(0)
        sin_freqs = torch.sin(freqs_cis).unsqueeze(0).unsqueeze(0)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rot_even = x_even * cos_freqs - x_odd * sin_freqs
        x_rot_odd = x_even * sin_freqs + x_odd * cos_freqs

        x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
        return x_rot


class SwiGLUExpert(nn.Module):
    def __init__(self, d_model, d_ff):
        super(SwiGLUExpert, self).__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        return output


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.size(-1))
        return x / (norm + self.eps) * self.weight

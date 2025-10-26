import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_layers import RoPEEmbedding, SwiGLUExpert, RMSNorm


class KimiK2(nn.Module):
    def __init__(self, d_model=7168, num_heads=64, num_layers=51, num_experts=384, num_active_experts=8, vocab_size=160000, max_seq_len=131072, num_classes=10, d_ff=18432):
        super(KimiK2, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.d_ff = d_ff

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope_freqs = RoPEEmbedding.create_frequencies(self.d_k)
        self.layers = nn.ModuleList([KimiLayer(d_model, num_heads, num_experts, num_active_experts, d_ff) for _ in range(num_layers)])

        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        patch_size = 16
        self.patch_proj = nn.Linear(patch_size * patch_size * 3, d_model)

    def forward_torch(self, X):
        if len(X.shape) == 4:
            patch_size = 16
            X_patches = F.unfold(X.permute(0, 3, 1, 2), kernel_size=patch_size, stride=patch_size)
            X_patches = X_patches.transpose(1, 2)  # (batch, num_patches, patch_dim)
            X = self.patch_proj(X_patches)
        else:
            if X.dtype == torch.long:
                X = self.token_embedding(X)
            else:
                if X.shape[-1] != self.d_model:
                    if not hasattr(self, '_input_proj_cache'):
                        self._input_proj_cache = {}
                    input_dim = X.shape[-1]
                    if input_dim not in self._input_proj_cache:
                        self._input_proj_cache[input_dim] = nn.Linear(input_dim, self.d_model).to(X.device)
                    X = self._input_proj_cache[input_dim](X)

        batch_size, seq_len, d_model = X.shape

        for layer in self.layers:
            X = layer(X, seq_len)

        X = self.final_norm(X)
        pooled = torch.mean(X, dim=1)
        logits = self.classifier(pooled)

        return logits, None

    def forward(self, X):
        logits, attention_weights = self.forward_torch(X)
        predictions = F.softmax(logits, dim=1)
        return predictions, attention_weights

    def compute_loss(self, predictions, targets):
        return F.cross_entropy(predictions, targets)


class KimiLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_experts, num_active_experts, d_ff):
        super(KimiLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.moe = KimiMoE(d_model, num_experts, num_active_experts, d_ff)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, X):
        norm_X = self.norm1(X)
        attn_output = self.attention(norm_X)
        X = X + attn_output

        norm_X = self.norm2(X)
        moe_output = self.moe(norm_X)
        X = X + moe_output

        return X


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        # Linear projections: Q = XW_q, K = XW_k, V = XW_v
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        Q = self.q_proj(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = self.k_proj(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = self.v_proj(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).flatten(2)

        output = self.o_proj(output)

        return output


class KimiMoE(nn.Module):
    def __init__(self, d_model, num_experts, num_active_experts, d_ff):
        super(KimiMoE, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.d_ff = d_ff

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.shared_expert = SwiGLUExpert(d_model, d_ff)

        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)])

        self.register_buffer('expert_counts', torch.zeros(num_experts))

    def forward(self, X):
        batch_size, seq_len, d_model = X.shape
        X_flat = X.reshape(-1, d_model)

        router_logits = self.router(X_flat)
        router_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(router_weights, self.num_active_experts, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        if self.training:
            expert_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
            self.expert_counts += expert_mask.sum(dim=[0, 1])

        shared_output = self.shared_expert(X_flat)

        specialist_output = torch.zeros_like(X_flat)

        for expert_id in range(self.num_experts):
            # Find all tokens routed to this expert across all active slots
            expert_mask = (top_k_indices == expert_id)
            if expert_mask.any():
                weights = torch.where(expert_mask, top_k_weights, torch.zeros_like(top_k_weights))
                token_weights = weights.sum(dim=1, keepdim=True)

                # Find tokens that route to this expert
                has_expert = token_weights.squeeze(-1) > 0
                if has_expert.any():
                    expert_input = X_flat[has_expert]
                    expert_out = self.experts[expert_id](expert_input)
                    specialist_output[has_expert] += token_weights[has_expert] * expert_out

        output = shared_output + specialist_output
        output = output.reshape(batch_size, seq_len, d_model)

        return output

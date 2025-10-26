import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_layers import RoPEEmbedding, SwiGLUExpert, RMSNorm


class DeepSeekV3(nn.Module):
    def __init__(self, d_model=7168, num_heads=128, num_layers=61, num_experts=256, num_active_experts=8, vocab_size=129000, max_seq_len=131072, num_classes=10, d_ff=18432):
        super(DeepSeekV3, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.d_ff = d_ff

        self.d_k = d_model // num_heads
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rope_freqs = RoPEEmbedding.create_frequencies(self.d_k)
        self.layers = nn.ModuleList([DeepSeekLayer(d_model, num_heads, num_experts, num_active_experts, d_ff) for _ in range(num_layers)])

        self.final_norm = RMSNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        patch_size = 16
        self.patch_proj = nn.Linear(patch_size * patch_size * 3, d_model)

    def forward_torch(self, X):
        if len(X.shape) == 4:
            batch_size, height, width, channels = X.shape
            patch_size = 16
            seq_len = (height // patch_size) * (width // patch_size)
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


class DeepSeekLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_experts, num_active_experts, d_ff):
        super(DeepSeekLayer, self).__init__()
        self.mla = MultiHeadLatentAttention(d_model, num_heads)
        self.moe = DeepSeekMoE(d_model, num_experts, num_active_experts, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, X):
        norm_X = self.norm1(X)
        attn_output = self.mla(norm_X)
        X = X + attn_output

        norm_X = self.norm2(X)
        moe_output = self.moe(norm_X)
        X = X + moe_output

        return X


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadLatentAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.W_q = nn.Linear(d_model, d_model, bias=False)

        self.d_kv_compressed = d_model // 4  # 93% reduction
        self.W_k_compressed = nn.Linear(d_model, self.d_kv_compressed, bias=False)
        self.W_v_compressed = nn.Linear(d_model, self.d_kv_compressed, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        Q = self.W_q(X).unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        K_compressed = self.W_k_compressed(X)
        V_compressed = self.W_v_compressed(X)

        K = K_compressed.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        V = V_compressed.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        output = output.transpose(1, 2).flatten(2)

        output = self.W_o(output)

        return output


class DeepSeekMoE(nn.Module):
    def __init__(self, d_model, num_experts, num_active_experts, d_ff):
        super(DeepSeekMoE, self).__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.num_active_experts = num_active_experts
        self.d_ff = d_ff

        self.router = nn.Linear(d_model, num_experts, bias=False)

        self.shared_expert = SwiGLUExpert(d_model, d_ff)

        self.experts = nn.ModuleList([SwiGLUExpert(d_model, d_ff) for _ in range(num_experts)])

        self.expert_bias = nn.Parameter(torch.zeros(num_experts))

    def forward(self, X):
        batch_size, seq_len, d_model = X.shape
        X_flat = X.reshape(-1, d_model)

        router_logits = self.router(X_flat) + self.expert_bias
        router_weights = F.softmax(router_logits, dim=-1)

        top_k_weights, top_k_indices = torch.topk(router_weights, self.num_active_experts, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)  # Renormalize

        shared_output = self.shared_expert(X_flat)

        # Process all tokens for each expert at once
        specialist_output = torch.zeros_like(X_flat)

        for expert_id in range(self.num_experts):
            # Find all tokens
            expert_mask = (top_k_indices == expert_id)
            if expert_mask.any():
                weights = torch.where(expert_mask, top_k_weights, torch.zeros_like(top_k_weights))
                # Sum weights across expert slots
                token_weights = weights.sum(dim=1, keepdim=True)

                # Find tokens that route to expert
                has_expert = token_weights.squeeze(-1) > 0
                if has_expert.any():
                    expert_input = X_flat[has_expert]
                    expert_out = self.experts[expert_id](expert_input)
                    specialist_output[has_expert] += token_weights[has_expert] * expert_out

        output = shared_output + specialist_output
        output = output.reshape(batch_size, seq_len, d_model)

        return output

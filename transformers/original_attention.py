import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OriginalTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_layers=6, d_ff=2048, num_classes=10, max_seq_len=1024, dropout=0.1):
        super(OriginalTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len

        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads

        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        self.layers = nn.ModuleList([TransformerLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Linear(d_model, num_classes)

    def _create_positional_encoding(self, max_seq_len, d_model):
        # Positional Encoding: PE(pos, 2i) = sin(pos/10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)  # (1, max_seq_len, d_model)

    def forward_torch(self, X):
        batch_size, seq_len, d_model = X.shape

        pos_encoding = self.positional_encoding[:, :seq_len, :].to(X.device)
        pos_encoding = pos_encoding.expand(batch_size, -1, -1)  # Expand for batch
        X = X + pos_encoding

        for layer in self.layers:
            X = layer(X)

        X = self.layer_norm(X)

        pooled = torch.mean(X, dim=1)  # (batch, d_model)

        logits = self.classifier(pooled)

        return logits, None

    def forward(self, X):
        if len(X.shape) == 4:
            N, H, W, C = X.shape
            X = X.reshape(N, H * W, C)
            if C != self.d_model:
                proj = nn.Linear(C, self.d_model).to(X.device)
                X = proj(X)
        logits, attention_weights = self.forward_torch(X)
        predictions = F.softmax(logits, dim=1)
        return predictions, attention_weights

    def compute_loss(self, predictions, targets):
        return F.cross_entropy(predictions, targets)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerLayer, self).__init__()

        # Multi-head attention
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X):
        norm_X = self.norm1(X)
        attn_output = self.self_attention(norm_X)

        X = X + self.dropout1(attn_output)
        norm_X = self.norm2(X)

        ff_output = self.feed_forward(norm_X)
        X = X + self.dropout2(ff_output)

        return X


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, X):
        Q = self.W_q(X)
        K = self.W_k(X)
        V = self.W_v(X)

        Q = Q.unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        K = K.unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)
        V = V.unflatten(-1, (self.num_heads, self.d_k)).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attention_weights, V)

        attn_output = attn_output.transpose(1, 2).flatten(2)

        output = self.W_o(attn_output)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        # Two linear transformations with ReLU activation
        self.linear1 = nn.Linear(d_model, d_ff)  # W_1 ∈ R^(d_model × d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)  # W_2 ∈ R^(d_ff × d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        z = self.linear1(x)
        h = F.relu(z)
        h = self.dropout(h)
        output = self.linear2(h)

        return output

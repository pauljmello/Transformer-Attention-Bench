import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseAttention(nn.Module):
    def __init__(self, d_model, num_classes, max_input_dim=3072):
        super(BaseAttention, self).__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_input_dim = max_input_dim
        self.spatial_proj = nn.Linear(max_input_dim, d_model)
        self.seq_proj = nn.Linear(max_input_dim, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def _convert_input(self, X):
        if len(X.shape) == 4:
            N, H, W, C = X.shape
            X = X.reshape(N, H * W, C)
            if C != self.d_model:
                if C <= self.max_input_dim:
                    padded = torch.zeros(N, H * W, self.max_input_dim, device=X.device)
                    padded[:, :, :C] = X
                    X = self.spatial_proj(padded)
                else:
                    X = self.spatial_proj(X[:, :, :self.max_input_dim])
        elif len(X.shape) == 3:
            if X.shape[-1] != self.d_model:
                if X.shape[-1] <= self.max_input_dim:
                    padded = torch.zeros(X.shape[0], X.shape[1], self.max_input_dim, device=X.device)
                    padded[:, :, :X.shape[-1]] = X
                    X = self.seq_proj(padded)
                else:
                    X = self.seq_proj(X[:, :, :self.max_input_dim])
        return X

    def forward_torch(self, X):
        raise NotImplementedError("Subclasses must implement forward_torch")

    def forward(self, X):
        X = self._convert_input(X)
        logits, attention_weights = self.forward_torch(X)
        predictions = F.softmax(logits, dim=1)
        return predictions, attention_weights

    def compute_loss(self, predictions, targets):
        return F.cross_entropy(predictions, targets)

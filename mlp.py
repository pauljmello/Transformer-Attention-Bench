import torch.nn as nn
import torch.nn.functional as F

from attention_layers import AttentionLayer


class MLP(nn.Module):
    def __init__(self, input_size=3072, hidden_size=512, num_classes=10, attention_type=None, attention_params=None):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU(inplace=True)

        self.attention = None
        if attention_type is not None:
            params = attention_params or {}
            self.attention = AttentionLayer(hidden_size, attention_type, **params)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        X = X.flatten(1)

        h1 = self.fc1(X)
        h1 = self.relu1(h1)

        attention_weights = None
        if self.attention is not None:
            h1_seq = h1.unsqueeze(1)
            h1_attended, attention_weights = self.attention(h1_seq)
            h1 = h1_attended.squeeze(1)

        h2 = self.fc2(h1)
        h2 = self.relu2(h2)

        logits = self.fc3(h2)
        return logits, attention_weights

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)

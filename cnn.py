import torch.nn as nn
import torch.nn.functional as F

from attention_layers import AttentionLayer


class CNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10, hidden_size=64, attention_type=None, attention_params=None, prefer_channels_last=True):
        super(CNN, self).__init__()

        base_channels = max(32, hidden_size // 4)

        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(base_channels, hidden_size, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True)
        )

        if attention_type is not None:
            params = attention_params or {}
            self.attention = AttentionLayer(hidden_size, attention_type, **params)
        else:
            self.attention = nn.Identity()

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        x = self.block1(X)
        x = self.block2(x)

        if isinstance(self.attention, nn.Identity):
            x, attention_weights = x, None
        else:
            x, attention_weights = self.attention(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.flatten(1)
        logits = self.fc(x)
        return logits, attention_weights

    def compute_loss(self, logits, targets):
        return F.cross_entropy(logits, targets)

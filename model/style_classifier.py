import torch.nn as nn


class StyleClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

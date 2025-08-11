import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class ModalityDiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # 2 classes: speech or text
        )

    def forward(self, x, lambd=1.0):
        x = GradReverse.apply(x, lambd)
        return self.net(x)

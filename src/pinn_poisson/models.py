import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, depth=4, out_dim=1, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def build_model(cfg):
    return MLP(in_dim=2, hidden=cfg["model"]["hidden"], depth=cfg["model"]["depth"], out_dim=1)

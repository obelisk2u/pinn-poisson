import torch

def sample_interior(n, device):
    xy = torch.rand(n, 2, device=device)
    xy.requires_grad_(True)
    return xy

def sample_boundary(n, device):
    s = torch.rand(n, 1, device=device)
    z = torch.zeros_like(s)
    o = torch.ones_like(s)
    edges = torch.randint(0, 4, (n, 1), device=device)
    top = torch.cat([s, o], dim=1)
    bottom = torch.cat([s, z], dim=1)
    left = torch.cat([z, s], dim=1)
    right = torch.cat([o, s], dim=1)
    xy = torch.where((edges == 0).expand(-1, 2), top,
         torch.where((edges == 1).expand(-1, 2), bottom,
         torch.where((edges == 2).expand(-1, 2), left, right)))
    xy.requires_grad_(True)
    return xy

import torch, numpy as np
from .physics import u_true

def evaluate_on_grid(model, N=121, device=None):
    device = device or next(model.parameters()).device
    x = torch.linspace(0,1,N,device=device)
    y = torch.linspace(0,1,N,device=device)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)
    with torch.no_grad():
        up = model(grid).reshape(N, N).cpu().numpy()
    ut = u_true(grid).reshape(N, N).cpu().numpy()
    err = np.abs(up - ut)
    return up, ut, err

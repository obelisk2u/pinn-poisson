import torch, numpy as np
from .models import build_model
from .physics import pde_residual, bc_residual
from .sampling import sample_interior, sample_boundary

def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    hist = []

    for it in range(cfg["train"]["steps"]):
        opt.zero_grad()
        xi = sample_interior(cfg["train"]["batch_int"], device)
        xb = sample_boundary(cfg["train"]["batch_bnd"], device)
        r_pde = pde_residual(model, xi)
        r_bc = bc_residual(model, xb)
        loss = cfg["loss"]["w_pde"]*(r_pde**2).mean() + cfg["loss"]["w_bc"]*(r_bc**2).mean()
        loss.backward()
        opt.step()
        hist.append([loss.item()])
    return model, np.array(hist)

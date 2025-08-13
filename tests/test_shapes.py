import torch
from src.pinn_poisson.models import MLP
def test_forward_shape():
    m = MLP()
    x = torch.rand(5,2,requires_grad=True)
    y = m(x)
    assert y.shape == (5,1)

import torch, yaml
__version__ = "0.1.0"

from .models import MLP, build_model
from .physics import u_true, f_true, pde_residual, bc_residual
from .sampling import sample_interior, sample_boundary
from .train import train
from .evaluate import evaluate_on_grid

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(path):
    return yaml.safe_load(open(path))

__all__ = [
    "MLP",
    "build_model",
    "u_true",
    "f_true",
    "pde_residual",
    "bc_residual",
    "sample_interior",
    "sample_boundary",
    "train",
    "evaluate_on_grid",
    "get_device",
    "load_config",
]

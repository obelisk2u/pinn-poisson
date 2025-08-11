import math, torch
from torch.autograd import grad

pi = math.pi

def u_true(xy):
    x, y = xy[:, :1], xy[:, 1:]
    return torch.sin(pi*x)*torch.sin(pi*y)

def f_true(xy):
    x, y = xy[:, :1], xy[:, 1:]
    return -2*(pi**2)*torch.sin(pi*x)*torch.sin(pi*y)

def laplacian(u, xy):
    g = grad(u, xy, torch.ones_like(u), create_graph=True)[0]
    ux, uy = g[:, :1], g[:, 1:]
    uxx = grad(ux, xy, torch.ones_like(ux), create_graph=True)[0][:, :1]
    uyy = grad(uy, xy, torch.ones_like(uy), create_graph=True)[0][:, 1:]
    return uxx + uyy

def pde_residual(model, xy):
    u = model(xy)
    return laplacian(u, xy) + f_true(xy)

def bc_residual(model, xy_b):
    u_b = model(xy_b)
    return u_b

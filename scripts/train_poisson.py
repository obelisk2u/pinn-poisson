import argparse, yaml
from src.pinn_poisson.train import train
from src.pinn_poisson.evaluate import evaluate_on_grid
from src.pinn_poisson.viz import plot_loss, plot_fields
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/poisson_zeros.yaml")
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))
    model, hist = train(cfg)
    up, ut, err = evaluate_on_grid(model, cfg["eval"]["grid_N"])
    plot_loss(hist)
    plot_fields(up, ut, err)
    plt.show()

if __name__ == "__main__":
    main()

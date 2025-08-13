import matplotlib.pyplot as plt

def plot_loss(hist):
    plt.figure()
    plt.plot(hist[:,0])
    plt.title("Loss")
    plt.xlabel("step"); plt.ylabel("loss"); plt.tight_layout()

def plot_fields(up, ut, err):
    for arr, title in [(up, "u_pred"), (ut, "u_true"), (err, "|err|")]:
        plt.figure()
        plt.imshow(arr, origin="lower", extent=[0,1,0,1])
        plt.colorbar(); plt.title(title); plt.tight_layout()

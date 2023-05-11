import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

datas = [[[] for _ in range(16)] for _ in range(16)]
decays = [[[] for _ in range(16)] for _ in range(16)]
starts = [[[] for _ in range(16)] for _ in range(16)]

for i in range(16):
    for j in range(16):
        with open(f"output/results_{i}_{j}.pk", "rb") as f:
            data = pickle.load(f)
            datas[i][j] = data["pairs"]
            decays[i][j] = data["decay"]
            starts[i][j] = data["start"]

def better_log_norm(vmin, vmax, ramp_factor=8):
    range = vmax - vmin
    scale_outer = np.log(ramp_factor + 1)
    def forward(x):
        scaled = (x - vmin) / range
        return np.log(ramp_factor * scaled + 1) / scale_outer

    def inverse(x):
        return (np.exp(x * scale_outer) - 1) / ramp_factor
    
    return colors.FuncNorm((forward, inverse), vmin=vmin, vmax=vmax)

for i in range(16):
    print(i)
    for j in range(16):
        fig = plt.figure(figsize=(10, 5))
        
        if len(datas[i][j]) != 0:
            first, second = zip(*datas[i][j])
            both = first + second
        else:
            first = []
            second = []
            both = []
        
        plt.hist(both, bins=50, alpha=0.5, range=(0,99), label="Both")
        plt.hist(first, bins=50, alpha=0.5, range=(0,99), label="First")
        plt.hist(second, bins=50, alpha=0.5, range=(0,99), label="Second")
        # title, vals shown to 3 decimal places
        plt.title(f"Decay {decays[i][j]:.3f}, Start {starts[i][j]:.3f}")
        plt.xlabel("Pair member index")
        plt.ylabel("Frequency per run")
        plt.legend()
        # scale yticks by 1/1024
        ax = plt.gca()
        ax.set_ylim([0, 2048])
        plt.yticks([2048/10 * i for i in range(11)], [f"{0.1*i:.1f}" for i in range(11)])
        plt.savefig(f"plots/hist/hist_{i}_{j}.png")
        plt.close(fig)

        dist = np.zeros((100, 100))
        
        both_orderings = [(x, y) for x, y in datas[i][j]] + [(y, x) for x, y in datas[i][j]]

        for x, y in both_orderings:
            dist[x, y] += 1
        
        row_sums = dist.sum(axis=1)
        dist = dist / row_sums[:, np.newaxis]
        dist[np.isnan(dist)] = 0

        norm = better_log_norm(0, 1)

        fig = plt.figure(figsize=(10, 10))
        plt.imshow(dist, norm=norm)
        plt.colorbar()
        plt.title(f"Decay {decays[i][j]:.3f}, Start {starts[i][j]:.3f}")
        plt.xlabel("Partner location")
        plt.ylabel("Feature")
        plt.savefig(f"plots/dist/dist_{i}_{j}.png")
        plt.close(fig)
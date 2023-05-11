import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import scipy.stats as stats

datas = [[[] for _ in range(16)] for _ in range(16)]
decays = np.zeros((16, 16))
starts = np.zeros((16, 16))

for i in range(16):
    for j in range(16):
        with open(f"output/results_{i}_{j}.pk", "rb") as f:
            data = pickle.load(f)
            datas[i][j] = data["pairs"]
            decays[i, j] = data["decay"]
            starts[i, j] = data["start"]

both_hist = np.zeros((16, 16, 50))
first_hist = np.zeros((16, 16, 50))
second_hist = np.zeros((16, 16, 50))

for i in range(16):
    for j in range(16):
        if len(datas[i][j]) != 0:
            first, second = zip(*datas[i][j])
            both = first + second
        else:
            first = []
            second = []
            both = []
        
        both_hist[i, j] = np.histogram(both, bins=50, range=(0,99), density=True)[0]
        first_hist[i, j] = np.histogram(first, bins=50, range=(0,99), density=True)[0]
        second_hist[i, j] = np.histogram(second, bins=50, range=(0,99), density=True)[0]

#both_hist = np.nan_to_num(both_hist)
#first_hist = np.nan_to_num(first_hist)
#second_hist = np.nan_to_num(second_hist)

bins = np.arange(50)

# for each decay (first index) plot mean of the hist against start (second index)

hues = np.linspace(0.5, 1, 16)

# convert hues to 0 < x < 1
hues = hues % 1

rgb_vals = np.array([colors.hsv_to_rgb((hue, 0.75, 0.85)) for hue in hues])

both_mean = both_hist.dot(bins)
first_mean = first_hist.dot(bins)
second_mean = second_hist.dot(bins)

both_variance = both_hist.dot(bins**2) - both_mean**2
first_variance = first_hist.dot(bins**2) - first_mean**2
second_variance = second_hist.dot(bins**2) - second_mean**2

both_skewness = both_hist.dot(bins**3) - 3 * both_mean * both_variance - both_mean**3
both_skewness = both_skewness / both_variance**(3/2)
first_skewness = first_hist.dot(bins**3) - 3 * first_mean * first_variance - first_mean**3
first_skewness = first_skewness / first_variance**(3/2)
second_skewness = second_hist.dot(bins**3) - 3 * second_mean * second_variance - second_mean**3
second_skewness = second_skewness / second_variance**(3/2)

# delete zero values (set to nan)

"""
both_mean[both_mean == 0] = np.nan
first_mean[first_mean == 0] = np.nan
second_mean[second_mean == 0] = np.nan

both_variance[both_variance == 0] = np.nan
first_variance[first_variance == 0] = np.nan
second_variance[second_variance == 0] = np.nan

both_skewness[both_skewness == 0] = np.nan
first_skewness[first_skewness == 0] = np.nan
second_skewness[second_skewness == 0] = np.nan
"""

# plot images of mean, variance, skewness

plots = [both_mean, first_mean, second_mean, both_variance, first_variance, second_variance, both_skewness, first_skewness, second_skewness]
stat_name = ["Mean", "Mean", "Mean", "Variance", "Variance", "Variance", "Skewness", "Skewness", "Skewness"]
filenames = ["both_mean", "first_mean", "second_mean", "both_variance", "first_variance", "second_variance", "both_skewness", "first_skewness", "second_skewness"]
titles = ["Mean index of antipodally-stored features", "Mean index of first member of antipodal pair", "Mean index of second member of antipodal pair", "Variance of index of antipodally-stored features", "Variance of index of first member of antipodal pair", "Variance of index of second member of antipodal pair", "Skewness of index of antipodally-stored features", "Skewness of index of first member of antipodal pair", "Skewness of index of second member of antipodal pair"]

for plot, name, filename, title in zip(plots, stat_name, filenames, titles):
    fig = plt.figure(figsize=(10, 5))
    for i in range(16):
        plt.plot(starts[i], plot[i], label=f"Decay {decays[i][0]:.3f}", color=rgb_vals[i])
    plt.xlabel("Start")
    plt.ylabel(name)
    plt.title(title)

    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f"summary_stats/{filename}_lines.png")
    plt.close(fig)

    # plot grid
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(plot)
    plt.colorbar()
    plt.xlabel("Start")
    plt.ylabel("Decay")
    plt.title(title)
    idxs = np.array(range(8)) * 2
    plt.xticks(idxs, [f"{starts[0][i]:.3f}" for i in idxs])
    plt.yticks(idxs, [f"{decays[i][0]:.3f}" for i in idxs])
    plt.savefig(f"summary_stats/{filename}_grid.png")
    plt.close(fig)

"""
fig = plt.figure(figsize=(10, 5))
for i in range(16):
    plt.plot(starts[i], both_mean[i], label=f"Decay {decays[i][0]:.3f}", color=rgb_vals[i])
plt.legend(loc="upper right")
plt.xlabel("Start")
plt.ylabel("Mean index")
plt.title("Mean index of antipodally-stored features")
plt.savefig("both_means_decay.png")
plt.close(fig)

fig = plt.figure(figsize=(10, 5))
for i in range(16):
    plt.plot(starts[i], first_mean[i], label=f"Decay {decays[i][0]:.3f}", color=rgb_vals[i])
plt.legend(loc="upper right")
plt.xlabel("Start")
plt.ylabel("Mean index")
plt.title("Mean index of first member of antipodal pair")
plt.savefig("first_mean_decay.png")
plt.close(fig)

fig = plt.figure(figsize=(10, 5))
for i in range(16):
    plt.plot(starts[i], second_mean[i], label=f"Decay {decays[i][0]:.3f}", color=rgb_vals[i])
plt.legend(loc="upper right")
plt.xlabel("Start")
plt.ylabel("Mean index")
plt.title("Mean index of second member of antipodal pair")
plt.savefig("second_mean_decay.png")

# for each start (second index) plot mean of the hist against decay (first index)

fig = plt.figure(figsize=(10, 5))
for i in range(16):
    plt.plot(decays[:, i], both_mean[:, i], label=f"Start {starts[0][i]:.3f}", color=rgb_vals[i])
plt.legend(loc="upper left")
plt.xlabel("Decay")
plt.ylabel("Mean index")
plt.title("Mean index of antipodally-stored features")
plt.savefig("both_means_start.png")
plt.close(fig)

fig = plt.figure(figsize=(10, 5))
for i in range(16):
    plt.plot(decays[:, i], first_mean[:, i], label=f"Start {starts[0][i]:.3f}", color=rgb_vals[i])
plt.legend(loc="upper left")
plt.xlabel("Decay")
plt.ylabel("Mean index")
plt.title("Mean index of first member of antipodal pair")
plt.savefig("first_mean_start.png")
plt.close(fig)

fig = plt.figure(figsize=(10, 5))
for i in range(16):
    plt.plot(decays[:, i], second_mean[:, i], label=f"Start {starts[0][i]:.3f}", color=rgb_vals[i])
plt.legend(loc="upper left")
plt.xlabel("Decay")
plt.ylabel("Mean index")
plt.title("Mean index of second member of antipodal pair")
plt.savefig("second_mean_start.png")
plt.close(fig)
"""

pair_lengths = np.zeros((16, 16))

for i in range(16):
    for j in range(16):
        n_pairs = len(datas[i][j])

        sparsity = starts[i][j] * decays[i][j] ** np.arange(100)

        sum_pair_lengths = sum([sparsity[x]/sparsity[y] for x, y in datas[i][j]])
        if n_pairs != 0:
            pair_lengths[i, j] = sum_pair_lengths / n_pairs
        else:
            pair_lengths[i, j] = np.nan

# cut two leftmost columns

pair_lengths = pair_lengths[:, 2:]

# plot lines + grid

fig = plt.figure(figsize=(10, 5))
for i in range(16):
    plt.plot(starts[i, 2:], pair_lengths[i], label=f"Decay {decays[i][0]:.3f}", color=rgb_vals[i])
plt.xlabel("Start")
plt.ylabel("Mean pair length")
plt.title("Mean pair sparsity ratio")

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("summary_stats/pair_lengths_lines.png")
plt.close(fig)

fig = plt.figure(figsize=(10, 10))
plt.imshow(pair_lengths)
plt.colorbar()
plt.xlabel("Start")
plt.ylabel("Decay")
plt.title("Mean pair sparsity ratio")
idxs = np.array(range(8)) * 2
plt.xticks(np.arange(7)*2, [f"{starts[0][i+2]:.3f}" for i in np.arange(7) * 2])
plt.yticks(idxs, [f"{decays[i][0]:.3f}" for i in idxs])
plt.savefig("summary_stats/pair_lengths_grid.png")
plt.close(fig)
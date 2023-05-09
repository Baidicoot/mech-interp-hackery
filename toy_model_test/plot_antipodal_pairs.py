import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.seterr(divide='ignore', invalid='ignore')

argparser = argparse.ArgumentParser()

argparser.add_argument("data_folder", type=str)

# args: data folder, output folder (in that order)
# set up argparse

args = argparser.parse_args()

print(args)

data_folder = args.data_folder

acts_to_test = ["quad", "cube", "quart", "relu", "gelu", "smooth_abs", "gelu_smooth_abs"]

#acts_to_test = ["relu"]

# load from "toy_model/anthropic_toy_{act}.pk"

loss_curves = {}
weights = {}
biases = {}

for act in acts_to_test:
    with open(f"{data_folder}/anthropic_toy_{act}.pk", "rb") as f:
        data = pickle.load(f)
        loss_curves[act] = data["losses"]
        weights[act] = data["weights"]
        biases[act] = data["bias"]

def count_antipodal(w):
    w_T_w = np.einsum('i f r, i g r -> i f g', w, w)
    
    # take the lower triangle

    w_T_w_lt = np.tril(w_T_w, k=-1)

    # count the number of places with < -0.8 interference

    pairs = []

    for i in range(w_T_w_lt.shape[0]):
        if i % 64 == 0:
            print("Instance", i, "Pairs", len(pairs))
        for x in range(w_T_w_lt.shape[1]):
            for y in range(w_T_w_lt.shape[2]):
                if w_T_w_lt[i, x, y] < -0.6:
                    if x > y:
                        pairs.append((y, x))
                    else:
                        pairs.append((x, y))
    
    return pairs

# for each act, make a histogram of the location of the antipodal pairs

for act in acts_to_test:
    pairs = count_antipodal(weights[act])

    if len(pairs) == 0:
        continue

    firsts, seconds = zip(*pairs)

    bins = np.linspace(0, 100, 100)

    plt.figure(figsize=(10, 5))
    plt.hist(firsts + seconds, bins, alpha=0.5, label="Both")
    plt.hist(firsts, bins, alpha=0.5, label="Primary")
    plt.hist(seconds, bins, alpha=0.5, label="Secondary")
    plt.title(f"Antipodal pairs for {act}")
    plt.legend()
    plt.savefig(f"plots/antipodal_pairs_{act}.png")
    plt.close()

#pairs = count_antipodal(weights["relu"])
pairs = pairs + [(y, x) for x, y in pairs]

dist = np.zeros((100, 100))

for x, y in pairs:
    dist[x, y] += 1

# norm rows, filling nans with 0

row_sums = dist.sum(axis=1)
dist = dist / row_sums[:, np.newaxis]
dist[np.isnan(dist)] = 0

plt.figure(figsize=(10, 7.5))
plt.imshow(dist)
plt.colorbar()
plt.xlabel("Partner location")
plt.ylabel("Feature")
plt.title("Distribution of antipodal pairs")
plt.savefig("plots/antipodal_pairs_dist.png")

# expected locations

loc = np.einsum("f, x f -> x", np.arange(100), dist)

plt.figure(figsize=(10, 5))
plt.plot(loc)
plt.xlabel("Feature")
plt.ylabel("Expected location of partner")
plt.title("Expected location of partner")
plt.savefig("plots/antipodal_pairs_expected.png")
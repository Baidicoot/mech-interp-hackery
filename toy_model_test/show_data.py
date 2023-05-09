import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument("data_folder", type=str)

# args: data folder, output folder (in that order)
# set up argparse

args = argparser.parse_args()

print(args)

data_folder = args.data_folder

acts_to_test = ["linear", "quad", "cube", "quart", "relu", "gelu", "smooth_abs", "gelu_smooth_abs", "sigmoid", "tanh"]

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

# plot loss curves

plt.figure(figsize=(10, 5))
for act in acts_to_test:
    plt.plot(loss_curves[act], label=act)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curves")
plt.loglog()
plt.savefig(f"plots/loss_curves.png")

#sparsity = np.array([0.5 * 0.93 ** i for i in range(100)])
sparsity = [0.24 for _ in range(25)] + [0.039 for _ in range(25)] + [0.006 for _ in range(25)] + [0.001 for _ in range(25)]

def get_cool_weight_data(w1):
    # length of embeddings
    lengths = np.linalg.norm(w1, axis=2).mean(axis=0)

    w_T_w = np.einsum('i f r, i g r -> i f g', w1, w1).mean(axis=0)

    # angle between embeddings
    angles = np.arccos(w_T_w / np.outer(lengths, lengths)) / np.pi * 180

    # set diagonal to 90 degrees
    np.fill_diagonal(angles, 90)

    w_T_w_nd = w_T_w - np.diag(np.diag(w_T_w))

    # interference per feature (sum of w_T_w - diagonal)
    interference = w_T_w_nd.sum(axis=1) / (w_T_w.shape[1] - 1)

    # expected interference for feature i (\sum_{j \neq i} sparsity[j] * (W_i \cdot W_j))

    expected_interference = w_T_w_nd @ sparsity

    return lengths, angles, interference, w_T_w_nd, expected_interference

for act in acts_to_test:
    lengths, angles, interference, w_T_w_clip, expected_interference = get_cool_weight_data(weights[act])

    # plot length of embeddings

    plt.figure(figsize=(10, 5))
    plt.plot(lengths)
    plt.xlabel("Feature")
    plt.ylabel("Length")
    plt.title(f"Length of embeddings ({act})")
    plt.savefig(f"plots/embedding_lengths_{act}.png")

    # plot angle between embeddings

    #plt.figure(figsize=(7.5, 7.5))
    #plt.imshow(angles)
    #plt.colorbar()
    #plt.xlabel("Feature")
    #plt.ylabel("Feature")
    #plt.title(f"Angle between embeddings ({act})")
    #plt.savefig(f"plots/embedding_angles_{act}.png")


    # plot w_T_w_nd

    plt.figure(figsize=(7.5, 7.5))
    plt.imshow(w_T_w_clip)
    plt.colorbar()
    plt.xlabel("Feature")
    plt.ylabel("Feature")
    plt.title(f"w.T @ w ({act})")
    plt.savefig(f"plots/w_T_w_nd_{act}.png")

    # plot expected interference per feature

    plt.figure(figsize=(10, 5))
    plt.plot(expected_interference)
    plt.xlabel("Feature")
    plt.ylabel("Expected interference")
    plt.title(f"Expected interference ({act})")
    plt.savefig(f"plots/expected_interference_{act}.png")

    # expected interference vs embedding length

    #fig, ax = plt.subplots(figsize=(10, 5))
    #ax2 = ax.twinx()

    #ax.plot(expected_interference, label="Expected Interference")
    #ax2.plot(lengths, label="Embedding Length", color="orange")
    #ax.set_xlabel("Feature")
    #ax.set_ylabel("Expected Interference")
    #ax2.set_ylabel("Embedding Length")
    #ax.set_title(f"Expected Interference vs Embedding Length ({act})")
    #fig.legend()
    #fig.savefig(f"plots/expected_interference_vs_embedding_length_{act}.png")

    # expected interference vs bias

    #fig, ax = plt.subplots(figsize=(10, 5))

    #ax.plot(expected_interference, label="Expected Interference")
    #ax.plot(biases[act].mean(axis=0), label="Bias")
    #ax.set_xlabel("Feature")
    #ax.set_ylabel("Expected Interference/Bias")
    #ax.set_title(f"Expected Interference vs Bias ({act})")
    #fig.legend()
    #fig.savefig(f"plots/expected_interference_vs_bias_{act}.png")

    # embedding length vs bias

    #fig, ax = plt.subplots(figsize=(10, 5))

    #ax.plot(lengths, label="Embedding Length")
    #ax.plot(biases[act].mean(axis=0), label="Bias")
    #ax.set_xlabel("Feature")
    #ax.set_ylabel("Embedding Length/Bias")
    #ax.set_title(f"Embedding Length vs Bias ({act})")
    #fig.legend()
    #fig.savefig(f"plots/embedding_length_vs_bias_{act}.png")
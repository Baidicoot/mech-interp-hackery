import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

with open("output/anthropic_toy_cube.pk", "rb") as f:
    data = pickle.load(f)
    weights_sparsity = data["weights"]
    biases_sparsity = data["bias"]

sparsity = np.array([0.5 * 0.93 ** i for i in range(100)])

def clean_w(w):
    w_T_w = np.einsum('f r, g r -> f g', w, w)
    w_T_w_nd = w_T_w - np.diag(np.diag(w_T_w))
    return w_T_w_nd

# plot clean_w for model #0-#20 on a single plot

# make 16 subplots

fig, axs = plt.subplots(5, 4, figsize=(20, 20))
fig.tight_layout()

images = []

for i in range(5):
    for j in range(4):
        img = clean_w(weights_sparsity[i * 4 + j])
        images.append(axs[i, j].imshow(img))
        axs[i, j].label_outer()

# Find the min and max of all colors for use in setting the color scale.
vmin = min(image.get_array().min() for image in images)
vmax = max(image.get_array().max() for image in images)
norm = colors.Normalize(vmin=vmin, vmax=vmax)
for im in images:
    im.set_norm(norm)

fig.colorbar(images[0], ax=axs)

plt.legend()
plt.savefig("example_no_monosemantic.png")

# plot the mean w_T_w

plt.figure(figsize=(10, 5))

w_T_w = np.einsum('i f r, i g r -> i f g', weights_sparsity, weights_sparsity).mean(axis=0)
w_T_w_nd = w_T_w - np.diag(np.diag(w_T_w))

plt.imshow(w_T_w_nd)
plt.colorbar()
plt.title("Mean interference")
plt.savefig("mean_interference.png")
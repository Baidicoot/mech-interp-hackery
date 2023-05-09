import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("toy_model_sparsity/anthropic_toy_relu.pk", "rb") as f:
    data = pickle.load(f)
    weights_sparsity = data["weights"]
    biases_sparsity = data["bias"]

with open("toy_model_importance/anthropic_toy_relu.pk", "rb") as f:
    data = pickle.load(f)
    weights_importance = data["weights"]
    biases_importance = data["bias"]

sparsity = np.array([0.5 * 0.93 ** i for i in range(100)])

def get_expected_interference(w, sparsity):
    w_T_w = np.einsum('i f r, i g r -> i f g', w, w).mean(axis=0)
    w_T_w_nd = w_T_w - np.diag(np.diag(w_T_w))
    return w_T_w_nd @ sparsity * 0.5

# plot expected interference of the two models

plt.figure(figsize=(10, 5))
plt.plot(get_expected_interference(weights_sparsity, sparsity), label="Uniform importance")
plt.plot(get_expected_interference(weights_importance, np.mean(sparsity).repeat(100)), label="Uniform sparsity")
plt.legend()
plt.xlabel("Feature")
plt.ylabel("Expected interference")
plt.title("Expected interference per feature")
plt.show()
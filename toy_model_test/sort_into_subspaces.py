import numpy as np
import pickle

def sort_into_subspaces(features, wTw, eps=0.1):
    buckets = []

    for i in range(features):
        added = False

        if wTw[i, i] < eps:
            continue

        for bucket_i in range(len(buckets)):
            if np.all(np.abs(wTw[i, buckets[bucket_i]]) > eps):
                buckets[bucket_i].append(i)
                added = True
                break
        if not added:
            buckets.append([i])
    
    return buckets

with open(f"output/anthropic_toy_relu.pk", "rb") as f:
    data = pickle.load(f)
    loss_curves = data["losses"]
    weights = data["weights"]
    biases = data["bias"]

wTw = np.einsum('i f r, i g r -> i f g', weights, weights)

scores = np.zeros(100)

for i in range(1024):
    buckets = sort_into_subspaces(100, wTw[0])
    for bucket in buckets:
        scores[len(bucket)] += 1
    
    if i % 100 == 0:
        print(i)

# show wTw

import matplotlib.pyplot as plt

plt.plot(scores)
plt.show()

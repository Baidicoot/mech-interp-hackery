import matplotlib.pyplot as plt
import pickle
import numpy as np

# load data

with open('output/anthropic_toy_relu.pk', "rb") as f:
    data = pickle.load(f)
    losses = data['losses']
    weights = data['weights']
    bias = data['bias']

# plot loss

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.loglog()
plt.show()

# plot interference pattern

wTw = np.einsum('i f r, i g r -> i f g', weights, weights).mean(axis=0)

plt.imshow(wTw)
plt.colorbar()
plt.xlabel('Feature')
plt.ylabel('Feature')
plt.title('Interference Pattern')
plt.show()

# plot embedding lengths

plt.plot(np.linalg.norm(weights[0], axis=1))
plt.xlabel('Feature')
plt.ylabel('Embedding Length')
plt.title('Embedding Lengths')
plt.show()
import pickle
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt

# load data

with open('output/anthropic_toy_relu.pk', "rb") as f:
    data = pickle.load(f)
    losses = data['losses']
    weights = data['weights']
    # identity padded with zeros
    #weights = np.zeros((1, 100, 40))
    #weights[0, :40, :] = np.eye(40)

    # random rotation
    #R = sp.stats.special_ortho_group.rvs(40)

    #weights = np.einsum("i f k, k r -> i f r", weights, R)

    bias = data['bias']

# plot weights

plt.imshow(weights[0])
plt.colorbar()
plt.ylabel('Feature')
plt.xlabel('Representation')
plt.title('Weights')
plt.show()

# generate a bunch of random datapoints

dataset_size = 16384
x = np.random.rand(dataset_size, 100)

# sparsitify them

sparsity = np.array(sum([0.005 for _ in range(100)]))
x[np.random.rand(dataset_size, 100) > sparsity] = 0

# ICA

def g(x):
    return np.tanh(x)

def g_der(x):
    return 1 - g(x) * g(x)

def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whiten = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whiten

def center(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    return X- mean

def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis=1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new

def ica(X, iterations, tolerance=1e-5):
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]
    W = np.zeros((components_nr, components_nr), dtype=X.dtype)

    for i in range(components_nr):
        print("doing component " + str(i) + "of " + str(components_nr))
        w = np.random.rand(components_nr)
        for j in range(iterations):
            w_new = calculate_new_w(w, X)

            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])
            
            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            w = w_new
            
            if distance < tolerance:
                break
        W[i, :] = w
    S = np.dot(W, X)
    
    return S

# run ICA

y = np.einsum('i f r,b f ->i b r', weights, x)[0]

S = ica(y.T, 1000)

components = S @ y

# normalise the components

components /= np.linalg.norm(components, axis=1, keepdims=True)

with open("output/ica_components.pk", "wb") as f:
    pickle.dump(components, f)

translations = np.einsum("i f r, n r -> i f n", weights, components).mean(axis=0)

# plot ICA components

plt.imshow(translations)
plt.colorbar()
plt.ylabel('Feature')
plt.xlabel('ICA Component')
plt.title('ICA component to feature correspondence')
plt.show()
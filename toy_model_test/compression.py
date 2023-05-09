import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle

activation_functions = {
    'relu': F.relu,
    'sigmoid': torch.sigmoid,
    'tanh': torch.tanh,
    'gelu': F.gelu,
    'leaky_relu': F.leaky_relu,
    'squared_relu': lambda x: F.relu(x) ** 2,
    'smooth_abs': lambda x: torch.sqrt(x ** 2 + 1e-6),
    'gelu_smooth_abs': lambda x: F.gelu(x) + F.gelu(-x),
    'linear': lambda x: x,
    'quad': lambda x: x ** 2,
    "cube": lambda x: x ** 3,
    "quart": lambda x: x ** 4,
}

class AnthropicToyModel(nn.Module):
    def __init__(self, instances, features, representation, activation, bias=True):
        super().__init__()
        self.instances = instances
        self.features = features
        self.representation = representation
        self.w1 = nn.Parameter(torch.randn(instances, features, representation) * 0.02)
        if bias:
            self.b = nn.Parameter(torch.randn(instances, features) * 0.02)
        else:
            self.b = torch.zeros(instances, features)
        self.act = activation_functions[activation]
    
    def forward(self, x):
        x = torch.einsum('i f r,i b f ->i b r', self.w1, x)
        x = torch.einsum('i f r,i b r ->i b f', self.w1, x) + self.b.unsqueeze(1)
        x = self.act(x)
        return x

# train on the task of recovering the input

def train(model, optimizer, importance, sparsity, device="cuda", epochs=10000, batch_size=1024):
    model.to(device)
    importance = torch.tensor(importance, device=device)
    sparsity = torch.tensor(sparsity, device=device)
    losses = []
    for epoch in range(epochs):
        x = torch.rand(model.instances, batch_size, model.features, device=device)
        x[torch.rand(model.instances, batch_size, model.features, device=device) > sparsity] = 0

        optimizer.zero_grad()
        y = model(x)
        loss = ((y - model.act(x)) ** 2 * importance).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, loss {loss.item():.4f}')
    return losses

#acts_to_test = ["relu", "gelu", "linear", "quad", "squared_relu", "smooth_abs", "gelu_smooth_abs", "sigmoid"]

acts_to_test = ["sigmoid", "tanh"]

importance = [1 for _ in range(100)]
sparsity = [0.7 * 0.98 ** i for i in range(100)]
#sparsity = sum([0.5 * 0.93 ** i for i in range(100)])/100
#importance = [0.93 ** i for i in range(100)]
#sparsity = [0.24 for _ in range(25)] + [0.039 for _ in range(25)] + [0.006 for _ in range(25)] + [0.001 for _ in range(25)]

for act in acts_to_test:

    epochs = 5000

    print(f"Testing activation function '{act}'")
    model = AnthropicToyModel(1025, 100, 40, act).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = train(model, optimizer, importance=importance, sparsity=sparsity, epochs=epochs)
    with open(f"output/anthropic_toy_{act}.pk", "wb") as f:
        pickle.dump({"losses": losses, "weights": model.w1.detach().cpu().numpy(), "bias": model.b.detach().cpu().numpy()}, f)
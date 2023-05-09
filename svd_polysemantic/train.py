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

def train(model, optimizer, importance, sparsity, device="cpu", epochs=5000, batch_size=1024):
    model.to(device)
    importance = torch.tensor(importance, device=device)
    sparsity = torch.tensor(sparsity, device=device)
    losses = []
    for epoch in range(epochs):
        x = torch.rand(model.instances, batch_size, model.features, device=device)
        x[torch.rand(model.instances, batch_size, model.features, device=device) > sparsity] = 0

        optimizer.zero_grad()
        y = model(x)
        loss = (importance * torch.pow(y - x, 2)).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, loss {loss.item():.4f}')
    return losses

acts_to_test = ["relu"]

sparsity = sum([0.0005 for _ in range(100)])
importance = [0.93 ** i for i in range(100)]
for act in acts_to_test:
    print(f"Training activation function '{act}'")
    model = AnthropicToyModel(1, 100, 40, act).to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = train(model, optimizer, importance=importance, sparsity=sparsity)
    with open(f"output/anthropic_toy_{act}.pk", "wb") as f:
        pickle.dump({"losses": losses, "weights": model.w1.detach().cpu().numpy(), "bias": model.b.detach().cpu().numpy()}, f)
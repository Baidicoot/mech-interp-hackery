import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pickle

class AnthropicToyModel(nn.Module):
    def __init__(self, instances, features, representation, bias=True):
        super().__init__()
        self.instances = instances
        self.features = features
        self.representation = representation
        self.w1 = nn.Parameter(torch.randn(instances, features, representation) * 0.02)
        if bias:
            self.b = nn.Parameter(torch.randn(instances, features) * 0.02)
        else:
            self.b = torch.zeros(instances, features)
    
    def forward(self, x):
        x = torch.einsum('i f r,i b f ->i b r', self.w1, x)
        x = torch.einsum('i f r,i b r ->i b f', self.w1, x) + self.b.unsqueeze(1)
        x = F.relu(x)
        return x

def count_antipodal(wTw):
    dataset = []
    for iter in range(wTw.size(0)):
        for first in range(wTw.size(1)-1):
            for second in range(first+1, wTw.size(2)):
                if wTw[iter, first, second] < -0.8:
                    dataset.append((first, second))
    return dataset

def train(model, optimizer, sparsity, device="cuda", epochs=2000, batch_size=1024):
    model.to(device)
    sparsity = torch.tensor(sparsity, device=device)
    losses = []
    for epoch in range(epochs):
        x = torch.rand(model.instances, batch_size, model.features, device=device)
        x[torch.rand(model.instances, batch_size, model.features, device=device) > sparsity] = 0
        optimizer.zero_grad()
        y = model(x)
        loss = F.mse_loss(y, x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % 500 == 0:
            print(f'Epoch {epoch}, loss {loss.item():.4f}')
    return losses

def train_instances(sparsity_decay_vals, sparsity_start_vals):
    for i, decay in enumerate(sparsity_decay_vals):
        for j, start in enumerate(sparsity_start_vals):
            print(f"training model with decay {decay} and start {start}")
            model = AnthropicToyModel(1024, 100, 40)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            losses = train(model, optimizer, [start * decay ** i for i in range(100)])
            wTw = torch.einsum('i f r, i g r -> i f g', model.w1, model.w1)
            dataset = count_antipodal(wTw)
            with open(f"output/results_{i}_{j}.pk", "wb") as f:
                pickle.dump({"losses": losses, "pairs": dataset, "start": start, "decay": decay}, f)

decay_vals = np.linspace(0.9, 0.99, 16)
start_vals = np.linspace(0.01, 1, 16)

train_instances(decay_vals, start_vals)
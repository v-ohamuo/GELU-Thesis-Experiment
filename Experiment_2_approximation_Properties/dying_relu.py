import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- Target function ---
def f3(x):
    return 0.1 * x - 1

N_train = 200
x_train = np.linspace(-2, 2, N_train).reshape(-1, 1)
y_train = f3(x_train)

x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

class DeepNet(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(1, 125),
            activation_fn,
            nn.Linear(125, 125),
            activation_fn,
            nn.Linear(125, 125),
            activation_fn,
            nn.Linear(125, 125),
            activation_fn,
            nn.Linear(125, 1)
        ])
    
    def forward(self, x, return_hidden=False):
        hidden_acts = []
        out = x
        for layer in self.layers:
            out = layer(out)
            if isinstance(layer, (nn.ReLU, nn.GELU)):
                hidden_acts.append(out.detach().cpu().numpy())
        if return_hidden:
            return out, hidden_acts
        return out

def train_model(model, x_train, y_train, epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()

activations = {}

for name, activation in [
    ("ReLU", nn.ReLU()),
    ("GELU", nn.GELU())
]:
    print(f"Training with {name} activation...")
    model = DeepNet(activation)
    train_model(model, x_train_t, y_train_t)
    model.eval()
    with torch.no_grad():
        _, hidden_acts = model(x_train_t, return_hidden=True)
    activations[name] = hidden_acts[-1].flatten()  

for name, acts in activations.items():
    dead_ratio = np.mean(np.isclose(acts, 0.0, atol=1e-6))
    total_neurons = acts.shape[0]
    dead_neurons = int(dead_ratio * total_neurons)
    print(f"{name}: {dead_neurons}/{total_neurons} neurons ({dead_ratio*100:.2f}%) inactive (~0 activation)")

plt.figure(figsize=(7, 5))
for name, color in zip(["ReLU", "GELU"], ['red', 'blue']):
    plt.hist(activations[name], bins=50, alpha=0.6, label=name, color=color)
plt.title("Distribution of Hidden Activations (Dying ReLU Visualization)")
plt.xlabel("Activation Value")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_exp2_dying_relu_histogram.png", dpi=300)
plt.show()

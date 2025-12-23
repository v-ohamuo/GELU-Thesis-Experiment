import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- Target function ---
def f2(x):
    return 0.5 * x**2

N_train = 200
x_train = np.linspace(-2, 2, N_train).reshape(-1, 1)
y_train = f2(x_train)

x_test = np.linspace(-4, 4, 400).reshape(-1, 1)
y_test = f2(x_test)

x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
x_test_t = torch.tensor(x_test, dtype=torch.float32)

class DeepNet(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 125),
            activation_fn,
            nn.Linear(125, 125),
            activation_fn,
            nn.Linear(125, 125),
            activation_fn,
            nn.Linear(125, 125),
            activation_fn,
            nn.Linear(125, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

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
    return loss.item()

models = {}
for name, activation in [("Tanh", nn.Tanh()), ("GELU", nn.GELU())]:
    print(f"Training with {name} activation...")
    model = DeepNet(activation)
    train_model(model, x_train_t, y_train_t)
    models[name] = model

with torch.no_grad():
    y_pred = {name: model(x_test_t).numpy() for name, model in models.items()}

plt.figure(figsize=(10, 5))
plt.plot(x_test, y_test, 'k--', label='True Function $f_2(x)=0.5x^2$')
plt.plot(x_test, y_pred["Tanh"], 'r', label='Tanh Network')
plt.plot(x_test, y_pred["GELU"], 'b', label='GELU Network')
plt.axvspan(-4, -2, color='gray', alpha=0.1)
plt.axvspan(2, 4, color='gray', alpha=0.1)
plt.title("Visualizing Saturation vs. Unboundedness")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fig_exp2_saturation_unboundedness.png", dpi=300)
plt.show()

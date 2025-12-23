import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Target function ---
def f1(x):
    return np.sin(2 * np.pi * x) + 0.5 * x

N_train, N_val = 200, 1000
x_train = np.linspace(-1, 1, N_train).reshape(-1, 1)
y_train = f1(x_train)
x_val = np.linspace(-1, 1, N_val).reshape(-1, 1)
y_val = f1(x_val)

x_train_t = torch.tensor(x_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
x_val_t = torch.tensor(x_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)

class DeepNet(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.model = nn.Sequential(
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
        return self.model(x)

def train_model(model, x_train, y_train, x_val, y_val, epochs=1000, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        y_val_pred = model(x_val)
        val_loss = loss_fn(y_val_pred, y_val).item()

    return y_val_pred.numpy(), train_losses, val_loss

results = {}
for name, activation in [("GELU", nn.GELU()), ("ReLU", nn.ReLU())]:
    print(f"\nTraining deep network with {name} activation...")
    model = DeepNet(activation)
    y_pred, losses, val_mse = train_model(model, x_train_t, y_train_t, x_val_t, y_val_t)
    results[name] = {"pred": y_pred, "losses": losses, "val_mse": val_mse}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(x_val, y_val, 'k--', label='True Function', linewidth=1.5)
axes[0].plot(x_val, results["ReLU"]["pred"], label='ReLU', color='red')
axes[0].plot(x_val, results["GELU"]["pred"], label='GELU', color='blue')
axes[0].set_title("Function Approximation (Deep Network, 4×125 Neurons)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].legend()
axes[0].grid(True)

epochs = np.arange(1, 1001)
axes[1].plot(epochs, results["ReLU"]["losses"], label='ReLU Loss', color='red')
axes[1].plot(epochs, results["GELU"]["losses"], label='GELU Loss', color='blue')
axes[1].set_title("Training Loss (MSE) vs Epochs")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("fig_exp2_deep_approx_with_loss.png", dpi=300)
plt.show()

mse_table = pd.DataFrame([
    {"Activation": k, "Validation MSE": v["val_mse"]} for k, v in results.items()
])
print("\nValidation MSE:")
print(mse_table)

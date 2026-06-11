import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

# ── Device setup ──────────────────────────────────────────────
device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "xpu":
    print(f"  GPU: {torch.xpu.get_device_name(0)}")


# ── Model ─────────────────────────────────────────────────────
class SinNet(nn.Module):
    """3-layer MLP matching the architecture in nn.py: 1→40→40→1"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1),
        )

    def forward(self, x):
        return self.net(x)


def f(x):
    return np.sin(x)


# ── Training ──────────────────────────────────────────────────
if __name__ == "__main__":
    net = SinNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-2)
    criterion = nn.MSELoss()

    epochs = 2000
    n_samples = 10000
    n_test = 100

    # Generate data
    x_np = np.linspace(-np.pi, np.pi, n_samples)
    y_np = f(x_np)
    x_test_np = np.linspace(-np.pi, np.pi, n_test)
    y_test_np = f(x_test_np)

    # Convert test data once (used in every epoch for eval)
    x_test = torch.tensor(x_test_np, dtype=torch.float32).unsqueeze(1).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

    indices = list(range(n_samples))
    losses = []

    for epoch in range(epochs):
        # Random single-point SGD (same style as original nn.py)
        idx = np.random.choice(indices)
        x0 = torch.tensor([[x_np[idx]]], dtype=torch.float32).to(device)
        y0 = torch.tensor([[y_np[idx]]], dtype=torch.float32).to(device)

        # Forward + loss + backward + step
        optimizer.zero_grad()
        y0_pred = net(x0)
        loss = criterion(y0_pred, y0)
        loss.backward()
        optimizer.step()

        # Evaluate on full test set (no grad)
        with torch.no_grad():
            y_test_pred = net(x_test)
            loss_mean = criterion(y_test_pred, y_test).item()
        losses.append(loss_mean)

        if (epoch + 1) % 200 == 0:
            print(f"epoch: {epoch+1}/{epochs} | loss: {loss_mean:.8f}")

    print(f"Final loss: {losses[-1]:.8f}")

    # ── Plot results ──────────────────────────────────────────
    with torch.no_grad():
        x_plot_np = np.linspace(-np.pi, np.pi, 100)
        x_plot = torch.tensor(x_plot_np, dtype=torch.float32).unsqueeze(1).to(device)
        y_pred_np = net(x_plot).cpu().numpy().flatten()

    y_true_np = f(x_plot_np)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(x_plot_np, y_true_np, label="True", linestyle="--")
    plt.plot(x_plot_np, y_pred_np, label="Prediction")
    plt.legend()
    plt.title("sin(x) fit")

    plt.subplot(1, 2, 2)
    plt.plot(list(range(1, epochs + 1)), losses, label="Loss")
    plt.legend()
    plt.title("Loss curve")

    plt.tight_layout()
    plt.show()

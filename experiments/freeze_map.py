import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW, RAdam
import matplotlib.pyplot as plt
import numpy as np
import os

from optimizers.dynamo import DynamoV2
from lion_pytorch import Lion


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x




def train_model(optimizer_class, name, train_loader, device, epochs=1,
                thresholds=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4], **opt_kwargs):
    model = SimpleCNN().to(device)
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    # Tracking
    threshold_counts = {th: [] for th in thresholds}
    update_hist = []
    per_layer_freeze = {i: [] for i, _ in enumerate(model.parameters())}

    prev_params = [p.data.clone() for p in model.parameters()]

    for epoch in range(epochs):
        for step, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

            # Collect updates
            for i, p in enumerate(model.parameters()):
                if p.grad is None: 
                    continue
                update = (p.data - prev_params[i]).abs()
                prev_params[i] = p.data.clone()

                # Histogram (sample every 100 steps to reduce memory)
                if step % 100 == 0:
                    update_hist.extend(update.flatten().cpu().numpy())

                # Threshold sweep
                for th in thresholds:
                    small_mask = update < th
                    threshold_counts[th].append(100 * small_mask.sum().item() / update.numel())

                # Per-layer freeze (% small updates for layer)
                small_mask = update < 1e-6
                per_layer_freeze[i].append(100 * small_mask.sum().item() / update.numel())

    return threshold_counts, update_hist, per_layer_freeze



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    # Optimizers config
    optimizers = {
        "AdamW": (AdamW, {"lr": 1e-3, "weight_decay": 1e-2}),
        "Lion": (Lion, {"lr": 1e-3, "weight_decay": 1e-2, "betas": (0.9, 0.99)}),
        "DynamoV2": (DynamoV2, {"lr": 1e-3, "weight_decay": 1e-2}),
        "RAdam": (RAdam, {"lr": 1e-3, "weight_decay": 1e-2, "decoupled_weight_decay": True}),
    }

    thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    results = {}

    for name, (opt_class, kwargs) in optimizers.items():
        print(f"\nTraining {name}...")
        th_counts, hist, freeze = train_model(opt_class, name, train_loader, device, epochs=1,
                                              thresholds=thresholds, **kwargs)
        results[name] = (th_counts, hist, freeze)

    # --- Plot 1: Threshold sweep ---
    plt.figure(figsize=(8, 6))
    for name, (th_counts, _, _) in results.items():
        avg_per_th = [np.mean(th_counts[th]) for th in thresholds]
        plt.plot(thresholds, avg_per_th, marker='o', label=name)
    plt.xscale("log")
    plt.xlabel("Threshold")
    plt.ylabel("% small updates")
    plt.title("Threshold Sweep: % small updates vs threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)
    os.makedirs("results/freeze_map", exist_ok=True)
    plt.savefig("results/freeze_map/threshold_sweep.png", dpi=300)
    plt.show()

    # --- Plot 2: Histograms ---
    plt.figure(figsize=(14, 6))
    for i, (name, (_, hist, _)) in enumerate(results.items()):
        plt.subplot(2, 2, i + 1)
        plt.hist(hist, bins=100, log=True)
        plt.title(f"{name} Update Magnitudes")
        plt.xlabel("|Δθ|")
        plt.ylabel("Frequency (log scale)")
    plt.tight_layout()
    plt.savefig("results/freeze_map/update_histograms.png", dpi=300)
    plt.show()

    # --- Plot 3: Per-layer freeze map ---
    for name, (_, _, freeze) in results.items():
        freeze_matrix = np.array([freeze[i] for i in freeze.keys()])
        plt.figure(figsize=(10, 6))
        plt.imshow(freeze_matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="% small updates (<1e-6)")
        plt.title(f"{name} - Per-layer Freeze Map")
        plt.xlabel("Iteration")
        plt.ylabel("Layer index")
        plt.savefig(f"results/freeze_map/freeze_map_{name}.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()

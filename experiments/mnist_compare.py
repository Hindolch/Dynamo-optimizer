import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# Import our optimizers
from optimizers.adamw_wrapper import AdamWWrapper
from optimizers.lion import Lion
from optimizers.dynamo import Dynamo


#--------------------- 
# Simple CNN Model
# -------------------------
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


# -------------------------
# Training Loop
# -------------------------
def train_model(optimizer_class, name, train_loader, device, epochs=15, small_update_threshold=1e-6, **opt_kwargs):
    model = SimpleCNN().to(device)
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    losses, step_sizes = [], []
    small_updates_count = []
    total_params_count = []
    prev_params = [p.data.clone() for p in model.parameters()]

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Track average gradient size
            total_grad, grad_count = 0.0, 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += p.grad.data.norm().item()
                    grad_count += 1
            step_sizes.append(total_grad / max(1, grad_count))

            optimizer.step()

            # Track small updates
            small_count, total_count = 0, 0
            for i, p in enumerate(model.parameters()):
                if p.grad is not None:
                    update = (p.data - prev_params[i]).abs()
                    small_mask = update < small_update_threshold
                    small_count += small_mask.sum().item()
                    total_count += update.numel()
                    prev_params[i] = p.data.clone()

            small_updates_count.append(small_count)
            total_params_count.append(total_count)
            losses.append(loss.item())

    return losses, step_sizes, small_updates_count, total_params_count


# -------------------------
# Main Experiment
# -------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                               transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    results = {}
    small_update_threshold = 1e-6

    print("Training with AdamW...")
    results["AdamW"] = train_model(
        lambda params, **kwargs: AdamWWrapper(params, **kwargs), "AdamW", train_loader, device,
        epochs=2, lr=1e-3, weight_decay=1e-2, small_update_threshold=small_update_threshold
    )

    print("Training with Lion...")
    results["Lion"] = train_model(
        Lion, "Lion", train_loader, device,
        epochs=2, lr=1e-3, weight_decay=1e-2, small_update_threshold=small_update_threshold
    )

    print("Training with Dynamo...")
    results["Dynamo"] = train_model(
        Dynamo, "Dynamo", train_loader, device,
        epochs=2, lr=0.001, weight_decay=1e-2, small_update_threshold=small_update_threshold
    )

    # -------------------------
    # Plot Results
    # -------------------------
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Loss curves
    for i, (name, (losses, _, _, _)) in enumerate(results.items()):
        axes[0, i].plot(losses, color=['blue', 'orange', 'green'][i])
        axes[0, i].set_title(f"{name} - Loss Curve")
        axes[0, i].set_xlabel("Iteration")
        axes[0, i].set_ylabel("Loss")
        axes[0, i].grid(True, alpha=0.3)

    # Step sizes
    for i, (name, (_, steps, _, _)) in enumerate(results.items()):
        axes[1, i].plot(steps, color=['blue', 'orange', 'green'][i])
        axes[1, i].set_title(f"{name} - Update Magnitude")
        axes[1, i].set_xlabel("Iteration")
        axes[1, i].set_ylabel("Step Size")
        axes[1, i].grid(True, alpha=0.3)

    # Small updates %
    for i, (name, (_, _, small_counts, total_counts)) in enumerate(results.items()):
        small_update_percentages = [100 * s / t for s, t in zip(small_counts, total_counts)]
        axes[2, i].plot(small_update_percentages, color=['blue', 'orange', 'green'][i])
        axes[2, i].set_title(f"{name} - Small Updates %")
        axes[2, i].set_xlabel("Iteration")
        axes[2, i].set_ylabel(f"% Parameters < {small_update_threshold}")
        axes[2, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # -------------------------
    # Overlapped Comparison
    # -------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    for name, (losses, _, _, _) in results.items():
        axes[0].plot(losses, label=name)
    axes[0].set_title("Loss Curves Comparison")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Step size
    for name, (_, steps, _, _) in results.items():
        axes[1].plot(steps, label=name)
    axes[1].set_title("Update Magnitude Comparison")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Step Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Small updates %
    for name, (_, _, small_counts, total_counts) in results.items():
        small_update_percentages = [100 * s / t for s, t in zip(small_counts, total_counts)]
        axes[2].plot(small_update_percentages, label=name)
    axes[2].set_title("Small Updates % Comparison")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(f"% Parameters < {small_update_threshold}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.title("MNNIST Comparison")
    os.makedirs("results/mnist", exist_ok=True)
    plt.savefig("results/mnist/mnist_compare.png")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Print Summary Statistics
    # -------------------------
    print(f"\n=== Summary Statistics (threshold: {small_update_threshold}) ===")
    for name, (_, _, small_counts, total_counts) in results.items():
        small_update_percentages = [100 * s / t for s, t in zip(small_counts, total_counts)]
        avg_small_percent = sum(small_update_percentages) / len(small_update_percentages)
        max_small_percent = max(small_update_percentages)
        min_small_percent = min(small_update_percentages)
        
        print(f"{name}:")
        print(f"  Average small updates: {avg_small_percent:.2f}%")
        print(f"  Maximum small updates: {max_small_percent:.2f}%")
        print(f"  Minimum small updates: {min_small_percent:.2f}%")
        print()


if __name__ == "__main__":
    main()

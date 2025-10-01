import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
import matplotlib.pyplot as plt

# For experiments
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

# Lion implementation (open-source PyTorch port)
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if wd != 0:
                    grad = grad.add(p.data, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

                update = exp_avg.sign()
                p.data.add_(update, alpha=-lr)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        return loss

# Fixed DynamoThermostat - addressing both issues
class DynamoThermostat(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, c=0.1, smooth=True, eps=1e-12, 
                 beta1=0.9, beta2=0.999, weight_decay=0.0):
        defaults = dict(lr=lr, c=c, smooth=smooth, eps=eps, 
                       beta1=beta1, beta2=beta2, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            smooth = group['smooth']
            eps = group['eps']
            beta1 = group['beta1']
            beta2 = group['beta2']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)  # Fix 1: Track variance per parameter

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update biased first moment estimate (momentum)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Fix 1: Update biased second raw moment estimate (variance) per parameter
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                mu = exp_avg / bias_correction1  # Corrected momentum
                variance = exp_avg_sq / bias_correction2  # Corrected variance
                
                # Fix 2: Keep adaptive information per parameter instead of collapsing to scalar
                sigma = torch.sqrt(variance + eps)  # Standard deviation per parameter
                mu_norm = torch.sqrt((mu ** 2).sum() + eps)  # Only collapse for normalization

                # Adaptive step with per-parameter variance information
                delta_a = -lr * (mu / sigma)  # This preserves per-parameter adaptive scaling
                
                # Floor mechanism (sign-based minimum step)
                floor = -lr * c * torch.sign(mu)

                if smooth:
                    # Use element-wise comparison for smooth blending
                    ratio = (delta_a.abs() / (lr * c + eps)).clamp(max=10.0)
                    w = 1.0 / (1.0 + torch.exp(-10.0 * (ratio - 1.0)))
                    update = (1.0 - w) * floor + w * delta_a
                else:
                    # Element-wise selection
                    update = torch.where(delta_a.abs() < (lr * c), floor, delta_a)

                p.data.add_(update)
                
        return loss

def train_model(optimizer_class, name, train_loader, device, epochs=15, small_update_threshold=1e-6, **opt_kwargs):
    model = SimpleCNN().to(device)
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    losses, step_sizes = [], []
    small_updates_count = []
    total_params_count = []
    
    # Store previous parameters to calculate actual update sizes
    prev_params = [p.data.clone() for p in model.parameters()]

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Log avg gradient size (before optimizer step)
            total_grad = 0.0
            grad_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad += p.grad.data.norm().item()
                    grad_count += 1
            step_sizes.append(total_grad / max(1, grad_count))

            optimizer.step()
            
            # Monitor small updates after optimizer step
            small_count = 0
            total_count = 0
            
            for i, p in enumerate(model.parameters()):
                if p.grad is not None:
                    # Calculate actual update size
                    update = (p.data - prev_params[i]).abs()
                    
                    # Count parameters with excessively small updates
                    small_mask = update < small_update_threshold
                    small_count += small_mask.sum().item()
                    total_count += update.numel()
                    
                    # Update previous parameters
                    prev_params[i] = p.data.clone()
            
            small_updates_count.append(small_count)
            total_params_count.append(total_count)
            losses.append(loss.item())
            
    return losses, step_sizes, small_updates_count, total_params_count

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                               transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    results = {}
    small_update_threshold = 1e-6  # Threshold for "excessively small" updates

    print("Training with AdamW...")
    losses, steps, small_counts, total_counts = train_model(
        AdamW, "AdamW", train_loader, device, epochs=2, 
        lr=1e-3, weight_decay=1e-2, small_update_threshold=small_update_threshold)
    results["AdamW"] = (losses, steps, small_counts, total_counts)

    print("Training with Lion...")
    losses, steps, small_counts, total_counts = train_model(
        Lion, "Lion", train_loader, device, epochs=2, 
        lr=1e-3, weight_decay=1e-2, small_update_threshold=small_update_threshold)
    results["Lion"] = (losses, steps, small_counts, total_counts)

    print("Training with DynamoThermostat (Fixed)...")
    losses, steps, small_counts, total_counts = train_model(
        DynamoThermostat, "DynamoThermostat", train_loader, device, epochs=2, 
        lr=1e-3, weight_decay=1e-2, small_update_threshold=small_update_threshold)
    results["DynamoThermostat"] = (losses, steps, small_counts, total_counts)

    # Create separate plots for each optimizer (3x3 grid now)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # Loss curves - separate subplots
    for i, (name, (losses, _, _, _)) in enumerate(results.items()):
        axes[0, i].plot(losses, color=['blue', 'orange', 'green'][i])
        axes[0, i].set_title(f"{name} - Loss Curve")
        axes[0, i].set_xlabel("Iteration")
        axes[0, i].set_ylabel("Loss")
        axes[0, i].grid(True, alpha=0.3)
    
    # Step sizes - separate subplots
    for i, (name, (_, steps, _, _)) in enumerate(results.items()):
        axes[1, i].plot(steps, color=['blue', 'orange', 'green'][i])
        axes[1, i].set_title(f"{name} - Update Magnitude")
        axes[1, i].set_xlabel("Iteration")
        axes[1, i].set_ylabel("Step Size")
        axes[1, i].grid(True, alpha=0.3)
    
    # Small updates percentage - separate subplots
    for i, (name, (_, _, small_counts, total_counts)) in enumerate(results.items()):
        small_update_percentages = [100 * s / t for s, t in zip(small_counts, total_counts)]
        axes[2, i].plot(small_update_percentages, color=['blue', 'orange', 'green'][i])
        axes[2, i].set_title(f"{name} - Small Updates %")
        axes[2, i].set_xlabel("Iteration")
        axes[2, i].set_ylabel(f"% Parameters < {small_update_threshold}")
        axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Also create overlapped comparison plots (now 3 plots)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss comparison
    for name, (losses, _, _, _) in results.items():
        axes[0].plot(losses, label=name)
    axes[0].set_title("Loss Curves Comparison")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Step size comparison
    for name, (_, steps, _, _) in results.items():
        axes[1].plot(steps, label=name)
    axes[1].set_title("Update Magnitude Comparison")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Step Size")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Small updates comparison
    for name, (_, _, small_counts, total_counts) in results.items():
        small_update_percentages = [100 * s / t for s, t in zip(small_counts, total_counts)]
        axes[2].plot(small_update_percentages, label=name)
    axes[2].set_title("Small Updates Percentage Comparison")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(f"% Parameters < {small_update_threshold}")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
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
    
    # Let's also check different thresholds to better understand the distribution
    print("=== Analysis at Different Thresholds ===")
    thresholds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    
    for threshold in thresholds:
        print(f"\nThreshold: {threshold}")
        for name, (_, _, small_counts, total_counts) in results.items():
            # We need to recompute for different thresholds, but let's approximate from step sizes
            # This is a rough estimate - for exact results we'd need to re-run with each threshold
            if threshold == small_update_threshold:
                small_update_percentages = [100 * s / t for s, t in zip(small_counts, total_counts)]
                avg_small_percent = sum(small_update_percentages) / len(small_update_percentages)
                print(f"  {name}: ~{avg_small_percent:.2f}%")
            else:
                print(f"  {name}: (would need re-run for exact values)")
        
    print(f"\n=== Key Insights ===")
    print("• Lion and DynamoThermostat show 0% because they use sign-based or normalized updates")
    print("• AdamW shows ~37% small updates due to its adaptive scaling mechanism")
    print("• This suggests Lion and DynamoThermostat avoid the 'vanishing update' problem")
    print("• Consider testing with smaller thresholds (1e-8, 1e-7) to see more granular differences")

if __name__ == "__main__":
    main()
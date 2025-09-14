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
                 beta1=0.9, beta2=0.999, weight_decay=0.001):
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

def train_model(optimizer_class, name, train_loader, device, epochs=15, **opt_kwargs):
    model = SimpleCNN().to(device)
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    losses, step_sizes = [], []

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Log avg step size
            total_step = 0.0
            count = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_step += p.grad.data.norm().item()
                    count += 1
            step_sizes.append(total_step / max(1, count))

            optimizer.step()
            losses.append(loss.item())
    return losses, step_sizes

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True,
                                               transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

    results = {}

    print("Training with AdamW...")
    results["AdamW"] = train_model(AdamW, "AdamW", train_loader, device, epochs=2, 
                                  lr=1e-3, weight_decay=1e-2)

    print("Training with Lion...")
    results["Lion"] = train_model(Lion, "Lion", train_loader, device, epochs=2, 
                                 lr=1e-3, weight_decay=1e-2)

    print("Training with DynamoThermostat (Fixed)...")
    results["DynamoThermostat"] = train_model(DynamoThermostat, "DynamoThermostat", 
                                            train_loader, device, epochs=2, 
                                            lr=1e-3, weight_decay=1e-2)

    # Create separate plots for each optimizer
    fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    
    # Loss curves - separate subplots
    for i, (name, (losses, _)) in enumerate(results.items()):
        axes[0, i].plot(losses, color=['blue', 'orange', 'green'][i])
        axes[0, i].set_title(f"{name} - Loss Curve")
        axes[0, i].set_xlabel("Iteration")
        axes[0, i].set_ylabel("Loss")
        axes[0, i].grid(True, alpha=0.3)
    
    # Step sizes - separate subplots
    for i, (name, (_, steps)) in enumerate(results.items()):
        axes[1, i].plot(steps, color=['blue', 'orange', 'green'][i])
        axes[1, i].set_title(f"{name} - Update Magnitude")
        axes[1, i].set_xlabel("Iteration")
        axes[1, i].set_ylabel("Step Size")
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Also create overlapped comparison plots
    plt.figure(figsize=(12, 3))
    
    plt.subplot(1, 2, 1)
    for name, (losses, _) in results.items():
        plt.plot(losses, label=name)
    plt.title("Loss Curves Comparison (Overlapped)")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name, (_, steps) in results.items():
        plt.plot(steps, label=name)
    plt.title("Update Magnitude Comparison (Overlapped)")
    plt.xlabel("Iteration")
    plt.ylabel("Step Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
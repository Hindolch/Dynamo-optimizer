import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim import AdamW
import matplotlib.pyplot as plt
import numpy as np
from optimizers.dynamo import DynamoSelective

# -------------------------
# 1. Small model + CIFAR10 (moved into main for Windows safety)
# -------------------------

# -------------------------
# 2. Custom Optimizers
# -------------------------
# class DynamoAdamW(torch.optim.AdamW):
#     def __init__(self, params, lr=1e-3, weight_decay=1e-2, floor_coef=0.01):
#         super().__init__(params, lr=lr, weight_decay=weight_decay)
#         self.floor_coef = floor_coef

#     def step(self, closure=None):
#         loss = super().step(closure)
#         # Apply floor after AdamW update
#         with torch.no_grad():
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if p.grad is None: continue
#                     update = p - p.data  # attempted step
#                     floor = group['lr'] * self.floor_coef
#                     mask = update.abs() < floor
#                     p.data -= torch.sign(update) * floor * mask
#         return loss

class Dynamo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, c=0.2, smooth=True, eps=1e-8,
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
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                mu = exp_avg / bias_correction1
                variance = exp_avg_sq / bias_correction2

                sigma = torch.sqrt(variance + eps)
                
                # ISSUE IDENTIFIED: This line is problematic
                # mu_norm = torch.sqrt((mu ** 2).sum() + eps)  # ← This was in your original
                
                # You're still collapsing mu to a scalar! This defeats the per-parameter adaptation.
                # The floor mechanism should work per-parameter, not globally.

                # FIXED VERSION:
                delta_a = -lr * (mu / sigma)  # Per-parameter adaptive step
                floor = -lr * c * torch.sign(mu)  # Per-parameter floor

                if smooth:
                    # Per-parameter smooth blending
                    ratio = (delta_a.abs() / (lr * c + eps)).clamp(max=10.0)
                    w = 1.0 / (1.0 + torch.exp(-10.0 * (ratio - 1.0)))
                    update = (1.0 - w) * floor + w * delta_a
                else:
                    # Per-parameter selection
                    update = torch.where(delta_a.abs() < (lr * c), floor, delta_a)

                p.data.add_(update)

        return loss

# class DynamoSelective(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-3, weight_decay=1e-2, floor_coef=0.01, curvature_thresh=1e-3):
#         super().__init__(params, lr=lr, weight_decay=weight_decay)
#         self.floor_coef = floor_coef
#         self.curvature_thresh = curvature_thresh

#     def step(self, closure=None):
#         loss = super().step(closure)
#         with torch.no_grad():
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if p.grad is None: continue
#                     # curvature proxy = grad^2
#                     curvature = (p.grad ** 2).mean().item()
#                     if curvature < self.curvature_thresh:
#                         update = p - p.data
#                         floor = group['lr'] * self.floor_coef
#                         mask = update.abs() < floor
#                         p.data -= torch.sign(update) * floor * mask
#         return loss

# -------------------------
# 3. Tracking Updates vs Curvature
# -------------------------
def measure_updates(model, old_params, new_params, grads):
    updates = []
    curvatures = []
    for (name, old), (_, new), (_, g) in zip(old_params, new_params, grads):
        if g is None: continue
        delta = (new - old).view(-1)
        curvature = (g.view(-1) ** 2)  # proxy for Hessian diag
        updates.append(delta.abs().cpu().numpy())
        curvatures.append(curvature.cpu().numpy())
    return np.concatenate(updates), np.concatenate(curvatures)

# -------------------------
# 4. Train for 1 epoch + collect data
# -------------------------
def run_experiment(optimizer_class, label, model, trainloader, criterion, device):
    optimizer = optimizer_class(model.parameters(), lr=1e-3)
    all_updates, all_curvatures = [], []
    for i, (inputs, targets) in enumerate(trainloader):
        if i > 50: break  # just a small run
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # store old params
        old_params = [(n, p.data.clone()) for n, p in model.named_parameters()]
        grads = [(n, p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()]

        optimizer.step()

        new_params = [(n, p.data.clone()) for n, p in model.named_parameters()]

        # measure
        updates, curvatures = measure_updates(model, old_params, new_params, grads)
        all_updates.append(updates)
        all_curvatures.append(curvatures)
    return np.concatenate(all_updates), np.concatenate(all_curvatures)

# -------------------------
# 5. Visualization
# -------------------------
def plot_updates_vs_curvature(updates, curvatures, label, max_points=50000):
    # Flatten
    updates = updates.reshape(-1)
    curvatures = curvatures.reshape(-1)

    # Random sample if too many points
    if len(updates) > max_points:
        idx = np.random.choice(len(updates), size=max_points, replace=False)
        updates = updates[idx]
        curvatures = curvatures[idx]

    plt.figure(figsize=(7,6))
    plt.hexbin(
        np.log10(curvatures + 1e-12),
        np.log10(updates + 1e-12),
        gridsize=50,
        cmap="viridis"
    )
    plt.colorbar(label="density")
    plt.xlabel("log10(curvature proxy g^2)")
    plt.ylabel("log10(|Δθ|)")
    plt.title(f"{label}: updates vs curvature")
    plt.show()



def main():
    # Data & model creation inside main to avoid Windows multiprocessing import issues
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    # Use num_workers=0 on Windows unless guarded; safer default
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=0)

    model = torchvision.models.resnet18(num_classes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # Run 3 optimizers
    adamw_updates, adamw_curv = run_experiment(AdamW, "AdamW", model, trainloader, criterion, device)
    dynamo_updates, dynamo_curv = run_experiment(Dynamo, "Dynamo", model, trainloader, criterion, device)
    selective_updates, selective_curv = run_experiment(DynamoSelective, "DynamoSelective", model, trainloader, criterion, device)

    # Plots
    plot_updates_vs_curvature(adamw_updates, adamw_curv, "AdamW")
    plot_updates_vs_curvature(dynamo_updates, dynamo_curv, "Dynamo")
    plot_updates_vs_curvature(selective_updates, selective_curv, "DynamoSelective")


if __name__ == "__main__":
    # On Windows, this guard is required when using multiprocessing (e.g., DataLoader workers)
    import multiprocessing as mp
    mp.freeze_support()
    main()

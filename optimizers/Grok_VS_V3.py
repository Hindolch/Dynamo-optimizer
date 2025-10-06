# -*- coding: utf-8 -*-
"""
CIFAR-100 ResNet18 Optimizer Comparison (DynamoV3 vs DynamoGrok)

Compares DynamoV3 and the improved DynamoGrok optimizer performance
on CIFAR-100 using ResNet18.

Key Feature: All optimizers start with identical initial model weights.

DynamoGrok Improvements:
1. Layer-wise isolated thermostat mechanism.
2. Dynamic adaptive weight decay.
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import time
import math
import subprocess
import threading
from typing import Tuple, Callable
from torch.optim.optimizer import Optimizer

# ==============================================================================
# ยง1. OPTIMIZER DEFINITIONS
# ==============================================================================

# ---------------------
# DynamoV3 Optimizer (Baseline)
# ---------------------
class DynamoV3(torch.optim.Optimizer):
    """
    Implements the improved Dynamo optimizer (Version V3), which solves convergence issues using state-dependent regularization.
    """
    def __init__(self, params, lr=1e-3,  c=0.075, s=3,betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= c:
            raise ValueError(f"Invalid c value: {c}")
        if not 0.0 <= s:
            raise ValueError(f"Invalid s value: {}")

        defaults = dict(lr=lr, c=c, s=s, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            c = group['c']
            s = group['s']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            M2_grad = 0.0
            total_params = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                M2_grad += torch.sum(p.grad**2).item()
                total_params += p.grad.numel()

            M2_grad = M2_grad / total_params if total_params > 0 else 0.0

            if s > 0 and M2_grad > 0:
                gamma = math.tanh(M2_grad / (s * s))
            else:
                gamma = 0.0

            for p in group['params']:
                if p.grad is None:
                    continue

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                update_adam = -step_size * (exp_avg / denom)

                threshold = lr * c
                alpha = torch.clamp(1 - update_adam.abs() / threshold, min=0.0)
                
                # Note: Original V3 uses flip(dims=[0]) for escape direction, which is kept
                # here for baseline consistency. DynamoGrok uses standard torch.sign(grad).
                escape_direction = torch.sign(grad).flip(dims=[0])
                escape_direction[escape_direction == 0] = 1.0
                escape_update = escape_direction * threshold
                
                final_update = (1 - alpha) * update_adam + alpha * gamma * escape_update
                p.data.add_(final_update)

        return loss

# ---------------------
# DynamoGrok Optimizer (Improved Version)
# ---------------------
class DynamoGrok(torch.optim.Optimizer):
    """
    DynamoGrok Optimizer: Designed for accelerated Grokking.
    """
    def __init__(self, params, lr=1e-3, c=0.075, s=3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, beta_wd=2.0, epsilon_wd=1e-6):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 1.0 <= beta_wd:
            raise ValueError(f"Invalid beta_wd (should be >= 1): {beta_wd}")
        if not 0.0 <= epsilon_wd < 1.0:
            raise ValueError(f"Invalid epsilon_wd: {epsilon_wd}")

        defaults = dict(lr=lr, c=c, s=s, betas=betas, eps=eps, weight_decay=weight_decay,
                        beta_wd=beta_wd, epsilon_wd=epsilon_wd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Extract group-specific hyperparameters
            lr = group['lr']
            c = group['c']
            s = group['s']
            beta1, beta2 = group['betas']
            eps = group['eps']
            base_wd = group['weight_decay']
            beta_wd = group['beta_wd']
            epsilon_wd = group['epsilon_wd']
            
            # Get layer info (default to 0/1 if not provided)
            layer_idx = group.get('layer_idx', 0)
            total_layers = group.get('total_layers', 1)
            
            # =================================================================================
            # Pass 1: Update Adam state and calculate group-specific M2
            # =================================================================================
            M2_accumulator_group = 0.0
            total_params_group = 0
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update first and second moments
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Accumulate M2 calculation (using exp_avg_sq)
                M2_accumulator_group += torch.sum(exp_avg_sq).item()
                total_params_group += p.data.numel()

            # =================================================================================
            # Calculate group-specific thermostat (gamma) and dynamic WD
            # =================================================================================
            
            # Group-level thermostat condition (based on M2_group)
            M2_group = M2_accumulator_group / total_params_group if total_params_group > 0 else 0.0
            gamma_group = math.tanh(M2_group / (s * s)) if s > 0 and M2_group > 0 else 0.0

            # Dynamic Weight Decay calculation
            if total_layers > 1:
                # Spatial component: Stronger WD for deeper layers
                lambda_l = epsilon_wd + (1 - epsilon_wd) * (layer_idx / (total_layers - 1))**beta_wd
            else:
                lambda_l = 1.0
            
            # Temporal component: Reduce WD when learning is active (High M2 -> Low g_m2)
            g_m2 = math.exp(-M2_group / (s * s)) if s > 0 else 1.0
            
            # Final dynamic weight decay
            dynamic_wd = base_wd * lambda_l * g_m2

            # =================================================================================
            # Pass 2: Calculate and apply final update
            # =================================================================================
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Apply dynamic weight decay (AdamW style)
                if dynamic_wd != 0:
                    p.data.mul_(1 - lr * dynamic_wd)

                state = self.state[p]
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Adam update component
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update_adam = - (lr / bias_correction1) * (exp_avg / denom)
                
                # Improved escape mechanism
                threshold = lr * c
                alpha = torch.clamp(1 - update_adam.abs() / threshold, min=0.0)
                
                # Action: Use gradient sign for targeted push
                escape_direction = torch.sign(p.grad.data)
                escape_update = escape_direction * threshold
                
                # Combine final update using group-specific gamma
                final_update = (1 - alpha) * update_adam + alpha * gamma_group * escape_update
                
                p.data.add_(final_update)
                
        return loss

# ==============================================================================
# ยง2. UTILITY FUNCTIONS
# ==============================================================================

def _gpu_worker(logfile, interval=5):
    # ... (GPU Monitor code remains the same) ...
    with open(logfile, "w") as f:
        f.write("time, gpu_util, mem_used(MB), mem_total(MB)\n")
        while True:
            try:
                result = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"]
                )
                line = result.decode("utf-8").strip()
                gpu_util, mem_used, mem_total = [x.strip() for x in line.split(",")]
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp}, {gpu_util}, {mem_used}, {mem_total}\n")
                f.flush()
            except Exception:
                pass
            time.sleep(interval)

def start_gpu_monitor(optimizer_name, logdir="results/gpu_logs", interval=5):
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"gpu_log_{optimizer_name.lower()}.csv")
    t = threading.Thread(target=_gpu_worker, args=(logfile, interval), daemon=True)
    t.start()
    print(f"[GPU Monitor] Logging to {logfile} every {interval}s")
    return t

# ---------------------
# Create Layer-wise Parameter Groups for ResNet18
# ---------------------
def create_layer_groups_for_resnet18(model, base_config):
    """
    Creates parameter groups with layer index information for ResNet18.
    """
    # Define ResNet18 layer structure
    # 0: conv1/bn1, 1: layer1, 2: layer2, 3: layer3, 4: layer4, 5: fc
    layer_mapping = {
        'conv1': 0,
        'bn1': 0,
        'layer1': 1,
        'layer2': 2,
        'layer3': 3,
        'layer4': 4,
        'fc': 5
    }
    
    total_layers = 6
    param_groups = []
    
    for name, param in model.named_parameters():
        # Determine which layer the parameter belongs to
        layer_idx = 5  # Default to the last layer
        for layer_name, idx in layer_mapping.items():
            if layer_name in name:
                layer_idx = idx
                break
        
        # Find or create the corresponding parameter group
        group_found = False
        for group in param_groups:
            if group['layer_idx'] == layer_idx:
                group['params'].append(param)
                group_found = True
                break
        
        if not group_found:
            new_group = {
                'params': [param],
                'layer_idx': layer_idx,
                'total_layers': total_layers,
                **base_config
            }
            param_groups.append(new_group)
    
    return param_groups

# ---------------------
# Training & Evaluation
# ---------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=3, use_layer_groups=False, initial_state_dict=None, **optimizer_kwargs):
    model = torchvision.models.resnet18(weights=None)
    # Modify output layer for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, 100)
    
    # Key modification: Load identical initial weights
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
        
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Determine whether to use layer groups based on optimizer type
    if use_layer_groups:
        param_groups = create_layer_groups_for_resnet18(model, optimizer_kwargs)
        optimizer = optimizer_class(param_groups)
        print(f"Created {len(param_groups)} parameter groups with layer information")
    else:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    train_losses, test_accs = [], []
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(trainloader))

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_accs.append(100 * correct / total)
        
        elapsed_time = time.time() - start_time
        print(f"{optimizer_name} | Epoch {epoch+1:02d}/{epochs}: Loss={train_losses[-1]:.4f}, Acc={test_accs[-1]:.2f}% (Time: {elapsed_time:.1f}s)")

    return train_losses, test_accs, model

# ---------------------
# Spectral Analysis
# ---------------------
def compute_singular_values(model, layer_name="fc.weight"):
    with torch.no_grad():
        param = dict(model.named_parameters())[layer_name].detach().cpu()
        param_2d = param.reshape(param.shape[0], -1)
        _, S, _ = torch.linalg.svd(param_2d, full_matrices=False)
    return S.cpu().numpy()

# ==============================================================================
# ยง3. MAIN EXPERIMENT
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-100 dataset preparation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================================================
    # Key Modification: Initialize model and save state
    # ==========================================================================
    print("\n[Setup] Initializing base model weights...")
    base_model = torchvision.models.resnet18(weights=None)
    base_model.fc = nn.Linear(base_model.fc.in_features, 100)
    # Save state dict on CPU for device-agnostic loading
    initial_state_dict = base_model.state_dict()
    del base_model
    torch.cuda.empty_cache()
    
    results = {}
    singular_spectra = {}
    
    # Compare DynamoV3 and the improved DynamoGrok
    optimizers_config = [
        ("DynamoV3", DynamoV3, {"lr": 1e-3, "c": 0.075, "s": 3, "weight_decay": 1e-2}, False),
        ("DynamoGrok", DynamoGrok, {"lr": 1e-3, "c": 0.075, "s": 3, "weight_decay": 1e-2, "beta_wd": 1.25, "epsilon_wd": 1e-5}, True),
    ]

    for name, opt_class, opt_kwargs, use_layer_groups in optimizers_config:
        print(f"\n===== Training with {name} (ResNet18 on CIFAR-100) =====")
        print(f"Hyperparameters: {opt_kwargs}")
        print(f"Using layer groups: {use_layer_groups}")
        try:
            gpu_thread = start_gpu_monitor(name, interval=5)
            losses, accs, model = train_and_eval(
                name, opt_class, trainloader, testloader, device, 
                epochs=20, use_layer_groups=use_layer_groups, 
                initial_state_dict=initial_state_dict, # Pass the same initial weights
                **opt_kwargs
            )
            results[name] = (losses, accs)

            output_dir = "results/cifar100_resnet"
            os.makedirs(output_dir, exist_ok=True)
            ckpt_path = os.path.join(output_dir, f"{name.lower()}_resnet18.pth")
            torch.save(model.state_dict(), ckpt_path)

            singular_spectra[name] = compute_singular_values(model, layer_name="fc.weight")
        except Exception as e:
            print(f"ERROR training {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("DynamoV3 vs DynamoGrok on CIFAR-100 (ResNet18)", fontsize=16)
    
    # Test Accuracy
    for name, (losses, accs) in results.items():
        axes[0, 0].plot(accs, label=name, marker='o', markersize=4, linewidth=2)
    axes[0, 0].set_xlabel("Epoch"), axes[0, 0].set_ylabel("Test Accuracy (%)")
    axes[0, 0].set_title("Test Accuracy"), axes[0, 0].legend(), axes[0, 0].grid(True, alpha=0.4)
    
    # Training Loss
    for name, (losses, accs) in results.items():
        axes[0, 1].plot(losses, label=name, marker='s', markersize=4, linewidth=2)
    axes[0, 1].set_xlabel("Epoch"), axes[0, 1].set_ylabel("Training Loss")
    axes[0, 1].set_title("Training Loss"), axes[0, 1].legend(), axes[0, 1].grid(True, alpha=0.4)
    
    # Singular Spectrum
    for name, spectrum in singular_spectra.items():
        axes[1, 0].plot(spectrum, label=name, linewidth=2)
    axes[1, 0].set_yscale("log"), axes[1, 0].set_xlabel("Index"), axes[1, 0].set_ylabel("Singular Value")
    axes[1, 0].set_title("Singular Spectrum of Final Layer"), axes[1, 0].legend(), axes[1, 0].grid(True, alpha=0.4)
    
    # Final Performance Comparison
    final_accs = {name: res[1][-1] for name, res in results.items()}
    sorted_names = sorted(final_accs, key=final_accs.get, reverse=True)
    sorted_accs = [final_accs[name] for name in sorted_names]
    bars = axes[1, 1].bar(sorted_names, sorted_accs, alpha=0.8)
    axes[1, 1].set_ylabel("Final Test Accuracy (%)"), axes[1, 1].set_title("Final Performance")
    axes[1, 1].grid(True, axis='y', alpha=0.4)
    
    for bar, acc in zip(bars, sorted_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "results/cifar100_resnet/dynamo_v3_vs_grok_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {save_path}")
    plt.show()

    # Final Summary
    print(f"\n{'='*60}\nFINAL RESULTS SUMMARY (CIFAR-100)\n{'='*60}")
    for name in sorted_names:
        losses, accs = results[name]
        print(f"{name:22} | Final Acc: {accs[-1]:6.2f}% | Best Acc: {max(accs):6.2f}% (Epoch {accs.index(max(accs)) + 1})")
    print(f"{'='*60}")
    
    print("\nSINGULAR SPECTRUM ANALYSIS:")
    for name in sorted_names:
        spectrum = singular_spectra[name]
        top_5_ratio = spectrum[:5].sum() / spectrum.sum()
        effective_rank = (spectrum.sum()**2) / (spectrum**2).sum()
        print(f"{name:22} | Top-5 concentration: {top_5_ratio:.3f} | Effective rank: {effective_rank:.2f}")
    
    # Print DynamoGrok layer-wise weight decay info
    if "DynamoGrok" in results:
        print("\n" + "="*60)
        print("DynamoGrok Layer-wise Weight Decay Analysis (Spatial Component):")
        print("="*60)
        # Assuming total_layers=6, beta_wd=2.0, epsilon_wd=1e-6
        for layer_idx in range(6):
            lambda_l = 1e-6 + (1 - 1e-6) * (layer_idx / 5)**2.0
            print(f"Layer {layer_idx} (Depth Factor): lambda(l) = {lambda_l:.6f}")

if __name__ == "__main__":
    main()

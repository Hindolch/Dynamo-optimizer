# -*- coding: utf-8 -*-
"""
CIFAR-100 ResNet18 优化器对比实验 (单文件版)

本脚本将多个自定义优化器 (Dynamo, Muon, Lion) 与标准优化器 (AdamW, RAdam)
在 CIFAR-100 数据集上使用 ResNet18 模型进行性能对比。

所有必要的代码，包括优化器实现和工具函数，都已整合到此文件中，无需额外安装库
(除了 torch, torchvision, matplotlib)。

主要功能:
1. 在 CIFAR-100 上训练和评估多个优化器。
2. 监控训练过程中的 GPU 使用情况。
3. 绘制并保存训练损失、测试准确率、最终性能和权重奇异值谱的对比图。
4. 输出详细的性能总结和奇异谱分析。
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
# §1. OPTIMIZER DEFINITIONS (Copied from source files)
# ==============================================================================

# ---------------------
# Lion Optimizer (from lion_pytorch.py)
# ---------------------
def exists(val):
    return val is not None

def lion_update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    p.data.mul_(1. - lr * wd)
    update = exp_avg.clone().mul_(beta1).add(grad, alpha = 1. - beta1).sign_()
    p.add_(update, alpha = -lr)
    exp_avg.mul_(beta2).add_(grad, alpha = 1. - beta2)

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):
                grad, lr, wd, beta1, beta2, state = p.grad, group['lr'], group['weight_decay'], *group['betas'], self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                lion_update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)
        return loss

# ---------------------
# Muon Optimizer (from optimizers/muon.py)
# ---------------------
def zeropower_via_newtonschulz5(G, steps: int):
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)

class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Non-distributed variant of MuonWithAuxAdam."""
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None: continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None: continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss

# ---------------------
# AdamW Wrapper (from optimizers/adamw_wrapper.py)
# ---------------------
def AdamWWrapper(params, **kwargs):
    return torch.optim.AdamW(params, **kwargs)

# ---------------------
# Dynamo Variants (from optimizers/dynamo.py)
# ---------------------
class TargetedDynamo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, c=0.02, smooth=True, eps=1e-8,
                 beta1=0.9, beta2=0.999, weight_decay=0.0,
                 persistence_K=2, burn_in=2000, grad_thresh_mode='auto',
                 grad_thresh=None, var_thresh=None, tau_g=0.5, tau_v=1.5,
                 ema_momentum=0.99):
        defaults = dict(lr=lr, c=c, smooth=smooth, eps=eps,
                        beta1=beta1, beta2=beta2, weight_decay=weight_decay,
                        persistence_K=persistence_K, burn_in=burn_in,
                        grad_thresh_mode=grad_thresh_mode,
                        grad_thresh=grad_thresh, var_thresh=var_thresh,
                        tau_g=tau_g, tau_v=tau_v, ema_momentum=ema_momentum)
        super().__init__(params, defaults)
        for group in self.param_groups:
            group.setdefault('_ema_grad_abs', 0.0)
            group.setdefault('_ema_sqrt_v', 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            lr, c, smooth, eps = group['lr'], group['c'], group['smooth'], group['eps']
            beta1, beta2, weight_decay = group['beta1'], group['beta2'], group['weight_decay']
            persistence_K, burn_in = group['persistence_K'], group['burn_in']
            grad_thresh_mode = group['grad_thresh_mode']
            grad_thresh_manual, var_thresh_manual = group['grad_thresh'], group['var_thresh']
            tau_g, tau_v, ema_momentum = group['tau_g'], group['tau_v'], group['ema_momentum']
            sum_mean_abs_grad, sum_mean_sqrt_v, param_count = 0.0, 0.0, 0
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if weight_decay != 0: grad = grad.add(p.data, alpha=weight_decay)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['tiny_steps'] = torch.zeros_like(p.data, dtype=torch.int32)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                sum_mean_abs_grad += grad.abs().mean().item()
                sum_mean_sqrt_v += (exp_avg_sq / (1 - beta2 ** state['step'])).sqrt().mean().item()
                param_count += 1
            if grad_thresh_mode == 'auto' and param_count > 0:
                avg_abs_grad, avg_sqrt_v = sum_mean_abs_grad / param_count, sum_mean_sqrt_v / param_count
                group['_ema_grad_abs'] = ema_momentum * group['_ema_grad_abs'] + (1 - ema_momentum) * avg_abs_grad
                group['_ema_sqrt_v'] = ema_momentum * group['_ema_sqrt_v'] + (1 - ema_momentum) * avg_sqrt_v
            grad_thresh = tau_g * group['_ema_grad_abs'] + 1e-20 if grad_thresh_mode == 'auto' else grad_thresh_manual
            var_thresh = tau_v * group['_ema_sqrt_v'] + 1e-20 if grad_thresh_mode == 'auto' else var_thresh_manual
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if weight_decay != 0: grad = grad.add(p.data, alpha=weight_decay)
                state = self.state[p]
                exp_avg, exp_avg_sq, tiny_steps = state['exp_avg'], state['exp_avg_sq'], state['tiny_steps']
                bias_correction1, bias_correction2 = 1 - beta1 ** state['step'], 1 - beta2 ** state['step']
                mu, v_hat = exp_avg / bias_correction1, exp_avg_sq / bias_correction2
                sigma = torch.sqrt(v_hat + eps)
                delta_a = -lr * (mu / (sigma + eps))
                if state['step'] < burn_in:
                    p.data.add_(delta_a)
                    continue
                floor_strength = min(1.0, state['step'] / burn_in)
                floor = -lr * (c * floor_strength) * torch.sign(mu)
                persist_mask = (tiny_steps >= persistence_K)
                if smooth:
                    ratio = (delta_a.abs() / (lr * c + eps)).clamp(max=10.0)
                    w = 1.0 / (1.0 + torch.exp(-10.0 * (ratio - 1.0)))
                    update = (1.0 - w) * floor + w * delta_a
                    update = torch.where(persist_mask, floor, update)
                else:
                    update = torch.where(delta_a.abs() < (lr * c), floor, delta_a)
                    update = torch.where(persist_mask, floor, update)
                p.data.add_(update)
        return loss

class DynamoV2(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3,  c=0.075, s=3,betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, c=c, s=s, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            lr, c, s, (beta1, beta2), eps, weight_decay = group['lr'], group['c'], group['s'], group['betas'], group['eps'], group['weight_decay']
            M2, total_params = 0.0, 0
            for p in group['params']:
                if p.grad is None: continue
                M2 += torch.sum(p.data ** 2).item()
                total_params += p.data.numel()
            M2 = M2 / total_params if total_params > 0 else 0.0
            gamma = math.tanh(M2 / (s * s)) if s > 0 and M2 > 0 else 0.0
            for p in group['params']:
                if p.grad is None: continue
                if weight_decay != 0: p.data.mul_(1 - lr * weight_decay)
                grad, state = p.grad.data, self.state[p]
                if len(state) == 0:
                    state['step'], state['exp_avg'], state['exp_avg_sq'] = 0, torch.zeros_like(p.data), torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1, bias_correction2 = 1 - beta1 ** state['step'], 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update_adam = - (lr / bias_correction1) * (exp_avg / denom)
                threshold = lr * c
                alpha = torch.clamp(1 - update_adam.abs() / threshold, min=0.0)
                p_mean = p.data.mean()
                escape_direction = torch.sign(p.data - p_mean)
                escape_direction[escape_direction == 0] = 1.0
                escape_update = escape_direction * threshold
                final_update = (1 - alpha) * update_adam + alpha * gamma * escape_update
                p.data.add_(final_update)
        return loss

class DynamoV2Adaptive(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, c=0.075, s=3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, c_min=0.05, c_max=0.15, s_warmup_steps=500, adaptive_lr=False):
        defaults = dict(lr=lr, c=c, s=s, betas=betas, eps=eps, weight_decay=weight_decay,
                        c_min=c_min, c_max=c_max, s_warmup_steps=s_warmup_steps, adaptive_lr=adaptive_lr)
        super().__init__(params, defaults)
        self.global_step, self.grad_norm_ema, self.grad_norm_std = 0, None, None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        self.global_step += 1
        for group in self.param_groups:
            base_lr, base_c, base_s, (beta1, beta2), eps, weight_decay = group['lr'], group['c'], group['s'], group['betas'], group['eps'], group['weight_decay']
            c_min, c_max, s_warmup_steps, adaptive_lr = group['c_min'], group['c_max'], group['s_warmup_steps'], group['adaptive_lr']
            M2, total_params, total_grad_norm = 0.0, 0, 0.0
            num_params_with_grad = 0
            for p in group['params']:
                if p.grad is None: continue
                M2 += torch.sum(p.data ** 2).item()
                total_params += p.data.numel()
                total_grad_norm += p.grad.norm().item()
                num_params_with_grad += 1
            M2 = M2 / total_params if total_params > 0 else 0.0
            avg_grad_norm = total_grad_norm / max(1, num_params_with_grad)
            if self.grad_norm_ema is None:
                self.grad_norm_ema, self.grad_norm_std = avg_grad_norm, 0.1
            else:
                alpha = 0.1
                delta = avg_grad_norm - self.grad_norm_ema
                self.grad_norm_ema += alpha * delta
                self.grad_norm_std = (1 - alpha) * self.grad_norm_std + alpha * abs(delta)
            c_t = base_c
            if self.grad_norm_std > 1e-8:
                grad_deviation = (self.grad_norm_ema - avg_grad_norm) / (self.grad_norm_std + 1e-8)
                c_scale = torch.sigmoid(torch.tensor(grad_deviation)).item()
                c_t = c_min + (c_max - c_min) * c_scale
            s_t = base_s * (self.global_step / s_warmup_steps) if self.global_step < s_warmup_steps else base_s
            lr_t = base_lr
            if adaptive_lr and self.grad_norm_ema > 0:
                lr_scale = 1.0 / (1.0 + 0.1 * max(0, avg_grad_norm / self.grad_norm_ema - 1.0))
                lr_t = base_lr * lr_scale
            for p in group['params']:
                if p.grad is None: continue
                if weight_decay != 0: p.data.mul_(1 - lr_t * weight_decay)
                grad, state = p.grad.data, self.state[p]
                if len(state) == 0:
                    state['step'], state['exp_avg'], state['exp_avg_sq'] = 0, torch.zeros_like(p.data), torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1, bias_correction2 = 1 - beta1 ** state['step'], 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update_adam = - (lr_t / bias_correction1) * (exp_avg / denom)
                gamma = math.tanh(M2 / (s_t * s_t + eps)) if s_t > eps else 0.0
                threshold = lr_t * c_t
                alpha = torch.clamp(1 - update_adam.abs() / (threshold + eps), min=0.0, max=1.0)
                p_mean = p.data.mean()
                escape_direction = torch.sign(p.data - p_mean)
                escape_direction[escape_direction == 0] = 1.0
                escape_update = escape_direction * threshold
                final_update = (1 - alpha) * update_adam + alpha * gamma * escape_update
                p.data.add_(final_update)
        return loss

class DynamoV2AdaptiveSimple(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, c=0.075, s=3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, c=c, s=s, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.grad_norm_history, self.max_history = [], 100

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        for group in self.param_groups:
            base_lr, base_c, base_s, (beta1, beta2), eps, weight_decay = group['lr'], group['c'], group['s'], group['betas'], group['eps'], group['weight_decay']
            M2, total_params, total_grad_norm = 0.0, 0, 0.0
            num_params_with_grad = 0
            for p in group['params']:
                if p.grad is None: continue
                M2 += torch.sum(p.data ** 2).item()
                total_params += p.data.numel()
                total_grad_norm += p.grad.norm().item()
                num_params_with_grad += 1
            M2 = M2 / total_params if total_params > 0 else 0.0
            avg_grad_norm = total_grad_norm / max(1, num_params_with_grad)
            self.grad_norm_history.append(avg_grad_norm)
            if len(self.grad_norm_history) > self.max_history: self.grad_norm_history.pop(0)
            c_t = base_c
            if len(self.grad_norm_history) > 10:
                recent_avg = sum(self.grad_norm_history[-10:]) / 10
                if avg_grad_norm < 0.5 * recent_avg: c_t = base_c * 1.5
                elif avg_grad_norm > 2.0 * recent_avg: c_t = base_c * 0.75
            for p in group['params']:
                if p.grad is None: continue
                if weight_decay != 0: p.data.mul_(1 - base_lr * weight_decay)
                grad, state = p.grad.data, self.state[p]
                if len(state) == 0:
                    state['step'], state['exp_avg'], state['exp_avg_sq'] = 0, torch.zeros_like(p.data), torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1, bias_correction2 = 1 - beta1 ** state['step'], 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update_adam = - (base_lr / bias_correction1) * (exp_avg / denom)
                gamma = math.tanh(M2 / (base_s * base_s + eps))
                threshold = base_lr * c_t
                alpha = torch.clamp(1 - update_adam.abs() / (threshold + eps), min=0.0, max=1.0)
                p_mean = p.data.mean()
                escape_direction = torch.sign(p.data - p_mean)
                escape_direction[escape_direction == 0] = 1.0
                escape_update = escape_direction * threshold
                final_update = (1 - alpha) * update_adam + alpha * gamma * escape_update
                p.data.add_(final_update)
        return loss

# ==============================================================================
# §2. UTILITY FUNCTIONS
# ==============================================================================

# ---------------------
# GPU Monitor (from gpu_monitor.py)
# ---------------------
def _gpu_worker(logfile, interval=5):
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
                pass # Suppress errors if nvidia-smi is not found or fails
            time.sleep(interval)

def start_gpu_monitor(optimizer_name, logdir="results/gpu_logs", interval=5):
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, f"gpu_log_{optimizer_name.lower()}.csv")
    t = threading.Thread(target=_gpu_worker, args=(logfile, interval), daemon=True)
    t.start()
    print(f"[GPU Monitor] Logging to {logfile} every {interval}s")
    return t

# ---------------------
# Muon Optimizer Setup
# ---------------------
def create_muon_optimizer_with_adam(model, muon_lr=0.02, adam_lr=3e-4, **kwargs):
    """
    Correctly configures Muon by partitioning ResNet18 parameters.
    - Muon optimizes 2D hidden weights (conv layers).
    - AdamW optimizes everything else (biases, batchnorm, final FC layer).
    """
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        # Heuristic: Muon for 2D+ weights, but not the final classifier layer ('fc')
        if p.ndim >= 2 and 'fc' not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)

    print(f"[Muon] Optimizing {sum(p.numel() for p in muon_params)/1e6:.2f}M params with Muon")
    print(f"[Muon] Optimizing {sum(p.numel() for p in adam_params)/1e6:.2f}M params with AdamW")

    param_groups = [
        dict(params=muon_params, use_muon=True, lr=muon_lr, momentum=0.95, weight_decay=0),
        dict(params=adam_params, use_muon=False, lr=adam_lr, weight_decay=1e-2, betas=(0.9, 0.99))
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)

# ---------------------
# Training & Evaluation
# ---------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=3, **optimizer_kwargs):
    model = torchvision.models.resnet18(weights=None)
    # Change output layer for CIFAR-100
    model.fc = nn.Linear(model.fc.in_features, 100)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if optimizer_name == "Muon":
        optimizer = create_muon_optimizer_with_adam(model, **optimizer_kwargs)
    else:
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    train_losses, test_accs = [], []
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
        print(f"{optimizer_name} | Epoch {epoch+1:02d}/{epochs}: Loss={train_losses[-1]:.4f}, Acc={test_accs[-1]:.2f}%")

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
# §3. MAIN EXPERIMENT
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-100 specific transforms with standard normalization
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

    results = {}
    singular_spectra = {}
    
    # NOTE: Hyperparameters are kept from the original script. They may not be optimal for CIFAR-100.
    optimizers_config = [
        ("AdamW", AdamWWrapper, {"lr": 3e-4, "weight_decay": 1e-2}),
        ("Lion", Lion, {"lr": 1e-4, "weight_decay": 3e-2}),
        # Muon is handled specially, class is None. Kwargs are for the setup function.
        ("Muon", None, {"muon_lr": 0.02, "adam_lr": 3e-4}),
        ("Dynamo", TargetedDynamo, {"lr": 2e-4, "weight_decay": 1e-2}),
        ("DynamoV2", DynamoV2, {"lr": 1e-3, "c": 0.075, "s": 2.75, "weight_decay": 1e-2}),
        ("DynamoV2Adaptive", DynamoV2Adaptive, {"lr": 1e-3, "c": 0.075, "s": 2.75, "weight_decay": 1e-2, "adaptive_lr": True}),
        ("DynamoV2AdaptiveSimple", DynamoV2AdaptiveSimple, {"lr": 1e-3, "c": 0.075, "s": 2.75, "weight_decay": 1e-2}),
        ("RAdam", torch.optim.RAdam, {"lr": 2e-4, "weight_decay": 1e-2, "decoupled_weight_decay": True}),
    ]

    for name, opt_class, opt_kwargs in optimizers_config:
        print(f"\n===== Training with {name} (ResNet18 on CIFAR-100) =====")
        print(f"Hyperparameters: {opt_kwargs}")
        try:
            gpu_thread = start_gpu_monitor(name, interval=5)
            # Increased epochs for the more complex CIFAR-100 task
            losses, accs, model = train_and_eval(name, opt_class, trainloader, testloader, device, epochs=20, **opt_kwargs)
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
    fig.suptitle("ResNet18 Optimizer Comparison on CIFAR-100", fontsize=16)
    
    for name, (losses, accs) in results.items():
        axes[0, 0].plot(accs, label=name, marker='o', markersize=4, linewidth=2)
    axes[0, 0].set_xlabel("Epoch"), axes[0, 0].set_ylabel("Test Accuracy (%)")
    axes[0, 0].set_title("Test Accuracy"), axes[0, 0].legend(), axes[0, 0].grid(True, alpha=0.4)
    
    for name, (losses, accs) in results.items():
        axes[0, 1].plot(losses, label=name, marker='s', markersize=4, linewidth=2)
    axes[0, 1].set_xlabel("Epoch"), axes[0, 1].set_ylabel("Training Loss")
    axes[0, 1].set_title("Training Loss"), axes[0, 1].legend(), axes[0, 1].grid(True, alpha=0.4)
    
    for name, spectrum in singular_spectra.items():
        axes[1, 0].plot(spectrum, label=name, linewidth=2)
    axes[1, 0].set_yscale("log"), axes[1, 0].set_xlabel("Index"), axes[1, 0].set_ylabel("Singular Value")
    axes[1, 0].set_title("Singular Spectrum of Final Layer"), axes[1, 0].legend(), axes[1, 0].grid(True, alpha=0.4)
    
    final_accs = {name: res[1][-1] for name, res in results.items()}
    sorted_names = sorted(final_accs, key=final_accs.get, reverse=True)
    sorted_accs = [final_accs[name] for name in sorted_names]
    bars = axes[1, 1].bar(sorted_names, sorted_accs, alpha=0.8)
    axes[1, 1].set_ylabel("Final Test Accuracy (%)"), axes[1, 1].set_title("Final Performance")
    axes[1, 1].grid(True, axis='y', alpha=0.4), axes[1, 1].tick_params(axis='x', rotation=45, ha='right')
    for bar, acc in zip(bars, sorted_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = "results/cifar100_resnet/comprehensive_comparison.png"
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

if __name__ == "__main__":
    main()

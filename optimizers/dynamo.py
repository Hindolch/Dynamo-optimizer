# import torch

# class Dynamo(torch.optim.Optimizer):
#     def __init__(self, params, lr=1e-3, c=0.1, smooth=True, eps=1e-12,
#                  beta1=0.9, beta2=0.999, weight_decay=0.0):
#         defaults = dict(lr=lr, c=c, smooth=smooth, eps=eps,
#                         beta1=beta1, beta2=beta2, weight_decay=weight_decay)
#         super().__init__(params, defaults)

#     @torch.no_grad()
#     def step(self, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure()

#         for group in self.param_groups:
#             lr = group['lr']
#             c = group['c']
#             smooth = group['smooth']
#             eps = group['eps']
#             beta1 = group['beta1']
#             beta2 = group['beta2']
#             weight_decay = group['weight_decay']

#             for p in group['params']:
#                 if p.grad is None:
#                     continue

#                 grad = p.grad.data
#                 if weight_decay != 0:
#                     grad = grad.add(p.data, alpha=weight_decay)

#                 state = self.state[p]
#                 if len(state) == 0:
#                     state['step'] = 0
#                     state['exp_avg'] = torch.zeros_like(p.data)
#                     state['exp_avg_sq'] = torch.zeros_like(p.data)

#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
#                 state['step'] += 1

#                 exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

#                 bias_correction1 = 1 - beta1 ** state['step']
#                 bias_correction2 = 1 - beta2 ** state['step']

#                 mu = exp_avg / bias_correction1
#                 variance = exp_avg_sq / bias_correction2

#                 sigma = torch.sqrt(variance + eps)
#                 mu_norm = torch.sqrt((mu ** 2).sum() + eps)

#                 delta_a = -lr * (mu / sigma)
#                 floor = -lr * c * torch.sign(mu)

#                 if smooth:
#                     ratio = (delta_a.abs() / (lr * c + eps)).clamp(max=10.0)
#                     w = 1.0 / (1.0 + torch.exp(-10.0 * (ratio - 1.0)))
#                     update = (1.0 - w) * floor + w * delta_a
#                 else:
#                     update = torch.where(delta_a.abs() < (lr * c), floor, delta_a)

#                 p.data.add_(update)

#         return loss



import torch
import math

from torch.optim import Optimizer

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


# Alternative version with different floor mechanism
class DynamoAlternative(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, c=0.01, smooth=True, eps=1e-8,
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

                # Standard Adam-like update
                adam_update = -lr * (mu / sigma)
                
                # Alternative: Blend with sign-based update (Lion-like)
                sign_update = -lr * c * torch.sign(mu)
                
                if smooth:
                    # Dynamic mixing based on magnitude
                    alpha = torch.sigmoid(torch.log(adam_update.abs() / (lr * c + eps)))
                    update = alpha * adam_update + (1 - alpha) * sign_update
                else:
                    # Hard switching
                    update = torch.where(adam_update.abs() > lr * c, adam_update, sign_update)

                p.data.add_(update)

        return loss

class TargetedDynamo(torch.optim.Optimizer):
    """
    Targeted thermostat applied to Adam-style updates.
    - Applies floor only if:
        |grad| < grad_thresh  AND  sqrt(v_hat) > var_thresh  AND  tiny_steps >= persistence_K
    - Two modes for thresholds:
        * manual: pass grad_thresh and var_thresh (scalars) in opt kwargs
        * auto: set grad_thresh_mode='auto' and provide tau_g, tau_v (fractions of EMAs)
    State per-parameter:
        - exp_avg, exp_avg_sq : standard Adam moments
        - tiny_steps : integer tensor counting consecutive tiny updates (elementwise)
    """

    def __init__(self, params, lr=1e-3, c=0.02, smooth=True, eps=1e-8,
                 beta1=0.9, beta2=0.999, weight_decay=0.0,
                 persistence_K=2,
                 burn_in=2000,                  # NEW: no thermostat during first burn_in steps
                 grad_thresh_mode='auto',      # 'manual' or 'auto'
                 grad_thresh=None, var_thresh=None,
                 tau_g=0.5, tau_v=1.5,        # looser auto thresholds
                 ema_momentum=0.99):
        defaults = dict(lr=lr, c=c, smooth=smooth, eps=eps,
                        beta1=beta1, beta2=beta2, weight_decay=weight_decay,
                        persistence_K=persistence_K, burn_in=burn_in,
                        grad_thresh_mode=grad_thresh_mode,
                        grad_thresh=grad_thresh, var_thresh=var_thresh,
                        tau_g=tau_g, tau_v=tau_v, ema_momentum=ema_momentum)
        super().__init__(params, defaults)

        # global EMAs for auto threshold mode (per-param group)
        for group in self.param_groups:
            group.setdefault('_ema_grad_abs', 0.0)
            group.setdefault('_ema_sqrt_v', 0.0)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr, c, smooth, eps = group['lr'], group['c'], group['smooth'], group['eps']
            beta1, beta2, weight_decay = group['beta1'], group['beta2'], group['weight_decay']
            persistence_K, burn_in = group['persistence_K'], group['burn_in']
            grad_thresh_mode = group['grad_thresh_mode']
            grad_thresh_manual, var_thresh_manual = group['grad_thresh'], group['var_thresh']
            tau_g, tau_v, ema_momentum = group['tau_g'], group['tau_v'], group['ema_momentum']

            # local accumulators for auto threshold mode
            sum_mean_abs_grad = 0.0
            sum_mean_sqrt_v = 0.0
            param_count = 0

            # First pass: update moments and collect global stats (for auto mode)
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

                mean_abs_grad = grad.abs().mean().item()
                mean_sqrt_v = (exp_avg_sq / (1 - beta2 ** state['step'])).sqrt().mean().item()
                sum_mean_abs_grad += mean_abs_grad
                sum_mean_sqrt_v += mean_sqrt_v
                param_count += 1

            if grad_thresh_mode == 'auto' and param_count > 0:
                avg_abs_grad = sum_mean_abs_grad / param_count
                avg_sqrt_v = sum_mean_sqrt_v / param_count
                group['_ema_grad_abs'] = ema_momentum * group['_ema_grad_abs'] + (1 - ema_momentum) * avg_abs_grad
                group['_ema_sqrt_v'] = ema_momentum * group['_ema_sqrt_v'] + (1 - ema_momentum) * avg_sqrt_v

            # derive thresholds
            if grad_thresh_mode == 'auto':
                grad_thresh = tau_g * group['_ema_grad_abs'] + 1e-20
                var_thresh = tau_v * group['_ema_sqrt_v'] + 1e-20
            else:
                grad_thresh = grad_thresh_manual
                var_thresh = var_thresh_manual

            # Second pass: perform updates
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                if weight_decay != 0: grad = grad.add(p.data, alpha=weight_decay)

                state = self.state[p]
                exp_avg, exp_avg_sq, tiny_steps = state['exp_avg'], state['exp_avg_sq'], state['tiny_steps']

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                mu = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                sigma = torch.sqrt(v_hat + eps)

                delta_a = -lr * (mu / (sigma + eps))
                floor_strength = min(1.0, state['step'] / burn_in)  # ramp-up
                floor = -lr * (c * floor_strength) * torch.sign(mu)

                # if still in burn-in, just do Adam
                if state['step'] < burn_in:
                    p.data.add_(delta_a)
                    continue

                grad_abs, sqrt_v = grad.abs(), sigma
                grad_mask = grad_abs < grad_thresh
                var_mask = sqrt_v > var_thresh
                apply_condition = grad_mask & var_mask

                tiny_steps[apply_condition] += 1
                tiny_steps[~apply_condition] = 0
                persist_mask = tiny_steps >= persistence_K

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





class DynamoRayleigh(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0.0, log_dir="logs"):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(DynamoRayleigh, self).__init__(params, defaults)
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "dynamo_probe.csv")
        # write CSV header
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "grad_norm", "second_moment",
                "thermostat_trigger", "rayleigh_quotient", "adaptive_threshold"
            ])
        self._step = 0
        self._cached_loss = None
        
        # FIXED: Global RQ history for the optimizer (not per parameter)
        self.rq_history = []

    def set_closure(self, closure):
        """Set a closure that computes the loss for Hessian computation."""
        self._closure = closure

    def step(self, closure=None):
        # Store the closure for Hessian computation
        if closure is not None:
            self._closure = closure

        loss = None
            
        beta1, beta2 = self.defaults['betas']
        eps = self.defaults['eps']
        lr = self.defaults['lr']
        wd = self.defaults['weight_decay']
        self._step += 1
        
        # Collect all parameters for Hessian computation BEFORE applying updates
        all_params = []
        all_grads = []
        
        # First pass: collect data for Hessian computation
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                        
                    all_params.append(p)
                    grad = p.grad.clone()  # Clone to preserve original
                    all_grads.append(grad)
        
        # ===== Compute Rayleigh quotient BEFORE parameter updates =====
        rq = self._compute_rayleigh_quotient(all_params, all_grads)
        
        # ===== FIXED: Adaptive threshold logic =====
        thermostat_triggered = False
        adaptive_threshold = float('inf')  # Default to never trigger
        
        # Update global RQ history
        if not torch.isnan(torch.tensor(rq)):
            self.rq_history.append(rq)
            if len(self.rq_history) > 100:  # Keep last 100 values
                self.rq_history.pop(0)
            
            # Compute adaptive threshold once we have enough history
            if len(self.rq_history) > 10:
                avg_rq = sum(self.rq_history) / len(self.rq_history)
                # Calculate standard deviation
                variance = sum((x - avg_rq) ** 2 for x in self.rq_history) / len(self.rq_history)
                std_rq = variance ** 0.5
                
                adaptive_threshold = avg_rq + std_rq  # 1 standard deviation above mean
                # Alternative: use 1.5x mean as threshold
                # adaptive_threshold = 1.5 * avg_rq
                
                if rq > adaptive_threshold:
                    thermostat_triggered = True
                
                # Debug logging every 50 steps
                if self._step % 50 == 0:
                    print(f"Step {self._step}: RQ={rq:.1f}, Mean={avg_rq:.1f}, "
                          f"Std={std_rq:.1f}, Threshold={adaptive_threshold:.1f}, Trigger={thermostat_triggered}")
        
        # Now compute updates with curvature-based triggering
        all_updates = []
        
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                        
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
                    # Momentum & variance updates
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    update = exp_avg / denom
                    
                    # Apply thermostat mechanism when triggered
                    if thermostat_triggered:
                        # Impose minimum floor when curvature is unusually high
                        update = torch.where(
                            update.abs() < 1e-3,
                            torch.sign(update) * 1e-3,
                            update
                        )
                    
                    all_updates.append(update)
        
        # Apply the parameter updates
        with torch.no_grad():
            update_idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    update = all_updates[update_idx]
                    update_idx += 1
                    
                    # Weight decay (AdamW style)
                    if wd != 0:
                        p.data.mul_(1 - lr * wd)
                    
                    # Apply update
                    p.add_(update, alpha=-lr)
        
        # Log the metrics (using the first parameter's metrics as representative)
        if all_grads:
            first_grad = all_grads[0]
            grad_norm = first_grad.norm().item()
            
            # Get second moment from first parameter
            first_param = all_params[0]
            state = self.state[first_param]
            second_moment = state['exp_avg_sq'].mean().item()
            
            # Use the thermostat_triggered flag from above
            trigger = int(thermostat_triggered)
            
            # Log probe with adaptive threshold info
            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self._step, grad_norm, second_moment, trigger, rq, adaptive_threshold
                ])
        
        return loss

    def _compute_rayleigh_quotient(self, params, grads, mode="diag"):
        """Compute Rayleigh quotient.
        mode = "hvp" (exact) or "diag" (approx).
        """
        if not hasattr(self, '_closure') or self._closure is None:
            return float('nan')

        try:
            flat_grads = torch.cat([g.view(-1) for g in grads])
            grad_norm = flat_grads.norm()
            if grad_norm < 1e-12:
                return float('nan')
            v = flat_grads / grad_norm

            # Ensure all parameters require gradients
            for p in params:
                p.requires_grad_(True)

            if mode == "hvp":
                # Exact Hessian-vector product
                with torch.enable_grad():
                    loss = self._closure()
                    if loss is None:
                        return float('nan')
                        
                    first_grads = torch.autograd.grad(
                        loss, params, create_graph=True, retain_graph=True
                    )
                    
                    # Reshape v to match parameter shapes
                    v_params, offset = [], 0
                    for p in params:
                        numel = p.numel()
                        v_params.append(v[offset:offset+numel].view_as(p))
                        offset += numel
                    
                    # Compute gradient-vector dot product
                    grad_v = sum((g * v_p).sum() 
                               for g, v_p in zip(first_grads, v_params))
                    
                    # Compute Hessian-vector product
                    hvs = torch.autograd.grad(grad_v, params, retain_graph=False)
                    flat_hv = torch.cat([hv.view(-1) for hv in hvs])
                    
                    rq = torch.dot(v, flat_hv).item()
                    return rq

            elif mode == "diag":
                # Diagonal approximation (faster but less accurate)
                with torch.enable_grad():
                    loss = self._closure()
                    if loss is None:
                        return float('nan')
                        
                    first_grads = torch.autograd.grad(
                        loss, params, create_graph=True, retain_graph=True
                    )
                    
                    diag_elems = []
                    for g, p in zip(first_grads, params):
                        # Compute diagonal Hessian elements
                        grad_outputs = torch.ones_like(g)
                        grad2 = torch.autograd.grad(
                            g, p, grad_outputs=grad_outputs, 
                            retain_graph=True, only_inputs=True
                        )[0]
                        diag_elems.append(grad2.reshape(-1))
                    
                    flat_diag = torch.cat(diag_elems)
                    
                    # Approximate Rayleigh quotient using diagonal elements
                    rq_approx = torch.dot(flat_grads**2, flat_diag) / torch.dot(flat_grads, flat_grads)
                    return rq_approx.item()

        except Exception as e:
            print(f"Error in Rayleigh quotient computation: {e}")
            return float('nan')


class DynamoV2(torch.optim.Optimizer):
    """
    Implements the improved Dynamo optimizer (Version V2), which solves convergence issues using state-dependent regularization.
    Core improvement: Utilizes the second-moment of parameter groups to adjust the strength of the escape mechanism, making it automatically weaken during convergence.
    """
    def __init__(self, params, lr=1e-3,  c=0.075, s=5.727,betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        """
        Initializes the improved Dynamo optimizer.

        Args:
            params (iterable): Model parameters.
            lr (float, optional): Learning rate (default: 1e-3).
            c (float, optional): Relative escape strength coefficient (default: 0.1).
            s (float, optional): Feature scale parameter, defines the activation boundary of the escape mechanism (default: 0.01).
            betas (Tuple[float, float], optional): Coefficients for calculating momentum and RMSprop (default: (0.9, 0.999)).
            eps (float, optional): Term added to the denominator for numerical stability (default: 1e-8).
            weight_decay (float, optional): Weight decay coefficient (default: 0.01).
        """
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
            raise ValueError(f"Invalid s value: {s}")

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
            
            # Calculate the average second-moment M2 of the entire parameter group
            M2 = 0.0
            total_params = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                M2 += torch.sum(p.data ** 2).item()
                total_params += p.data.numel()
            
            if total_params > 0:
                M2 /= total_params  # E[p^2]
            else:
                M2 = 0.0
            
            # Calculate the state-dependent modulation factor γ
            if s > 0 and M2 > 0:
                gamma = math.tanh(M2 / (s * s))
            else:
                gamma = 0.0

            for p in group['params']:
                if p.grad is None:
                    continue

                # 1. AdamW: Decouple weight decay from gradient
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                
                # 2. Calculate first-order and second-order momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 3. Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 4. Calculate standard Adam update
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1
                update_adam = -step_size * (exp_avg / denom)

                # 5. Dynamo-V2 core logic: State-dependent soft escape mechanism
                threshold = lr * c
                
                # Calculate soft mixing factor α
                alpha = torch.clamp(1 - update_adam.abs() / threshold, min=0.0)
                
                # Calculate escape update
                p_mean = p.data.mean()
                escape_direction = torch.sign(p.data - p_mean)
                # Handle cases where p.data == p_mean to avoid zero updates
                escape_direction[escape_direction == 0] = 1.0
                escape_update = escape_direction * threshold
                
                # 6. Synthesize final update: Soft mixing + state modulation
                final_update = (1 - alpha) * update_adam + alpha * gamma * escape_update

                # 7. Apply final update
                p.data.add_(final_update)

        return loss
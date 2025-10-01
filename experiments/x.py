import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys, os
from torch.optim.optimizer import Optimizer
import csv

# Official Lion optimizer - install with: pip install lion-pytorch
from lion_pytorch import Lion

# Your custom optimizers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizers.adamw_wrapper import AdamWWrapper
from optimizers.dynamo import TargetedDynamo, DynamoV2
from gpu_monitor import start_gpu_monitor
from torch.optim import RAdam




# class DynamoRayleigh(Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
#                  eps=1e-8, weight_decay=0.0, log_dir="logs"):
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay)
#         super(DynamoRayleigh, self).__init__(params, defaults)
#         os.makedirs(log_dir, exist_ok=True)
#         self.log_file = os.path.join(log_dir, "dynamo_probe.csv")
#         # write CSV header
#         with open(self.log_file, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "step", "grad_norm", "second_moment",
#                 "thermostat_trigger", "rayleigh_quotient"
#             ])
#         self._step = 0
#         self._cached_loss = None

#     def set_closure(self, closure):
#         """Set a closure that computes the loss for Hessian computation."""
#         self._closure = closure

#     def step(self, closure=None):
#         # Store the closure for Hessian computation
#         if closure is not None:
#             self._closure = closure

#         loss = None
            
#         beta1, beta2 = self.defaults['betas']
#         eps = self.defaults['eps']
#         lr = self.defaults['lr']
#         wd = self.defaults['weight_decay']
#         self._step += 1
        
#         # Collect all parameters for Hessian computation BEFORE applying updates
#         all_params = []
#         all_grads = []
        
#         # First pass: collect data for Hessian computation
#         with torch.no_grad():
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue
                        
#                     all_params.append(p)
#                     grad = p.grad.clone()  # Clone to preserve original
#                     all_grads.append(grad)
        
#         # ===== Compute Rayleigh quotient BEFORE parameter updates =====
#         rq = self._compute_rayleigh_quotient(all_params, all_grads)
        
#         # Now compute updates with curvature-based triggering
#         all_updates = []
#         thermostat_triggered = False
        
#         with torch.no_grad():
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue
                        
#                     grad = p.grad
#                     state = self.state[p]
#                     if len(state) == 0:
#                         state['exp_avg'] = torch.zeros_like(p)
#                         state['exp_avg_sq'] = torch.zeros_like(p)
                    
#                     exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
#                     # Momentum & variance updates
#                     exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                     exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
#                     denom = exp_avg_sq.sqrt().add_(eps)
#                     update = exp_avg / denom
                    
#                     # ===== Curvature-based thermostat trigger =====
#                     # Use the computed Rayleigh quotient for triggering
#                     # if rq > 15 and not torch.isnan(torch.tensor(rq)):
#                     #     # impose minimum floor when curvature is high
#                     #     update = torch.where(
#                     #         update.abs() < 1e-3,
#                     #         torch.sign(update) * 1e-3,
#                     #         update
#                     #     )
#                     #     thermostat_triggered = True
#                     # Trigger when RQ is above the running average
#                     if not hasattr(state, 'rq_history'):
#                         state['rq_history'] = []

#                     if not torch.isnan(torch.tensor(rq)):
#                         state['rq_history'].append(rq)
#                         if len(state['rq_history']) > 100:  # Keep last 100 values
#                             state['rq_history'].pop(0)
                        
#                         if len(state['rq_history']) > 10:  # Need some history first
#                             avg_rq = sum(state['rq_history']) / len(state['rq_history'])
#                             if rq > 1.5 * avg_rq:  # Trigger when 50% above average
#                                 thermostat_triggered = True
                    
#                     all_updates.append(update)
        
#         # Apply the parameter updates
#         with torch.no_grad():
#             update_idx = 0
#             for group in self.param_groups:
#                 for p in group['params']:
#                     if p.grad is None:
#                         continue
                    
#                     update = all_updates[update_idx]
#                     update_idx += 1
                    
#                     # Weight decay (AdamW style)
#                     if wd != 0:
#                         p.data.mul_(1 - lr * wd)
                    
#                     # Apply update
#                     p.add_(update, alpha=-lr)
        
#         # Log the metrics (using the first parameter's metrics as representative)
#         if all_grads:
#             first_grad = all_grads[0]
#             grad_norm = first_grad.norm().item()
            
#             # Get second moment from first parameter
#             first_param = all_params[0]
#             state = self.state[first_param]
#             second_moment = state['exp_avg_sq'].mean().item()
            
#             # Use the thermostat_triggered flag from above
#             trigger = int(thermostat_triggered)
            
#             # Log probe
#             with open(self.log_file, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     self._step, grad_norm, second_moment, trigger, rq
#                 ])
        
#         return loss

#     def _compute_rayleigh_quotient(self, params, grads, mode="diag"):
#         """Compute Rayleigh quotient.
#         mode = "hvp" (exact) or "diag" (approx).
#         """
#         if not hasattr(self, '_closure') or self._closure is None:
#             return float('nan')

#         try:
#             flat_grads = torch.cat([g.view(-1) for g in grads])
#             grad_norm = flat_grads.norm()
#             if grad_norm < 1e-12:
#                 return float('nan')
#             v = flat_grads / grad_norm

#             # Ensure all parameters require gradients
#             for p in params:
#                 p.requires_grad_(True)

#             if mode == "hvp":
#                 # Exact Hessian-vector product
#                 with torch.enable_grad():
#                     loss = self._closure()
#                     if loss is None:
#                         return float('nan')
                        
#                     first_grads = torch.autograd.grad(
#                         loss, params, create_graph=True, retain_graph=True
#                     )
                    
#                     # Reshape v to match parameter shapes
#                     v_params, offset = [], 0
#                     for p in params:
#                         numel = p.numel()
#                         v_params.append(v[offset:offset+numel].view_as(p))
#                         offset += numel
                    
#                     # Compute gradient-vector dot product
#                     grad_v = sum((g * v_p).sum() 
#                                for g, v_p in zip(first_grads, v_params))
                    
#                     # Compute Hessian-vector product
#                     hvs = torch.autograd.grad(grad_v, params, retain_graph=False)
#                     flat_hv = torch.cat([hv.view(-1) for hv in hvs])
                    
#                     rq = torch.dot(v, flat_hv).item()
#                     return rq

#             elif mode == "diag":
#                 # Diagonal approximation (faster but less accurate)
#                 with torch.enable_grad():
#                     loss = self._closure()
#                     if loss is None:
#                         return float('nan')
                        
#                     first_grads = torch.autograd.grad(
#                         loss, params, create_graph=True, retain_graph=True
#                     )
                    
#                     diag_elems = []
#                     for g, p in zip(first_grads, params):
#                         # Compute diagonal Hessian elements
#                         grad_outputs = torch.ones_like(g)
#                         grad2 = torch.autograd.grad(
#                             g, p, grad_outputs=grad_outputs, 
#                             retain_graph=True, only_inputs=True
#                         )[0]
#                         diag_elems.append(grad2.reshape(-1))
                    
#                     flat_diag = torch.cat(diag_elems)
                    
#                     # Approximate Rayleigh quotient using diagonal elements
#                     rq_approx = torch.dot(flat_grads**2, flat_diag) / torch.dot(flat_grads, flat_grads)
#                     return rq_approx.item()

#         except Exception as e:
#             print(f"Error in Rayleigh quotient computation: {e}")
#             return float('nan')


# class DynamoRayleigh(Optimizer):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999),
#                  eps=1e-8, weight_decay=0.0, log_dir="logs"):
#         defaults = dict(lr=lr, betas=betas, eps=eps,
#                         weight_decay=weight_decay)
#         super(DynamoRayleigh, self).__init__(params, defaults)
#         os.makedirs(log_dir, exist_ok=True)
#         self.log_file = os.path.join(log_dir, "dynamo_probe.csv")
#         # write CSV header
#         with open(self.log_file, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "step", "grad_norm", "second_moment",
#                 "thermostat_trigger", "rayleigh_quotient"
#             ])
#         self._step = 0
#         self._cached_loss = None

#     def set_closure(self, closure):
#         """Set a closure that computes the loss for Hessian computation."""
#         self._closure = closure

#     @torch.no_grad()
#     def step(self, closure=None):
#         # Store the closure for Hessian computation (do not evaluate under no_grad)
#         if closure is not None:
#             self._closure = closure

#         loss = None
            
#         beta1, beta2 = self.defaults['betas']
#         eps = self.defaults['eps']
#         lr = self.defaults['lr']
#         wd = self.defaults['weight_decay']
#         self._step += 1
        
#         # Collect all parameters for batch Hessian computation
#         all_params = []
#         all_grads = []
#         all_updates = []
        
#         for group in self.param_groups:
#             for p in group['params']:
#                 if p.grad is None:
#                     continue
                    
#                 all_params.append(p)
#                 grad = p.grad
#                 all_grads.append(grad)
                
#                 state = self.state[p]
#                 if len(state) == 0:
#                     state['exp_avg'] = torch.zeros_like(p)
#                     state['exp_avg_sq'] = torch.zeros_like(p)
                
#                 exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
#                 # Momentum & variance updates
#                 exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
#                 exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
#                 denom = exp_avg_sq.sqrt().add_(eps)
#                 update = exp_avg / denom
                
#                 # ===== Thermostat trigger =====
#                 grad_norm = grad.norm().item()
#                 second_moment = exp_avg_sq.mean().item()
#                 trigger = 0
#                 if grad_norm < 1e-3 and second_moment > 1e-3:
#                     # impose minimum floor
#                     update = torch.where(
#                         update.abs() < 1e-3,
#                         torch.sign(update) * 1e-3,
#                         update
#                     )
#                     trigger = 1
                
#                 all_updates.append(update)
                
#                 # Weight decay (AdamW style)
#                 if wd != 0:
#                     p.data.mul_(1 - lr * wd)
                
#                 # Apply update
#                 p.add_(update, alpha=-lr)
        
#         # ===== Compute Rayleigh quotient for all parameters =====
#         rq = self._compute_rayleigh_quotient(all_params, all_grads)
        
#         # Log the metrics (using the first parameter's metrics as representative)
#         if all_grads:
#             first_grad = all_grads[0]
#             first_update = all_updates[0]
#             grad_norm = first_grad.norm().item()
            
#             # Get second moment from first parameter
#             first_param = all_params[0]
#             state = self.state[first_param]
#             second_moment = state['exp_avg_sq'].mean().item()
            
#             # Determine if thermostat was triggered for any parameter
#             trigger = int(any(grad.norm().item() < 1e-3 and 
#                             self.state[p]['exp_avg_sq'].mean().item() > 1e-3 
#                             for p, grad in zip(all_params, all_grads)))
            
#             # Log probe
#             with open(self.log_file, "a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     self._step, grad_norm, second_moment, trigger, rq
#                 ])
        
#         return loss

#     # def _compute_rayleigh_quotient(self, params, grads):
#     #     """Compute Rayleigh quotient using Hessian-vector product."""
#     #     if not hasattr(self, '_closure') or self._closure is None:
#     #         return float('nan')
            
#     #     try:
#     #         # Flatten all gradients to create direction vector
#     #         flat_grads = torch.cat([g.view(-1) for g in grads])
#     #         grad_norm = flat_grads.norm()
            
#     #         if grad_norm < 1e-12:
#     #             return float('nan')
                
#     #         # Normalize direction vector
#     #         v = flat_grads / grad_norm
            
#     #         # Create a list to store Hv
#     #         hvs = []
            
#     #         # Compute Hessian-vector product using autograd
#     #         # We need to ensure parameters have gradients enabled
#     #         for p in params:
#     #             p.requires_grad_(True)
            
#     #         # Recompute loss with gradients enabled
#     #         loss = self._closure()
            
#     #         if loss is None:
#     #             return float('nan')
                
#     #         # Compute gradients
#     #         first_grads = torch.autograd.grad(
#     #             loss, params, create_graph=True, retain_graph=True
#     #         )
            
#     #         # Reshape v to match parameter shapes
#     #         offset = 0
#     #         v_params = []
#     #         for p in params:
#     #             numel = p.numel()
#     #             v_param = v[offset:offset+numel].view_as(p)
#     #             v_params.append(v_param)
#     #             offset += numel
            
#     #         # Compute Hessian-vector product
#     #         grad_v_product = sum(
#     #             torch.sum(g * v_p) 
#     #             for g, v_p in zip(first_grads, v_params)
#     #         )
            
#     #         hvs = torch.autograd.grad(
#     #             grad_v_product, params, retain_graph=False
#     #         )
            
#     #         # Flatten Hv and compute Rayleigh quotient
#     #         flat_hv = torch.cat([hv.view(-1) for hv in hvs])
            
#     #         # Rayleigh quotient: v^T H v / v^T v
#     #         # Since v is normalized, v^T v = 1
#     #         rq = torch.dot(v, flat_hv).item()
            
#     #         return rq
            
#     #     except Exception as e:
#     #         # If anything goes wrong, return nan
#     #         return float('nan')
#     def _compute_rayleigh_quotient(self, params, grads, mode="diag"):
#         """Compute Rayleigh quotient.
#         mode = "hvp" (exact) or "diag" (approx).
#         """
#         if not hasattr(self, '_closure') or self._closure is None:
#             return float('nan')

#         try:
#             flat_grads = torch.cat([g.view(-1) for g in grads])
#             grad_norm = flat_grads.norm()
#             if grad_norm < 1e-12:
#                 return float('nan')
#             v = flat_grads / grad_norm

#             if mode == "hvp":
#                 # exact Hessian-vector product
#                 loss = self._closure()  # must re-run with create_graph=True
#                 first_grads = torch.autograd.grad(loss, params,
#                                                 create_graph=True,
#                                                 retain_graph=True)
#                 v_params, offset = [], 0
#                 for p in params:
#                     numel = p.numel()
#                     v_params.append(v[offset:offset+numel].view_as(p))
#                     offset += numel
#                 grad_v = sum((g * v_p).sum()
#                             for g, v_p in zip(first_grads, v_params))
#                 hvs = torch.autograd.grad(grad_v, params, retain_graph=False)
#                 flat_hv = torch.cat([hv.view(-1) for hv in hvs])
#                 rq = torch.dot(v, flat_hv).item()
#                 return rq

#             elif mode == "diag":
#                 # diagonal approximation
#                 loss = self._closure()
#                 first_grads = torch.autograd.grad(loss, params,
#                                                 create_graph=True,
#                                                 retain_graph=True)
#                 diag_elems = []
#                 for g, p in zip(first_grads, params):
#                     # d/dp (g) = diagonal Hessian element
#                     grad2 = torch.autograd.grad(g.sum(), p,
#                                                 retain_graph=True)[0]
#                     diag_elems.append(grad2.reshape(-1))
#                 flat_diag = torch.cat(diag_elems)
#                 rq_approx = (flat_grads**2 @ flat_diag) / (flat_grads**2).sum()
#                 return rq_approx.item()

#         except Exception:
#             return float('nan')



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

# ---------------------
# Training utility with proper Lion hyperparameters
# ---------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=3, **optimizer_kwargs):
    # ResNet18 backbone
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    train_losses, test_accs = [], []

    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Provide a closure so optimizers that need to re-evaluate loss (e.g., HVP in DynamoRayleigh)
            # can do so without us calling backward again.
            def closure():
                with torch.enable_grad():
                    out = model(inputs)
                    return criterion(out, labels)

            optimizer.step(closure=closure)
            running_loss += loss.item()

        train_losses.append(running_loss / len(trainloader))

        # test
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
        print(f"{optimizer_name} | Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Acc={test_accs[-1]:.2f}%")

    return train_losses, test_accs, model


# ---------------------
# Spectral analysis
# ---------------------
def compute_singular_values(model, layer_name="fc.weight"):
    # obtain parameter tensor and move to cpu
    param = dict(model.named_parameters())[layer_name].detach().cpu()
    # reshape conv kernels to 2D if needed
    param_2d = param.reshape(param.shape[0], -1)
    # compute singular values on CPU
    _, S, _ = torch.linalg.svd(param_2d, full_matrices=False)
    return S.cpu().numpy()


# ---------------------
# Main
# ---------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    results = {}
    singular_spectra = {}

    # Define optimizers with proper hyperparameters based on Lion paper recommendations
    optimizers_config = [
        # AdamW baseline
        ("AdamW", AdamWWrapper, {
            "lr": 0.0003, 
            "weight_decay": 1e-2
        
        }),
        
        # Official Lion with recommended hyperparameters
        # Lion paper suggests 3-10x smaller lr and 3-10x larger weight_decay than AdamW
        ("Lion", Lion, {
            "lr": 0.0001,           # 3x smaller than AdamW
            "weight_decay": 3e-2, # 3x larger than AdamW
            # Note: betas default to (0.9, 0.99) in lion-pytorch package
            # Optional: use_triton=True for fused kernels (requires: pip install triton -U --pre)
        }),
        
        # Your Dynamo optimizer
        ("Dynamo", TargetedDynamo, {
            "lr": 0.0002, 
            "weight_decay": 0.01
        }),
        ("DynamoV2", DynamoV2, {
            "lr": 0.0002,
            "weight_decay": 0.01
        }),
        
        ("RAdam", RAdam, {
            "lr": 0.0002, 
            "weight_decay": 0.01,
            "decoupled_weight_decay": True
        }),
    ]

    for name, opt_class, opt_kwargs in optimizers_config:
        print(f"\n===== Training with {name} (ResNet18) =====")
        print(f"Hyperparameters: {opt_kwargs}")
        
        # start GPU monitor for this optimizer
        gpu_thread = start_gpu_monitor(name, interval=5)

        losses, accs, model = train_and_eval(name, opt_class, trainloader, testloader, device, **opt_kwargs)
        results[name] = (losses, accs)

        # save checkpoint and compute singular spectrum of final layer
        os.makedirs("results/cifar10_resnet", exist_ok=True)
        ckpt_path = os.path.join("results", "cifar10_resnet", f"{name.lower()}_resnet18.pth")
        torch.save(model.state_dict(), ckpt_path)

        singular_spectra[name] = compute_singular_values(model, layer_name="fc.weight")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Test Accuracy
    for name, (losses, accs) in results.items():
        axes[0, 0].plot(accs, label=name, marker='o', linewidth=2)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Test Accuracy (%)")
    axes[0, 0].set_title("CIFAR-10 ResNet18 Test Accuracy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training Loss
    for name, (losses, accs) in results.items():
        axes[0, 1].plot(losses, label=name, marker='s', linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Training Loss")
    axes[0, 1].set_title("CIFAR-10 ResNet18 Training Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Singular Spectrum
    for name, spectrum in singular_spectra.items():
        axes[1, 0].plot(spectrum, label=name, linewidth=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Index")
    axes[1, 0].set_ylabel("Singular Value")
    axes[1, 0].set_title("Singular Spectrum of fc.weight")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Final Performance Summary
    final_accs = [results[name][1][-1] for name in results.keys()]
    bars = axes[1, 1].bar(results.keys(), final_accs, 
                         color=['blue', 'orange', 'green', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel("Final Test Accuracy (%)")
    axes[1, 1].set_title("Final Performance Comparison")
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("results/cifar10_resnet/comprehensive_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for name, (losses, accs) in results.items():
        final_loss = losses[-1]
        final_acc = accs[-1]
        best_acc = max(accs)
        best_epoch = accs.index(best_acc) + 1
        
        print(f"{name:12} | Final: {final_acc:6.2f}% | Best: {best_acc:6.2f}% (Epoch {best_epoch})")
    
    print(f"{'='*60}")
    
    # Analysis of singular spectra
    print("\nSINGULAR SPECTRUM ANALYSIS:")
    for name, spectrum in singular_spectra.items():
        top_5_ratio = spectrum[:5].sum() / spectrum.sum()
        effective_rank = (spectrum.sum()**2) / (spectrum**2).sum()  # Participation ratio
        print(f"{name:12} | Top-5 concentration: {top_5_ratio:.3f} | Effective rank: {effective_rank:.2f}")


if __name__ == "__main__":
    print("Make sure you have installed lion-pytorch:")
    print("pip install lion-pytorch")
    print("For fused kernels (optional): pip install triton -U --pre")
    print()
    main()
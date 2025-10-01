

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
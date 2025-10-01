import torch

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
                
                # CORRECTED: Use interpolated momentum for update
                update = (beta1 * exp_avg + (1 - beta1) * grad).sign()
                p.data.add_(update, alpha=-lr)
                
                # Then update momentum for next iteration
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss

# Your original version for comparison
class LionOriginal(torch.optim.Optimizer):
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
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)  # Update first
                
                update = exp_avg.sign()  # Then use updated momentum
                p.data.add_(update, alpha=-lr)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # Another update?

        return loss
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from copy import deepcopy

# class TrainingDynamicsTracker:
#     def __init__(self):
#         self.grad_norms = []
#         self.losses = []
#         self.param_changes = []
#         self.update_magnitudes = []
#         self.steps = []
#         self.step_counter = 0

#     def record(self, model, loss, prev_params, optimizer_name=""):
#         self.steps.append(self.step_counter)
#         self.step_counter += 1
#         self.losses.append(loss.item())

#         # Gradient norm
#         total_grad_norm = 0.0
#         for p in model.parameters():
#             if p.grad is not None:
#                 total_grad_norm += p.grad.data.norm(2).item() ** 2
#         total_grad_norm = total_grad_norm ** 0.5
#         self.grad_norms.append(total_grad_norm)

#         # Parameter change magnitude
#         if prev_params is not None:
#             param_change = 0.0
#             for p, prev_p in zip(model.parameters(), prev_params):
#                 param_change += torch.sum((p.data - prev_p.data) ** 2).item()
#             self.param_changes.append(param_change ** 0.5)
#         else:
#             self.param_changes.append(0.0)

#     def plot_comparison(self, other_tracker, name1="AdamW", name2="DynamoV3"):
#         """Plot side-by-side comparison of two optimizers."""
#         fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
#         # Loss comparison
#         axs[0, 0].plot(self.steps, self.losses, label=name1, linewidth=2, alpha=0.8)
#         axs[0, 0].plot(other_tracker.steps, other_tracker.losses, label=name2, linewidth=2, alpha=0.8)
#         axs[0, 0].set_xlabel("Steps")
#         axs[0, 0].set_ylabel("Loss")
#         axs[0, 0].set_title("Loss Trajectory")
#         axs[0, 0].legend()
#         axs[0, 0].grid(True, alpha=0.3)

#         # Gradient norm comparison
#         axs[0, 1].plot(self.steps, self.grad_norms, label=name1, linewidth=2, alpha=0.8)
#         axs[0, 1].plot(other_tracker.steps, other_tracker.grad_norms, label=name2, linewidth=2, alpha=0.8)
#         axs[0, 1].set_xlabel("Steps")
#         axs[0, 1].set_ylabel("Gradient Norm")
#         axs[0, 1].set_title("Gradient Magnitude (freezing indicator)")
#         axs[0, 1].legend()
#         axs[0, 1].grid(True, alpha=0.3)
#         axs[0, 1].set_yscale('log')

#         # Parameter change comparison
#         axs[1, 0].plot(self.steps, self.param_changes, label=name1, linewidth=2, alpha=0.8)
#         axs[1, 0].plot(other_tracker.steps, other_tracker.param_changes, label=name2, linewidth=2, alpha=0.8)
#         axs[1, 0].set_xlabel("Steps")
#         axs[1, 0].set_ylabel("Parameter Change Magnitude")
#         axs[1, 0].set_title("Update Step Size (escape indicator)")
#         axs[1, 0].legend()
#         axs[1, 0].grid(True, alpha=0.3)
#         axs[1, 0].set_yscale('log')

#         # Moving average of gradient norms (smoothed view)
#         window = 50
#         if len(self.grad_norms) > window:
#             adamw_smooth = np.convolve(self.grad_norms, np.ones(window)/window, mode='valid')
#             dynamo_smooth = np.convolve(other_tracker.grad_norms, np.ones(window)/window, mode='valid')
#             steps_smooth = self.steps[window-1:]
            
#             axs[1, 1].plot(steps_smooth, adamw_smooth, label=f"{name1} (smoothed)", linewidth=2, alpha=0.8)
#             axs[1, 1].plot(steps_smooth, dynamo_smooth, label=f"{name2} (smoothed)", linewidth=2, alpha=0.8)
#             axs[1, 1].set_xlabel("Steps")
#             axs[1, 1].set_ylabel("Smoothed Gradient Norm")
#             axs[1, 1].set_title("Gradient Trend (detects plateaus)")
#             axs[1, 1].legend()
#             axs[1, 1].grid(True, alpha=0.3)

#         plt.tight_layout()
#         plt.savefig("saddle_escape_comparison.png", dpi=300, bbox_inches='tight')
#         plt.show()

#         # Print statistics about freezing behavior
#         self._print_freezing_stats(other_tracker, name1, name2)

#     def _print_freezing_stats(self, other_tracker, name1, name2):
#         """Analyze and print statistics about gradient freezing."""
#         print("\n" + "="*60)
#         print("GRADIENT FREEZING ANALYSIS")
#         print("="*60)
        
#         # Define "frozen" as gradient norm below 10% of initial value
#         adamw_initial = np.mean(self.grad_norms[:10])
#         dynamo_initial = np.mean(other_tracker.grad_norms[:10])
        
#         adamw_frozen_steps = sum(1 for g in self.grad_norms if g < 0.1 * adamw_initial)
#         dynamo_frozen_steps = sum(1 for g in other_tracker.grad_norms if g < 0.1 * dynamo_initial)
        
#         print(f"\n{name1}:")
#         print(f"  Initial gradient norm: {adamw_initial:.4f}")
#         print(f"  Final gradient norm: {self.grad_norms[-1]:.4f}")
#         print(f"  Steps with frozen gradients (<10% initial): {adamw_frozen_steps}/{len(self.grad_norms)}")
#         print(f"  Average parameter change: {np.mean(self.param_changes):.6f}")
        
#         print(f"\n{name2}:")
#         print(f"  Initial gradient norm: {dynamo_initial:.4f}")
#         print(f"  Final gradient norm: {other_tracker.grad_norms[-1]:.4f}")
#         print(f"  Steps with frozen gradients (<10% initial): {dynamo_frozen_steps}/{len(other_tracker.grad_norms)}")
#         print(f"  Average parameter change: {np.mean(other_tracker.param_changes):.6f}")
        
#         print(f"\n{name2} maintains {(1 - dynamo_frozen_steps/len(other_tracker.grad_norms))*100:.1f}% active gradient")
#         print(f"{name1} freezes for {(adamw_frozen_steps/len(self.grad_norms))*100:.1f}% of training")
#         print("="*60)


# def train_with_optimizer(model, optimizer, train_loader, criterion, device, steps_limit=1000):
#     """Train model and track dynamics."""
#     tracker = TrainingDynamicsTracker()
#     model.train()
    
#     step_count = 0
#     prev_params = None
    
#     print(f"Starting training for {steps_limit} steps...")
    
#     while step_count < steps_limit:
#         for images, labels in train_loader:
#             if step_count >= steps_limit:
#                 break
                
#             images, labels = images.to(device), labels.to(device)
            
#             # Save previous params
#             prev_params = [p.clone().detach() for p in model.parameters()]
            
#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             tracker.record(model, loss, prev_params)
#             step_count += 1
            
#             # Print progress every 100 steps
#             if step_count % 100 == 0:
#                 print(f"Step {step_count}/{steps_limit}, Loss: {loss.item():.4f}")
    
#     print(f"Training completed! Total steps: {step_count}")
#     return tracker


# def compare_optimizers_on_cifar(steps=5000):
#     """Compare AdamW vs DynamoV3 on CIFAR-10."""
#     import torchvision
#     import torchvision.transforms as transforms
#     from optimizers.adamw_wrapper import AdamWWrapper
#     from optimizers.dynamo import DynamoV3
    
#     print("Setting up device...")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print(f"Using device: {device}")
    
#     print("Loading CIFAR-10 dataset...")
#     # Load CIFAR-10
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])
    
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
#                                            download=True, transform=transform)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
#                                              shuffle=True, num_workers=2)
#     print(f"Dataset loaded. Total batches: {len(trainloader)}")
    
#     # Simple ResNet18
#     def get_model():
#         print("Creating ResNet18 model...")
#         model = torchvision.models.resnet18(weights=None)
#         model.fc = nn.Linear(model.fc.in_features, 10)
#         return model
    
#     criterion = nn.CrossEntropyLoss()
    
#     print("\n===== Training with AdamW =====")
#     model_adamw = get_model().to(device)
#     optimizer_adamw = AdamWWrapper(model_adamw.parameters(), lr=3e-4, weight_decay=1e-2)
#     tracker_adamw = train_with_optimizer(model_adamw, optimizer_adamw, trainloader, criterion, device, steps)
    
#     print("\n===== Training with DynamoV3 =====")
#     model_dynamo = get_model().to(device)
#     optimizer_dynamo = DynamoV3(model_dynamo.parameters(), lr=1e-3, c=0.075, s=3, weight_decay=1e-2)
#     tracker_dynamo = train_with_optimizer(model_dynamo, optimizer_dynamo, trainloader, criterion, device, steps)
    
#     print("\n===== Generating comparison plots =====")
#     # Visualize comparison
#     tracker_adamw.plot_comparison(tracker_dynamo, "AdamW", "DynamoV3")


# if __name__ == "__main__":
#     compare_optimizers_on_cifar(steps=5000)


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class TrainingDynamicsTracker:
    def __init__(self, device='cpu'):
        self.grad_norms = []
        self.losses = []
        self.param_changes = []
        self.steps = []
        self.grad_variances = []
        self.update_alignments = []
        self.local_sharpness = []
        self.device = device
        self.step_counter = 0
        self.loss_landscape = None  # store last landscape grid

    def record(self, model, loss, prev_params, optimizer_name="", lr=None):
        self.steps.append(self.step_counter)
        self.step_counter += 1
        self.losses.append(loss.item())

        total_grad_norm = 0.0
        grads = []
        updates = []
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach().flatten()
                grads.append(g)
                total_grad_norm += g.norm(2).item() ** 2
                if prev_params is not None:
                    updates.append((p.data - prev_params.pop(0).data).flatten())

        total_grad_norm = total_grad_norm ** 0.5
        self.grad_norms.append(total_grad_norm)

        # === Parameter change magnitude (update step size) ===
        if len(updates) > 0:
            update_vec = torch.cat(updates)
            self.param_changes.append(update_vec.norm(2).item())
        else:
            self.param_changes.append(0.0)

        # === 1️⃣ Gradient Variance ===
        if len(grads) > 0:
            grads_concat = torch.cat(grads)
            self.grad_variances.append(torch.var(grads_concat).item())
        else:
            self.grad_variances.append(0.0)

        # === 2️⃣ Update / Gradient Alignment ===
        if len(updates) > 0 and len(grads) > 0:
            g_vec = torch.cat(grads)
            u_vec = torch.cat(updates)
            cos_sim = F.cosine_similarity(g_vec.unsqueeze(0), u_vec.unsqueeze(0)).item()
            self.update_alignments.append(cos_sim)
        else:
            self.update_alignments.append(0.0)

        # === 3️⃣ Local Sharpness ===
        self.local_sharpness.append(self._estimate_local_sharpness(model, loss))

        # === 4️⃣ Loss Landscape (optional visualization every 500 steps) ===
        if self.step_counter % 500 == 0:
            self.loss_landscape = self._compute_loss_landscape(model, loss)

    def _estimate_local_sharpness(self, model, loss, epsilon=1e-3):
        """Approximate local sharpness via finite differences."""
        with torch.no_grad():
            perturbation = [torch.randn_like(p) * epsilon for p in model.parameters()]
            for p, delta in zip(model.parameters(), perturbation):
                p.add_(delta)
            loss_pos = self._forward_loss(model)
            for p, delta in zip(model.parameters(), perturbation):
                p.sub_(2 * delta)
            loss_neg = self._forward_loss(model)
            for p, delta in zip(model.parameters(), perturbation):
                p.add_(delta)
        sharpness = abs(loss_pos - 2 * loss.item() + loss_neg) / (epsilon ** 2)
        return sharpness

    def _forward_loss(self, model):
        """Compute loss on a small batch for sharpness estimation."""
        # Dummy forward if dataloader not provided
        return 0.0

    def _compute_loss_landscape(self, model, loss, scale=0.05, grid_size=20):
        """Project the loss landscape in 2D around current parameters."""
        params = torch.cat([p.data.flatten() for p in model.parameters()])
        direction1 = torch.randn_like(params)
        direction2 = torch.randn_like(params)
        direction1 /= direction1.norm()
        direction2 /= direction2.norm()
        alphas = np.linspace(-scale, scale, grid_size)
        betas = np.linspace(-scale, scale, grid_size)
        loss_surface = np.zeros((grid_size, grid_size))
        with torch.no_grad():
            for i, a in enumerate(alphas):
                for j, b in enumerate(betas):
                    new_params = params + a * direction1 + b * direction2
                    idx = 0
                    for p in model.parameters():
                        numel = p.numel()
                        p.data.copy_(new_params[idx:idx+numel].view_as(p))
                        idx += numel
                    loss_surface[i, j] = self._forward_loss(model)
        return loss_surface

    def _print_freezing_stats(self, other_tracker, name1, name2):
        """Analyze and print statistics about gradient freezing."""
        print("\n" + "="*60)
        print("GRADIENT FREEZING ANALYSIS")
        print("="*60)
        
        # Define "frozen" as gradient norm below 10% of initial value
        adamw_initial = np.mean(self.grad_norms[:10])
        dynamo_initial = np.mean(other_tracker.grad_norms[:10])
        
        adamw_frozen_steps = sum(1 for g in self.grad_norms if g < 0.1 * adamw_initial)
        dynamo_frozen_steps = sum(1 for g in other_tracker.grad_norms if g < 0.1 * dynamo_initial)
        
        print(f"\n{name1}:")
        print(f"  Initial gradient norm: {adamw_initial:.4f}")
        print(f"  Final gradient norm: {self.grad_norms[-1]:.4f}")
        print(f"  Steps with frozen gradients (<10% initial): {adamw_frozen_steps}/{len(self.grad_norms)}")
        print(f"  Average parameter change: {np.mean(self.param_changes):.6f}")
        
        print(f"\n{name2}:")
        print(f"  Initial gradient norm: {dynamo_initial:.4f}")
        print(f"  Final gradient norm: {other_tracker.grad_norms[-1]:.4f}")
        print(f"  Steps with frozen gradients (<10% initial): {dynamo_frozen_steps}/{len(other_tracker.grad_norms)}")
        print(f"  Average parameter change: {np.mean(other_tracker.param_changes):.6f}")
        
        print(f"\n{name2} maintains {(1 - dynamo_frozen_steps/len(other_tracker.grad_norms))*100:.1f}% active gradient")
        print(f"{name1} freezes for {(adamw_frozen_steps/len(self.grad_norms))*100:.1f}% of training")
        print("="*60)

    def plot_comparison(self, other_tracker, name1="AdamW", name2="DynamoV3"):
        """Plot side-by-side comparison of two optimizers."""
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Loss comparison
        axs[0, 0].plot(self.steps, self.losses, label=name1, linewidth=2, alpha=0.8)
        axs[0, 0].plot(other_tracker.steps, other_tracker.losses, label=name2, linewidth=2, alpha=0.8)
        axs[0, 0].set_xlabel("Steps")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].set_title("Loss Trajectory")
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # Gradient norm comparison
        axs[0, 1].plot(self.steps, self.grad_norms, label=name1, linewidth=2, alpha=0.8)
        axs[0, 1].plot(other_tracker.steps, other_tracker.grad_norms, label=name2, linewidth=2, alpha=0.8)
        axs[0, 1].set_xlabel("Steps")
        axs[0, 1].set_ylabel("Gradient Norm")
        axs[0, 1].set_title("Gradient Magnitude (freezing indicator)")
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        axs[0, 1].set_yscale('log')

        # Parameter change comparison
        axs[1, 0].plot(self.steps, self.param_changes, label=name1, linewidth=2, alpha=0.8)
        axs[1, 0].plot(other_tracker.steps, other_tracker.param_changes, label=name2, linewidth=2, alpha=0.8)
        axs[1, 0].set_xlabel("Steps")
        axs[1, 0].set_ylabel("Parameter Change Magnitude")
        axs[1, 0].set_title("Update Step Size (escape indicator)")
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        axs[1, 0].set_yscale('log')

        # Moving average of gradient norms (smoothed view)
        window = 50
        if len(self.grad_norms) > window and len(other_tracker.grad_norms) > window:
            adamw_smooth = np.convolve(self.grad_norms, np.ones(window)/window, mode='valid')
            dynamo_smooth = np.convolve(other_tracker.grad_norms, np.ones(window)/window, mode='valid')
            steps_smooth = self.steps[window-1:]

            axs[1, 1].plot(steps_smooth, adamw_smooth, label=f"{name1} (smoothed)", linewidth=2, alpha=0.8)
            axs[1, 1].plot(steps_smooth, dynamo_smooth, label=f"{name2} (smoothed)", linewidth=2, alpha=0.8)
            axs[1, 1].set_xlabel("Steps")
            axs[1, 1].set_ylabel("Smoothed Gradient Norm")
            axs[1, 1].set_title("Gradient Trend (detects plateaus)")
            axs[1, 1].legend()
            axs[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("saddle_escape_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistics about freezing behavior
        self._print_freezing_stats(other_tracker, name1, name2)

    def plot_extra_metrics(self):
        """Plot gradient variance, update alignment, and sharpness."""
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))

        axs[0].plot(self.steps, self.grad_variances, label="Gradient Variance")
        axs[0].set_title("Gradient Variance (Roughness Indicator)")
        axs[0].set_yscale('log')
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(self.steps, self.update_alignments, label="Update-Gradient Alignment", color='orange')
        axs[1].set_title("Update–Gradient Cosine Similarity")
        axs[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(self.steps, self.local_sharpness, label="Local Sharpness", color='red')
        axs[2].set_title("Local Sharpness (Loss Curvature Approximation)")
        axs[2].set_yscale('log')
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("extra_optimizer_metrics.png", dpi=300)
        plt.show()

        if self.loss_landscape is not None:
            plt.figure(figsize=(8,6))
            plt.imshow(self.loss_landscape, origin='lower', cmap='viridis')
            plt.colorbar(label="Loss")
            plt.title("2D Loss Landscape Around Current Parameters")
            plt.xlabel("Direction 1")
            plt.ylabel("Direction 2")
            plt.savefig("loss_landscape.png", dpi=300)
            plt.show()


def train_with_optimizer(model, optimizer, train_loader, criterion, device, steps_limit=1000):
    """Train model and track dynamics."""
    tracker = TrainingDynamicsTracker(device=device)
    model.train()
    
    step_count = 0
    prev_params = None
    
    print(f"Starting training for {steps_limit} steps...")
    
    while step_count < steps_limit:
        for images, labels in train_loader:
            if step_count >= steps_limit:
                break
                
            images, labels = images.to(device), labels.to(device)
            
            # Save previous params
            prev_params = [p.clone().detach() for p in model.parameters()]
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            tracker.record(model, loss, prev_params)
            step_count += 1
            
            # Print progress every 100 steps
            if step_count % 100 == 0:
                print(f"Step {step_count}/{steps_limit}, Loss: {loss.item():.4f}")
    
    print(f"Training completed! Total steps: {step_count}")
    return tracker


def compare_optimizers_on_cifar(steps=5000):
    """Compare AdamW vs DynamoV3 on CIFAR-10."""
    import torchvision
    import torchvision.transforms as transforms
    from optimizers.adamw_wrapper import AdamWWrapper
    from optimizers.dynamo import DynamoV3
    import torch.nn as nn
    
    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    
    print("Loading CIFAR-10 dataset...")
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, 
                                             shuffle=True, num_workers=2)
    print(f"Dataset loaded. Total batches: {len(trainloader)}")
    
    # Simple ResNet18
    def get_model():
        print("Creating ResNet18 model...")
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        return model
    
    criterion = nn.CrossEntropyLoss()
    
    print("\n===== Training with AdamW =====")
    model_adamw = get_model().to(device)
    optimizer_adamw = AdamWWrapper(model_adamw.parameters(), lr=3e-4, weight_decay=1e-2)
    tracker_adamw = train_with_optimizer(model_adamw, optimizer_adamw, trainloader, criterion, device, steps)
    
    print("\n===== Training with DynamoV3 =====")
    model_dynamo = get_model().to(device)
    optimizer_dynamo = DynamoV3(model_dynamo.parameters(), lr=1e-3, c=0.075, s=3, weight_decay=1e-2)
    tracker_dynamo = train_with_optimizer(model_dynamo, optimizer_dynamo, trainloader, criterion, device, steps)
    
    print("\n===== Generating comparison plots =====")
    # Visualize comparison
    tracker_adamw.plot_comparison(tracker_dynamo, "AdamW", "DynamoV3")


if __name__ == "__main__":
    compare_optimizers_on_cifar(steps=6000)
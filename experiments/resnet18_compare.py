# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import sys, os

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from optimizers.adamw_wrapper import AdamWWrapper
# from optimizers.lion import Lion
# from optimizers.dynamo import TargetedDynamo
# from gpu_monitor import start_gpu_monitor


# # ---------------------
# # Training utility
# # ---------------------
# def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=5):
#     # ResNet18 backbone
#     model = torchvision.models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optimizer_class(model.parameters(), lr=3e-4, weight_decay=1e-2)

#     train_losses, test_accs = [], []

#     for epoch in range(epochs):
#         # train
#         model.train()
#         running_loss = 0
#         for inputs, labels in trainloader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()

#         train_losses.append(running_loss / len(trainloader))

#         # test
#         model.eval()
#         correct, total = 0, 0
#         with torch.no_grad():
#             for inputs, labels in testloader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 _, predicted = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         test_accs.append(100 * correct / total)
#         print(f"{optimizer_name} | Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, Acc={test_accs[-1]:.2f}%")

#     return train_losses, test_accs, model


# # ---------------------
# # Spectral analysis
# # ---------------------
# def compute_singular_values(model, layer_name="fc.weight"):
#     # obtain parameter tensor and move to cpu
#     param = dict(model.named_parameters())[layer_name].detach().cpu()
#     # reshape conv kernels to 2D if needed
#     param_2d = param.reshape(param.shape[0], -1)
#     # compute singular values
#     _, S, _ = torch.linalg.svd(param_2d, full_matrices=False)
#     return S.cpu().numpy()


# # ---------------------
# # Main
# # ---------------------
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])

#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#     trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

#     testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
#     testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

#     results = {}
#     singular_spectra = {}

#     for name, opt in [
#         ("AdamW", AdamWWrapper),
#         ("Lion", Lion),
#         ("Dynamo", TargetedDynamo),
#     ]:
#         print(f"\n===== Training with {name} (ResNet18) =====")

#         gpu_thread = start_gpu_monitor(name, interval=5)

#         losses, accs, model = train_and_eval(name, opt, trainloader, testloader, device)
#         results[name] = (losses, accs)

#         # save checkpoint
#         os.makedirs("results/cifar10_resnet", exist_ok=True)
#         ckpt_path = os.path.join("results", "cifar10_resnet", f"{name.lower()}_resnet18.pth")
#         torch.save(model.state_dict(), ckpt_path)

#         # compute + print singular values
#         singular_spectra[name] = compute_singular_values(model, layer_name="fc.weight")
#         print(f"\n{name} singular values of fc.weight (top 10 shown):")
#         print(singular_spectra[name][:20])  # only first 10 for readability

#         # TODO: stop gpu monitor if you implemented stop_event
#         # gpu_thread.stop_event.set()

#     # ---------------------
#     # Plot test accuracy curves
#     # ---------------------
#     plt.figure(figsize=(10, 4))
#     for name, (losses, accs) in results.items():
#         plt.plot(accs, label=name)
#     plt.xlabel("Epoch")
#     plt.ylabel("Test Accuracy (%)")
#     plt.legend()
#     plt.title("CIFAR-10 ResNet18 Optimizer Comparison")
#     plt.savefig("results/cifar10_resnet/cifar10_resnet_compare.png")
#     plt.show()

#     # ---------------------
#     # Plot singular spectra
#     # ---------------------
#     plt.figure(figsize=(6, 4))
#     for name, spectrum in singular_spectra.items():
#         plt.plot(spectrum, label=name)
#     plt.yscale("log")
#     plt.xlabel("Index")
#     plt.ylabel("Singular value (log scale)")
#     plt.title("Singular Spectrum of fc.weight")
#     plt.legend()
#     plt.savefig("results/cifar10_resnet/singular_spectrum_fc.png")
#     plt.show()


# if __name__ == "__main__":
#     main()




import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys, os
import torch.distributed as dist

# Official Lion optimizer
from lion_pytorch import Lion

# Muon optimizer
from muon import Muon

# Your custom optimizers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optimizers.adamw_wrapper import AdamWWrapper
from optimizers.dynamo import BiostatisV2, TargetedDynamo
from optimizers.dynamo import DynamoV2, DynamoV2Adaptive, DynamoV2AdaptiveSimple, DynamoV3, BiostatisV2, DynamoGrok, BiostatisV3, BiostatisV4
from gpu_monitor import start_gpu_monitor
from torch.optim import RAdam
from optimizers.muon import SingleDeviceMuon

# ---------------------
# Muon wrapper for ResNet18
# ---------------------
def create_muon_optimizer(model, lr=0.02, momentum=0.95, weight_decay=0, **kwargs):
    """Create Muon optimizer for ResNet18 (single device)."""
    # Filter parameters: only use 2D+ parameters for Muon
    # According to the docstring, only hidden weight layers should use Muon
    muon_params = [p for p in model.parameters() if p.ndim >= 2]
    
    return SingleDeviceMuon(muon_params, lr=lr, momentum=momentum, weight_decay=weight_decay)


# ---------------------
# Training utility
# ---------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=3, **optimizer_kwargs):
    # ResNet18 backbone
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # Special handling for Muon optimizer
    if optimizer_name == "Muon":
        optimizer = create_muon_optimizer(model, **optimizer_kwargs)
    else:
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

            # Provide closure for optimizers that need it
            def closure():
                with torch.enable_grad():
                    out = model(inputs)
                    return criterion(out, labels)

            # DynamoRayleigh needs closure
            if optimizer_name == "DynamoV2":
                optimizer.step(closure=closure)
            else:
                optimizer.step()
            
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
    param = dict(model.named_parameters())[layer_name].detach().cpu()
    param_2d = param.reshape(param.shape[0], -1)
    _, S, _ = torch.linalg.svd(param_2d, full_matrices=False)
    return S.cpu().numpy()

# ---------------------
# Main
# ---------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize distributed training ONLY for Muon if needed
    # For single GPU, we can use Muon without distributed training
    # by using the simpler Muon class instead of MuonWithAuxAdam

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

    # Define optimizers with proper hyperparameters
    optimizers_config = [
        # AdamW baseline
        ("AdamW", AdamWWrapper, {
            "lr": 3e-4, 
            "weight_decay": 1e-2
        }),
    
        
        # Official Lion
        # ("Lion", Lion, {
        #     "lr": 1e-4,
        #     "weight_decay": 3e-2,
        # }),
        
        # # Muon optimizer
        # # From the paper: lr=0.02 works well for CIFAR-10
        # ("Muon", Muon, {
        #     "lr": 0.02,
        #     "momentum": 0.95,
        #     "weight_decay": 0,  # Muon handles weight decay internally
        # }),
        
        # Your Dynamo variants
        # ("Dynamo", TargetedDynamo, {
        #     "lr": 2e-4, 
        #     "weight_decay": 1e-2
        # }),
        
        # ("DynamoV2", DynamoV2, {
        #     "lr": 1e-3,
        #     "c": 0.075,
        #     "s": 3,
        #     "weight_decay": 1e-2,
        # }),
        
        # ("DynamoV2Adaptive", DynamoV2Adaptive, {
        #     "lr": 1e-3,
        #     "c": 0.075,
        #     "s": 3,
        #     "weight_decay": 1e-2,
        #     "adaptive_lr": True
        # }),
        # ("DynamoV3", DynamoV3, {
        #     "lr": 1e-3,
        #     "c": 3,
        #     "s": 3,
        #     "weight_decay": 1e-2
        # }),

        # ("DynamoGrok", DynamoGrok,{
        #     "lr": 1e-3,
        #     "c": 3,
        #     "s": 3,
        #     "weight_decay": 1e-2

        # }),

        # ("BioStatisV2", BiostatisV2,{
        #     "lr": 1e-3,
        #     "weight_decay": 1e-2,
        #     "homeo_rate": 0.5,
        #     "coherence_target": 0.8
        # }),
        
        # # ("DynamoV2AdaptiveSimple", DynamoV2AdaptiveSimple, {
        # #     "lr": 1e-3,
        # #     "c": 0.075,
        # #     "s": 3,
        # #     "weight_decay": 1e-2
        # # }),
        # ("BiostatisV3", BiostatisV3,{
        #     "lr": 1e-3,
        #     "weight_decay": 1e-2,
        #     "homeo_rate": 0.5,
        #     "coherence_target": 0.8,
        #     "energy_target": 1e-4,
        #     "lambda_energy": 0.1
        # }),
        ("BiostatisV4", BiostatisV4,{
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "homeo_rate": 0.5,
            "coherence_target": 0.8,
            "energy_target": 1e-4,
            "lambda_energy": 0.1
        }),

        ("RAdam", RAdam, {
            "lr": 2e-4, 
            "weight_decay": 1e-2,
            "decoupled_weight_decay": True
        }),
               
    ]

    for name, opt_class, opt_kwargs in optimizers_config:
        print(f"\n===== Training with {name} (ResNet18) =====")
        print(f"Hyperparameters: {opt_kwargs}")
        
        try:
            # Start GPU monitor
            gpu_thread = start_gpu_monitor(name, interval=5)

            losses, accs, model = train_and_eval(name, opt_class, trainloader, testloader, device, epochs=5, **opt_kwargs)
            results[name] = (losses, accs)

            # Save checkpoint and compute singular spectrum
            os.makedirs("results/cifar10_resnet", exist_ok=True)
            ckpt_path = os.path.join("results", "cifar10_resnet", f"{name.lower()}_resnet18.pth")
            torch.save(model.state_dict(), ckpt_path)

            singular_spectra[name] = compute_singular_values(model, layer_name="fc.weight")
            
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

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
    bars = axes[1, 1].bar(results.keys(), final_accs, alpha=0.7)
    axes[1, 1].set_ylabel("Final Test Accuracy (%)")
    axes[1, 1].set_title("Final Performance Comparison")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, final_accs):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{acc:.2f}%', ha='center', va='bottom', fontsize=8)
    
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
        
        print(f"{name:15} | Final: {final_acc:6.2f}% | Best: {best_acc:6.2f}% (Epoch {best_epoch})")
    
    print(f"{'='*60}")
    
    # Analysis of singular spectra
    print("\nSINGULAR SPECTRUM ANALYSIS:")
    for name, spectrum in singular_spectra.items():
        top_5_ratio = spectrum[:5].sum() / spectrum.sum()
        effective_rank = (spectrum.sum()**2) / (spectrum**2).sum()
        print(f"{name:15} | Top-5 concentration: {top_5_ratio:.3f} | Effective rank: {effective_rank:.2f}")


if __name__ == "__main__":
    print("Make sure you have installed:")
    print("  pip install lion-pytorch")
    print("  pip install git+https://github.com/KellerJordan/Muon")
    print()
    main()
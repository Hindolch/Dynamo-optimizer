import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers.adamw_wrapper import AdamWWrapper
from optimizers.lion import Lion
from optimizers.dynamo import TargetedDynamo
from gpu_monitor import start_gpu_monitor


# ---------------------
# Training utility
# ---------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=5):
    # ResNet18 backbone
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 has 10 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_class(model.parameters(), lr=3e-4, weight_decay=1e-2)

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
    # obtain parameter tensor and move to cpu
    param = dict(model.named_parameters())[layer_name].detach().cpu()
    # reshape conv kernels to 2D if needed
    param_2d = param.reshape(param.shape[0], -1)
    # compute singular values
    _, S, _ = torch.linalg.svd(param_2d, full_matrices=False)
    return S.cpu().numpy()


# ---------------------
# Main
# ---------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    for name, opt in [
        ("AdamW", AdamWWrapper),
        ("Lion", Lion),
        ("Dynamo", TargetedDynamo),
    ]:
        print(f"\n===== Training with {name} (ResNet18) =====")

        gpu_thread = start_gpu_monitor(name, interval=5)

        losses, accs, model = train_and_eval(name, opt, trainloader, testloader, device)
        results[name] = (losses, accs)

        # save checkpoint
        os.makedirs("results/cifar10_resnet", exist_ok=True)
        ckpt_path = os.path.join("results", "cifar10_resnet", f"{name.lower()}_resnet18.pth")
        torch.save(model.state_dict(), ckpt_path)

        # compute + print singular values
        singular_spectra[name] = compute_singular_values(model, layer_name="fc.weight")
        print(f"\n{name} singular values of fc.weight (top 10 shown):")
        print(singular_spectra[name][:20])  # only first 10 for readability

        # TODO: stop gpu monitor if you implemented stop_event
        # gpu_thread.stop_event.set()

    # ---------------------
    # Plot test accuracy curves
    # ---------------------
    plt.figure(figsize=(10, 4))
    for name, (losses, accs) in results.items():
        plt.plot(accs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.title("CIFAR-10 ResNet18 Optimizer Comparison")
    plt.savefig("results/cifar10_resnet/cifar10_resnet_compare.png")
    plt.show()

    # ---------------------
    # Plot singular spectra
    # ---------------------
    plt.figure(figsize=(6, 4))
    for name, spectrum in singular_spectra.items():
        plt.plot(spectrum, label=name)
    plt.yscale("log")
    plt.xlabel("Index")
    plt.ylabel("Singular value (log scale)")
    plt.title("Singular Spectrum of fc.weight")
    plt.legend()
    plt.savefig("results/cifar10_resnet/singular_spectrum_fc.png")
    plt.show()


if __name__ == "__main__":
    main()

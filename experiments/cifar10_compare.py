import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimizers.adamw_wrapper import AdamWWrapper
from optimizers.lion import Lion
from optimizers.dynamo import Dynamo


#---------------------
# Model: simple CNN for CIFAR10
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


#---------------------
# Training utility
# ---------------------------
def train_and_eval(optimizer_name, optimizer_class, trainloader, testloader, device, epochs=30):
    model = SimpleCNN().to(device)
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

    return train_losses, test_accs


#---------------------
# Main
# ---------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    results = {}

    for name, opt in [
        ("AdamW", AdamWWrapper),
        ("Lion", Lion),
        ("Dynamo", Dynamo),
    ]:
        print(f"\n===== Training with {name} =====")
        losses, accs = train_and_eval(name, opt, trainloader, testloader, device)
        results[name] = (losses, accs)

    # plot
    plt.figure(figsize=(10, 4))
    for name, (losses, accs) in results.items():
        plt.plot(accs, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.legend()
    plt.title("CIFAR-10 Optimizer Comparison")
    os.makedirs("results/cifar10", exist_ok=True)
    plt.savefig("results/cifar10/cifar10_compare.png")
    plt.show()


if __name__ == "__main__":
    main()

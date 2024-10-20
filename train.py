from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms

from ViTmodel import ViT
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    # Define the model
    model = ViT(in_channels=3, embedding_dim=768, num_blocks=8, nhead=8, num_classes=10, patch_size=16, num_patches=196)
    # let's use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define the transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Load CFar10 dataset
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    test = CIFAR10(root='./data', train=False, download=True, transform=preprocess)
    # Define the data loader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test, batch_size=32, shuffle=False)
    # Train the model
    for epoch in range(10):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{10}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy of the model on the test images: {100 * correct / total}%")
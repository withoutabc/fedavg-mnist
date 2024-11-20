import random

import torch
from torch.utils.data import Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def get_mnist_data():
    """ https://cloud.baidu.com/article/3331285
    """
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图片转换为Tensor
        transforms.Normalize((0.5,), (0.5,))  # 归一化，均值和标准差均为0.5
    ])

    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def split_data(dataset, num_clients):
    n = len(dataset) // num_clients
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    subsets = [Subset(dataset, indices[i * n:(i + 1) * n]) for i in range(num_clients)]
    return subsets


def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy of the network on the test images: {accuracy:.2f}%')
    return accuracy
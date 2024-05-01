from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class CCNNetwork(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 16*5*5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.output(x), dim=1)


def load_data():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(
        root='/Users/mikaelhellman/dev/ml-lab/data/mnist', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(
        root='/Users/mikaelhellman/dev/ml-lab/data/mnist', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

    return train_loader, test_loader


def create_cnn():
    conv1 = nn.Conv2d(1, 6, 3, 1)
    conv2 = nn.Conv2d(6, 16, 3, 1)

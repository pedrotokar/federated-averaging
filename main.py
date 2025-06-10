import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import MNIST_CNN
from utils import evaluate
from client import client_update

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

LEARNING_RATE = 0.01

global_model = MNIST_CNN()

optimizer = torch.optim.SGD(global_model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.NLLLoss()

test_loader = DataLoader(
    test_data
)

client_update(
    0,
    global_model,
    train_data,
    1,
    128,
    LEARNING_RATE
)

print(evaluate(global_model, test_loader))

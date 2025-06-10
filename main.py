import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import MNIST_CNN
from utils import evaluate

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

data_loader = DataLoader(
    Subset(train_data, range(1000)),
    batch_size=128,
    shuffle=True
)

for epoch in range(10):
    for idx, (features, target) in enumerate(data_loader):
        optimizer.zero_grad()
        out = global_model(features)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()

        print(loss)

print(evaluate(global_model, data_loader))

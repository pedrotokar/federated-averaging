import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import copy
import numpy as np

from models import MNIST_CNN
from utils import evaluate, distribute_data
from client import client_update


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)


NUM_CLIENTS     = 100   # K

CLIENT_FRAC     = 0.1   # C
NUM_PASSES      = 1     # E
BATCH_SIZE      = 128   # B

LEARNING_RATE   = 0.01  # n
NUM_ROUNDS      = 1

global_model = MNIST_CNN()

test_loader = DataLoader(
    test_data
)

clients_data = distribute_data(train_data, NUM_CLIENTS, iid=True)

for comm_round in range(NUM_ROUNDS):
    m = max(int(CLIENT_FRAC * NUM_CLIENTS), 1)
    clients = np.random.choice(range(NUM_CLIENTS), m, replace=False)

    print(clients)

    for client_id in clients:

        local_model = copy.deepcopy(global_model)

        client_update(
            client_id,
            local_model,
            clients_data.get(client_id),
            epochs=NUM_PASSES,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE
        )


print(evaluate(global_model, test_loader))

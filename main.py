import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import copy
import numpy as np

from models import MNIST_CNN
from utils import load_MNIST, evaluate, distribute_data
from client import client_update
from server import server_aggregate


NUM_CLIENTS     = 100   # K

CLIENT_FRAC     = 0.75   # C
NUM_PASSES      = 1     # E
BATCH_SIZE      = 128   # B

LEARNING_RATE   = 0.01  # n
NUM_ROUNDS      = 100
IID             = True

#device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando", device)

train_data, test_data = load_MNIST()

global_model = MNIST_CNN().to(device)

test_loader = DataLoader(
    test_data,
    shuffle=False,
    batch_size=128,
    num_workers=10
)


clients_data, clients_lens = distribute_data(train_data, NUM_CLIENTS, iid=IID)
loss_hist = []

for comm_round in range(NUM_ROUNDS):
    m = max(int(CLIENT_FRAC * NUM_CLIENTS), 1)
    clients = np.random.choice(range(NUM_CLIENTS), m, replace=False)

    print(clients)

    selected_clients_states = list()
    selected_clients_lens = list()

    for client_id in clients:

        local_model = copy.deepcopy(global_model)

        train_subset = clients_data.get(client_id)

        state = client_update(
            client_id,
            local_model,
            train_subset,
            epochs=NUM_PASSES,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=device
        )

        selected_clients_lens.append(clients_lens.get(client_id))
        selected_clients_states.append(state)

    new_global_state = server_aggregate(
        global_model.state_dict(),
        selected_clients_states,
        selected_clients_lens
    )

    global_model.load_state_dict(new_global_state)

    loss, acc = evaluate(global_model, test_loader, device)
    loss_hist.append((loss, acc))
    print(f"Round {comm_round + 1} | Loss: {loss:.4f} | Acc: {acc:.2f}%")

with open(f"results/mnist_{'iid' if IID else 'noniid'}_N{NUM_CLIENTS}_C{CLIENT_FRAC}_E{NUM_PASSES}_B{BATCH_SIZE}.txt", "w") as f:
    f.writelines([f"{loss[0]},{loss[1]}\n" for loss in loss_hist])

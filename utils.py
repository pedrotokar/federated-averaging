import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import numpy as np

def load_MNIST():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST("./data", train=False, download=True, transform=transform)

    return train_data, test_data

def evaluate(model, test_loader, device):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            test_loss += nn.NLLLoss(reduction="sum")(output, target.to(device)).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred).to(device)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)

    return test_loss, accuracy

def distribute_data(dataset, num_clients, iid=True):
    client_dict = dict()
    client_lens = dict()
    
    if iid:
        num_items = int(len(dataset) / num_clients)
        idxs = list(range(len(dataset)))
        np.random.shuffle(idxs)

        for i in range(num_clients):
            start_idx = i * num_items
            end_idx = start_idx + num_items
            client_indices = idxs[start_idx:end_idx]
            client_dict[i] = Subset(dataset, client_indices)
            client_lens[i] = len(client_indices)
            
    return client_dict, client_lens

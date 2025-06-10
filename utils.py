import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import numpy as np

def evaluate(model, test_loader):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += nn.NLLLoss(reduction="sum")(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / len(test_loader.dataset)

    return test_loss, accuracy

def distribute_data(dataset, num_clients, iid=True):
    client_dict = {}
    
    if iid:
        num_items = int(len(dataset) / num_clients)
        idxs = list(range(len(dataset)))
        np.random.shuffle(idxs)

        for i in range(num_clients):
            start_idx = i * num_items
            end_idx = start_idx + num_items
            client_indices = idxs[start_idx:end_idx]
            client_dict[i] = Subset(dataset, client_indices)
            
    return client_dict

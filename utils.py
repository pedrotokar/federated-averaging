import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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
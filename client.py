import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from models import MNIST_CNN

def client_update(client_id, model, train_subset, epochs, batch_size, learning_rate):

    print(f"  > Cliente {client_id}")
    
    if batch_size == float("inf"):
        batch_size = len(train_subset)

    data_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    for epoch in range(epochs):
        for idx, (features, target) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(features)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

    return model.state_dict()

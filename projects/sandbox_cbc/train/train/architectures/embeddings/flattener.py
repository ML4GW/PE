import torch


class Flattener(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = X.reshape(len(X), -1)
        return X

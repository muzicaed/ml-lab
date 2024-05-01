from torch import nn
import torch.nn.functional as F


class Model(nn.Module):

    def __init__(self, in_features=4, h1=8, h2=8, out_features=3):
        super().__init__()
        self.input = nn.Linear(in_features, h1)
        self.hidden = nn.Linear(h1, h2)
        self.output = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden(x))
        return self.output(x)

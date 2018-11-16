import torch.nn as nn
import torch.nn.functional as F

nclasses = 20


class LinearClassifier(nn.Module):
    def __init__(self, in_features):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        return F.softmax(self.fc2(F.relu(self.fc1(x))), dim=-1)

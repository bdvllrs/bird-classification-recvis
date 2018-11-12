import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet as torch_alexnet, resnet101 as torch_resnet101

nclasses = 20


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def simple_cnn():
    return SimpleCNN(), (64, 64)


def alexnet():
    model = torch_alexnet(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, nclasses),
    )
    return model, (227, 227)


def resnet101():
    model = torch_resnet101(pretrained=True)
    model_conv = nn.Sequential(*list(model.children())[:-3])
    for param in model_conv.parameters():
        param.requires_grad = False
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, nclasses)
    return model, (224, 224)


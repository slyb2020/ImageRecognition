import torch
import torchvision
from torch import nn
from MNIST_Dataset import *

class LeeNet(nn.Module):
    def __init__(self, inputChannel=3, padding=0):
        super(LeeNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inputChannel, inputChannel, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(inputChannel, eps=1e-5, momentum=0.1, affine=True),
            nn.Conv2d(inputChannel, 6, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm2d(6, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(6, eps=1e-5, momentum=0.1, affine=True),
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(120, eps=1e-5, momentum=0.1, affine=True),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 168),
            nn.BatchNorm1d(168),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(168, 10),
            nn.BatchNorm1d(10),
            # nn.Softmax(1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x


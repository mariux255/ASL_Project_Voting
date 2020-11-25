import tqdm
import numpy as np
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.pool = nn.MaxPool2d(4, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(128, 256, 3)
        self.conv3 = nn.Conv2d(256, 512, 3)
        self.conv4 = nn.Conv2d(512, 256, 3)
        self.fc3 = nn.Linear(2304, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 2304)
        x = self.fc3(x)
        return x
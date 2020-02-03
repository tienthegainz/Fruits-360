import torch
from torch import nn


class FruitNet(nn.Module):
    def __init__(self, class_num):
        super(FruitNet, self).__init__()
        # input size: 100x100x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # output: 48, 48, 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # output: 50, 50, 16
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(5, 5))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(5, 5))
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # m1 = [batch_size x in_features] || m2 = [in_features x out_features]
        # input feature: C x H x W
        self.fc1 = nn.Linear(128*2*2, 256)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(256, class_num)
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        # We need to reshape it into [batch_size x tensor]
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        x = self.sfmx(x)
        return x


if __name__ == "__main__":
    input = torch.ones([2, 3, 100, 100], dtype=torch.float32)
    model = FruitNet(2)
    print(model(input).shape)

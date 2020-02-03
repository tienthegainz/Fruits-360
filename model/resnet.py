import torch
from torch import nn
import torchvision.models as models
from torchviz import make_dot


class ResFruitNet(nn.Module):
    def __init__(self, num_classes):
        super(ResFruitNet, self).__init__()
        pretrained = models.resnet18(pretrained=True)
        modules = list(pretrained.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(nn.Linear(256*7*7, 1024),
                                nn.Linear(1024, 256),
                                nn.Linear(256, num_classes)
                                )

    def forward(self, input):
        x = self.resnet(input)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def train(self):
        for param in self.fc.parameters():
            param.requires_grad = True

    def eval(self):
        for param in self.fc.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    input = torch.ones([2, 3, 100, 100], dtype=torch.float32)
    model = ResFruitNet(20)
    print(model(input).shape)
    model.train()

import torch
from torch import nn
import torchvision.models as models
# from torchviz import make_dot


class ResFruitNet(nn.Module):
    def __init__(self, num_classes, freeze=True):
        """
            num_classes: number of class in dataset
            freeze: freeze pretrained resnet or not
        """
        super(ResFruitNet, self).__init__()
        self.name = "ResFruitNet"
        pretrained = models.resnet18(pretrained=True)
        modules = list(pretrained.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.freeze = freeze
        self.is_training = True
        if self.freeze == True:
            # Not freeze BatchNorm
            for param in self.resnet.parameters():
                param.requires_grad = False
            # Freeze batchnorm
            self.freeze_bn()

        self.drop1 = nn.Dropout2d(p=0.5)
        # self.bottleneck = nn.Sequential(
        #     nn.Conv2d(256, 124, kernel_size=(1, 1)),
        #     nn.ReLU(),
        #     nn.Conv2d(124, 124, kernel_size=(3, 3)),
        #     nn.ReLU(),
        #     nn.Conv2d(124, 256, kernel_size=(3, 3)),
        #     nn.ReLU(),
        # )
        self.inception_br_3x3_2 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=(1, 1)),
            nn.Conv2d(32, 96, kernel_size=(3, 3), padding=1),
            # nn.Conv2d(128, 256, kernel_size=(3, 3)),
        )
        self.inception_br_3x3_1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1)),
            nn.Conv2d(128, 192, kernel_size=(3, 3), padding=1),
        )
        self.inception_br_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3, 3), stride=1,
                         padding=1, ceil_mode=True),
            nn.Conv2d(256, 64, kernel_size=(1, 1))
        )
        self.inception_br_1x1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(1, 1))
        )
        # self.fc = nn.Sequential(nn.Linear(256*12*12, 1024),
        #                         nn.BatchNorm1d(1024),
        #                         nn.Linear(1024, 256),
        #                         nn.BatchNorm1d(256),
        #                         nn.Linear(256, num_classes)
        #                         )
        self.fc = nn.Sequential(nn.Linear(480*16*16, 1024),
                                nn.BatchNorm1d(1024),
                                nn.Linear(1024, 256),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, num_classes)
                                )
        self.sfmx = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.resnet(input)
        print(x.shape)
        x = self.drop1(x)
        a = self.inception_br_3x3_1(x)
        b = self.inception_br_3x3_2(x)
        c = self.inception_br_pool(x)
        d = self.inception_br_1x1(x)
        # print('3x3_1: {}\n3x3_1: {}\npool: {}\nbr1x1: {}'.format(
        #     a.shape, b.shape, c.shape, d.shape))
        x = [a, b, c, d]
        x = torch.cat(x, 1)

        # x = self.bottleneck(x)
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        outputs = self.sfmx(x)
        return outputs

    def train(self):
        self.is_training = True
        if self.freeze == False:
            for param in self.resnet.parameters():
                param.requires_grad = True
        for param in self.fc.parameters():
            param.requires_grad = True

    def eval(self):
        self.is_training = False
        if self.freeze == False:
            for param in self.resnet.parameters():
                param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.resnet.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == "__main__":
    input = torch.ones([2, 3, 256, 256], dtype=torch.float32)
    model = ResFruitNet(20)
    print(model(input).shape)

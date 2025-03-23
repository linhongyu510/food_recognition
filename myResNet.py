import torch
import torch.nn as nn
import torchvision.models as models


resNet = models.resnet18()

print(resNet)


class Residual_block(nn.Module):  #@save
    def __init__(self, input_channels, out_channels, down_sample=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, stride= 1)
        if input_channels != out_channels:
            self.conv3 = nn.Conv2d(input_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    def forward(self, X):
        out = self.relu(self.bn1(self.conv1(X)))
        out= self.bn2(self.conv2(out))

        if self.conv3:
            X = self.conv3(X)
        out += X
        return self.relu(out)

class MyResNet18(nn.Module):
    def __init__(self):
        super(MyResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU()

        self.layer1 = nn.Sequential(
            Residual_block(64, 64),
            Residual_block(64, 64)
        )
        self.layer2 = nn.Sequential(
            Residual_block(64, 128, strides=2),
            Residual_block(128, 128)
        )
        self.layer3 = nn.Sequential(
            Residual_block(128, 256, strides=2),
            Residual_block(256, 256)
        )
        self.layer4 = nn.Sequential(
            Residual_block(256, 512, strides=2),
            Residual_block(512, 512)
        )
        self.flatten = nn.Flatten()
        self.adv_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 1000)
    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.adv_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

myres = MyResNet18()

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


print(get_parameter_number(myres.layer1))
print(get_parameter_number(myres.layer1[0].conv1))
print(get_parameter_number(resNet.layer1[0].conv1))

x = torch.rand((1,3,224,224))
out = resNet(x)

out = myres(x)

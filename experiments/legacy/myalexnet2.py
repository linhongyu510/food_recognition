import torchvision.models as models


# mymodel = models.alexnet()
#
# print(mymodel)
import torch.nn as nn

class myAlexNet(nn.Module):
    def __init__(self, out_dim):
        super(myAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3,64,11,4,2)
        self.pool1 = nn.MaxPool2d(3, 2)

        self.conv2 = nn.Conv2d(64,192,5,1,2)
        self.pool2 = nn.MaxPool2d(3, 2)

        self.conv3 = nn.Conv2d(192,384,3,1,1)
        self.conv4 = nn.Conv2d(384, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)

        self.pool3 = nn.MaxPool2d(3, 2)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, out_dim)

    def forward(self,x):
        x =self.conv1(x)
        x = self.pool1(x)
        x =self.conv2(x)
        x = self.pool2(x)
        x =self.conv3(x)
        x =self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.pool4(x)

        x = x.view(x.size()[0], -1)  #拉直。 batch
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = myAlexNet(1000)

import torch
# a = torch.ones((4,3,224,224))   #batch， 3, 224,224
# pred = model(a)

def get_parameter_number(model):    #传入模型 获取参数量
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

print(get_parameter_number(model.pool1))
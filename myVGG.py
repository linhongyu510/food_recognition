import torchvision.models as models
import torch.nn as nn


vgg = models.vgg13()
print(vgg)

class vggLayer(nn.Module):
    def __init__(self,in_cha, mid_cha, out_cha):
        super(vggLayer, self).__init__()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(in_cha, mid_cha, 3, 1, 1)
        self.conv2 = nn.Conv2d(mid_cha, out_cha, 3, 1, 1)

    def forward(self,x):
        x = self.conv1(x)
        x= self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        return x



class MyVgg(nn.Module):
    def __init__(self):
        super(MyVgg, self).__init__()

        self.layer1 = vggLayer(3, 64, 64)

        self.layer2 = vggLayer(64, 128, 128)

        self.layer3 = vggLayer(128, 256, 256)

        self.layer4 = vggLayer(256, 512, 512)

        self.layer5 = vggLayer(512, 512, 512)
        self.adapool = nn.AdaptiveAvgPool2d(7)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.adapool(x)

        x= self.adapool(x)
        x = x.view(x.size()[0], -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        return x


import torch

myVgg = MyVgg()
#
img = torch.zeros((1, 3, 224,224))

out = myVgg(img)

print(out.size())

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


print(get_parameter_number(myVgg))
print(get_parameter_number(myVgg.layer1))
print(get_parameter_number(myVgg.layer1.conv1))
#
# print(get_parameter_number(vgg))
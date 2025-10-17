"""
AlexNet模型实现

包含标准AlexNet和自定义AlexNet实现
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MyAlexNet(nn.Module):
    """自定义AlexNet实现"""
    
    def __init__(self, num_classes=11):
        super(MyAlexNet, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 64, 11, 4, 2)
        self.pool1 = nn.MaxPool2d(3, 2)
        
        self.conv2 = nn.Conv2d(64, 192, 5, 1, 2)
        self.pool2 = nn.MaxPool2d(3, 2)
        
        self.conv3 = nn.Conv2d(192, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)
        
        self.pool3 = nn.MaxPool2d(3, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(6)
        
        # 全连接层
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def AlexNet(num_classes=11, pretrained=True):
    """标准AlexNet模型"""
    model = models.alexnet(pretrained=pretrained)
    if pretrained:
        # 冻结特征提取层
        for param in model.parameters():
            param.requires_grad = False
    # 替换分类头
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def get_parameter_number(model):
    """获取模型参数量"""
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}




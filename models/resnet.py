"""
ResNet模型实现

包含ResNet18、ResNet50和自定义ResNet实现
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResidualBlock(nn.Module):
    """残差块实现"""
    
    def __init__(self, input_channels, out_channels, down_sample=False, strides=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, stride=1)
        
        # 如果输入输出通道数不同，需要1x1卷积进行维度匹配
        if input_channels != out_channels:
            self.conv3 = nn.Conv2d(input_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
            
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # 残差连接
        if self.conv3:
            x = self.conv3(x)
        out += x
        return self.relu(out)


class MyResNet18(nn.Module):
    """自定义ResNet18实现"""
    
    def __init__(self, num_classes=11):
        super(MyResNet18, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU()
        
        # 残差层
        self.layer1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, strides=2),
            ResidualBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, strides=2),
            ResidualBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, strides=2),
            ResidualBlock(512, 512)
        )
        
        # 分类头
        self.flatten = nn.Flatten()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        
        return x


def ResNet18(num_classes=11, pretrained=True):
    """ResNet18模型"""
    model = models.resnet18(pretrained=pretrained)
    if pretrained:
        # 冻结特征提取层
        for param in model.parameters():
            param.requires_grad = False
    # 替换分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ResNet50(num_classes=11, pretrained=True):
    """ResNet50模型"""
    model = models.resnet50(pretrained=pretrained)
    if pretrained:
        # 冻结特征提取层
        for param in model.parameters():
            param.requires_grad = False
    # 替换分类头
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_parameter_number(model):
    """获取模型参数量"""
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}




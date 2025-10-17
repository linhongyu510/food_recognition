"""
EfficientNet + CBAM注意力机制模型

结合EfficientNet和CBAM注意力模块的食物识别模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class CBAM(nn.Module):
    """卷积块注意力模块 (Convolutional Block Attention Module)"""
    
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        
        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        x_channel = self.channel_att(x) * x
        
        # 空间注意力
        x_spatial = torch.cat([
            torch.mean(x_channel, dim=1, keepdim=True),
            torch.max(x_channel, dim=1, keepdim=True)[0]
        ], dim=1)
        x_spatial = self.spatial_att(x_spatial) * x_channel
        
        return x_spatial


class EfficientNet_CBAM(nn.Module):
    """EfficientNet + CBAM模型"""
    
    def __init__(self, num_classes=11, pretrained=True):
        super(EfficientNet_CBAM, self).__init__()
        
        # 加载预训练的EfficientNet-B0
        if pretrained:
            self.backbone = models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.efficientnet_b0(weights=None)
            
        # CBAM注意力模块 (EfficientNet-B0最后特征通道数是1280)
        self.cbam = CBAM(1280)
        
        # 分类头
        self.fc = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        # 获取EfficientNet特征
        x = self.backbone.features(x)
        
        # 应用CBAM注意力
        x = self.cbam(x)
        
        # 全局平均池化和分类
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def get_parameter_number(model):
    """获取模型参数量"""
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}




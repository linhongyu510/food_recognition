"""
损失函数实现

包含各种损失函数的定义
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss实现，用于处理类别不平衡问题"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        初始化Focal Loss
        
        Args:
            alpha: 权重参数
            gamma: 聚焦参数
            reduction: 损失缩减方式
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失函数"""
    
    def __init__(self, num_classes, smoothing=0.1):
        """
        初始化标签平滑损失
        
        Args:
            num_classes: 类别数
            smoothing: 平滑参数
        """
        super(LabelSmoothingLoss, self).__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        计算标签平滑损失
        
        Args:
            pred: 预测结果
            target: 真实标签
        
        Returns:
            损失值
        """
        log_pred = F.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * log_pred, dim=1))


class DiceLoss(nn.Module):
    """Dice Loss，用于分割任务"""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class ContrastiveLoss(nn.Module):
    """对比损失，用于度量学习"""
    
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output1, output2, label):
        """
        计算对比损失
        
        Args:
            output1: 第一个样本的特征
            output2: 第二个样本的特征
            label: 标签（1表示相似，0表示不相似）
        """
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive




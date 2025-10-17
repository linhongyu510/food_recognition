"""
评估指标实现

包含各种评估指标的计算
"""

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class Accuracy:
    """准确率计算"""
    
    def __init__(self):
        self.reset()
    
    def update(self, preds, targets):
        """更新预测结果"""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """计算准确率"""
        preds = np.array(self.predictions)
        targets = np.array(self.targets)
        return (preds == targets).mean()
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []


class Precision:
    """精确率计算"""
    
    def __init__(self, average='macro'):
        self.average = average
        self.reset()
    
    def update(self, preds, targets):
        """更新预测结果"""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """计算精确率"""
        return precision_score(self.targets, self.predictions, average=self.average)
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []


class Recall:
    """召回率计算"""
    
    def __init__(self, average='macro'):
        self.average = average
        self.reset()
    
    def update(self, preds, targets):
        """更新预测结果"""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """计算召回率"""
        return recall_score(self.targets, self.predictions, average=self.average)
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []


class F1Score:
    """F1分数计算"""
    
    def __init__(self, average='macro'):
        self.average = average
        self.reset()
    
    def update(self, preds, targets):
        """更新预测结果"""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """计算F1分数"""
        return f1_score(self.targets, self.predictions, average=self.average)
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []


class ConfusionMatrix:
    """混淆矩阵"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.reset()
    
    def update(self, preds, targets):
        """更新预测结果"""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """计算混淆矩阵"""
        return confusion_matrix(self.targets, self.predictions)
    
    def plot(self, figsize=(10, 8)):
        """绘制混淆矩阵"""
        cm = self.compute()
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []


class ClassificationReport:
    """分类报告"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names
        self.reset()
    
    def update(self, preds, targets):
        """更新预测结果"""
        self.predictions.extend(preds.cpu().numpy())
        self.targets.extend(targets.cpu().numpy())
    
    def compute(self):
        """计算分类报告"""
        from sklearn.metrics import classification_report
        return classification_report(
            self.targets, 
            self.predictions, 
            target_names=self.class_names
        )
    
    def reset(self):
        """重置"""
        self.predictions = []
        self.targets = []




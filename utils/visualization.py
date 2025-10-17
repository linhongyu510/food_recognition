"""
可视化工具

包含Grad-CAM、预测可视化等功能
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class GradCAM:
    """Grad-CAM可视化类"""
    
    def __init__(self, model, target_layer):
        """
        初始化Grad-CAM
        
        Args:
            model: 模型
            target_layer: 目标层
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """保存激活值"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """保存梯度"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        生成CAM图
        
        Args:
            input_tensor: 输入张量
            class_idx: 类别索引
        
        Returns:
            CAM图
        """
        # 前向传播
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3))
        
        # 生成CAM
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
        
        # 应用ReLU
        cam = F.relu(cam)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def visualize(self, input_tensor, class_names=None, save_path=None):
        """
        可视化Grad-CAM
        
        Args:
            input_tensor: 输入张量
            class_names: 类别名称
            save_path: 保存路径
        """
        # 生成CAM
        cam = self.generate_cam(input_tensor)
        
        # 获取原始图像
        img = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        # 调整CAM大小
        cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
        
        # 创建热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        
        # 叠加图像
        superimposed_img = heatmap * 0.4 + img * 0.6
        
        # 显示结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(superimposed_img)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def visualize_predictions(model, dataloader, device, num_samples=9, class_names=None):
    """
    可视化预测结果
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        num_samples: 样本数量
        class_names: 类别名称
    """
    if class_names is None:
        class_names = [
            'Bread', 'Dairy', 'Dessert', 'Egg', 'Fried', 
            'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Vegetable'
        ]
    
    model.eval()
    
    # 获取样本
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        probabilities = F.softmax(outputs, dim=1)
    
    # 可视化
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # 原始图像
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        # 标题
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predictions[i].item()]
        confidence = probabilities[i][predictions[i]].item()
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                         color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    绘制训练历史
    
    Args:
        train_losses: 训练损失
        val_losses: 验证损失
        train_accs: 训练准确率
        val_accs: 验证准确率
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()




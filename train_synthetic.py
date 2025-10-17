"""
使用合成数据进行食物识别训练

基于毕设.md中的EfficientNet + CBAM模型
"""

import random
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms, models
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 固定随机种子函数
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(0)

# CBAM模块（卷积块注意力模块）
class CBAM(nn.Module):
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
        x_spatial = torch.cat([torch.mean(x_channel, dim=1, keepdim=True),
                               torch.max(x_channel, dim=1, keepdim=True)[0]], dim=1)
        x_spatial = self.spatial_att(x_spatial) * x_channel
        return x_spatial


# EfficientNet-B0 + CBAM模型
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_CBAM, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cbam = CBAM(1280)
        self.fc = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)
        return x


# 合成数据集类
class SyntheticFoodDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=11, image_size=224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # 生成合成数据
        self.images = []
        self.labels = []
        
        for _ in range(num_samples):
            # 生成随机图像数据
            image = torch.randn(3, image_size, image_size)
            label = random.randint(0, num_classes - 1)
            
            # 为不同类别添加一些特征
            if label == 0:  # Bread - 添加一些纹理
                image[0] += torch.randn(image_size, image_size) * 0.1
            elif label == 1:  # Dairy - 添加一些白色区域
                image += 0.3
            elif label == 2:  # Dessert - 添加一些彩色
                image[1] += torch.randn(image_size, image_size) * 0.2
            # ... 其他类别类似
            
            self.images.append(image)
            self.labels.append(label)
        
        self.images = torch.stack(self.images)
        self.labels = torch.LongTensor(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
    def __len__(self):
        return self.num_samples


# 训练与验证函数
def train_model(model, train_loader, val_loader, device, epochs, optimizer, loss_fn, save_path):
    """训练模型"""
    model.to(device)
    best_acc = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_acc = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 训练"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        
        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, device, loss_fn)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return train_losses, train_accuracies, val_losses, val_accuracies


# 评估函数
def evaluate_model(model, val_loader, device, loss_fn):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            total_loss += loss.item()
            pred = outputs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy


# 计算详细指标
def calculate_metrics(model, test_loader, device, class_names):
    """计算详细评估指标"""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    # 各类别指标
    class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_mask = np.array(all_targets) == i
        if np.sum(class_mask) > 0:
            class_preds = np.array(all_preds)[class_mask]
            class_targets = np.array(all_targets)[class_mask]
            class_acc = accuracy_score(class_targets, class_preds)
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                class_targets, class_preds, average='weighted', zero_division=0
            )
            class_metrics[class_name] = {
                'accuracy': class_acc,
                'precision': class_precision,
                'recall': class_recall,
                'f1': class_f1
            }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics,
        'predictions': all_preds,
        'targets': all_targets
    }


# 绘制训练曲线
def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies, save_path):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    axes[0].plot(train_losses, label='训练损失', color='blue')
    axes[0].plot(val_losses, label='验证损失', color='red')
    axes[0].set_title('训练和验证损失')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(train_accuracies, label='训练准确率', color='blue')
    axes[1].plot(val_accuracies, label='验证准确率', color='red')
    axes[1].set_title('训练和验证准确率')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 绘制各类别性能
def plot_class_performance(class_metrics, save_path):
    """绘制各类别性能"""
    classes = list(class_metrics.keys())
    precision = [class_metrics[cls]['precision'] for cls in classes]
    recall = [class_metrics[cls]['recall'] for cls in classes]
    f1 = [class_metrics[cls]['f1'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(x - width, precision, width, label='精确率', alpha=0.8)
    ax.bar(x, recall, width, label='召回率', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1分数', alpha=0.8)
    
    ax.set_xlabel('类别')
    ax.set_ylabel('分数')
    ax.set_title('各类别性能指标')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=== 基于毕设模型的食物识别训练（合成数据） ===")
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 创建合成数据集
    print("创建合成数据集...")
    train_dataset = SyntheticFoodDataset(num_samples=1000, num_classes=11)
    val_dataset = SyntheticFoodDataset(num_samples=200, num_classes=11)
    test_dataset = SyntheticFoodDataset(num_samples=200, num_classes=11)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = EfficientNet_CBAM(num_classes=11)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # 训练模型
    print("\n开始训练...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, device, 15, optimizer, loss_fn, 
        str(results_dir / "best_model.pth")
    )
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(results_dir / "best_model.pth"))
    
    # 计算详细指标
    class_names = [
        'Bread', 'Dairy', 'Dessert', 'Egg', 'Fried',
        'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Vegetable'
    ]
    
    print("\n计算详细指标...")
    metrics = calculate_metrics(model, test_loader, device, class_names)
    
    # 打印结果
    print("\n=== 训练结果 ===")
    print(f"测试准确率: {metrics['accuracy']:.4f}")
    print(f"测试精确率: {metrics['precision']:.4f}")
    print(f"测试召回率: {metrics['recall']:.4f}")
    print(f"测试F1分数: {metrics['f1']:.4f}")
    
    # 各类别性能
    print("\n各类别性能:")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"{class_name}: 准确率={class_metrics['accuracy']:.3f}, "
              f"精确率={class_metrics['precision']:.3f}, "
              f"召回率={class_metrics['recall']:.3f}, "
              f"F1={class_metrics['f1']:.3f}")
    
    # 保存结果
    results = {
        'model': 'EfficientNet_CBAM',
        'epochs': 15,
        'test_metrics': {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        },
        'class_metrics': metrics['class_metrics'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 绘制图表
    print("\n生成可视化图表...")
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies,
                        results_dir / 'training_curves.png')
    plot_confusion_matrix(metrics['targets'], metrics['predictions'], class_names,
                         results_dir / 'confusion_matrix.png')
    plot_class_performance(metrics['class_metrics'], results_dir / 'class_performance.png')
    
    print(f"\n所有结果已保存到: {results_dir}")
    print("训练完成!")


if __name__ == "__main__":
    main()



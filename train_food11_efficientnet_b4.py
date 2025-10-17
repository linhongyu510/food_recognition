"""
使用Food-11数据集和EfficientNet-B4 + 注意力模块进行训练

基于毕设.md中的模型架构，使用真实数据集进行训练
"""

import random
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
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

HW = 224  # 定义图片尺寸为224x224

# 训练数据增强操作
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(HW),
    transforms.RandomRotation(50),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证数据增强操作
val_transform = transforms.Compose([
    transforms.Resize((HW, HW)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 自定义数据集类
class FoodDataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        if mode == "semi":
            self.image_paths = [os.path.join(path, img) for img in os.listdir(path)]
        else:
            for label in range(11):
                folder = os.path.join(path, f"{label:02d}")
                if os.path.exists(folder):
                    for img in os.listdir(folder):
                        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.image_paths.append(os.path.join(folder, img))
                            self.labels.append(label)
            self.labels = torch.LongTensor(self.labels)
        
        self.transform = train_transform if mode == "train" else val_transform
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        
        if self.mode == "semi":
            return image, self.image_paths[idx]
        else:
            return image, self.labels[idx]
    
    def __len__(self):
        return len(self.image_paths)


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


# EfficientNet-B4 + CBAM模型
class EfficientNet_B4_CBAM(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_B4_CBAM, self).__init__()
        # 使用EfficientNet-B4作为backbone
        self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        # EfficientNet-B4最后特征通道数是1792
        self.cbam = CBAM(1792)
        self.fc = nn.Linear(1792, num_classes)
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        x = self.fc(x)
        return x


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
            print(f"新的最佳模型! 验证准确率: {best_acc:.4f}")
        
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
    print("=== 使用Food-11数据集和EfficientNet-B4 + CBAM进行训练 ===")
    
    # 创建结果目录
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 检查数据集是否存在
    data_path = "data/food11"
    if not os.path.exists(data_path):
        print(f"数据集路径不存在: {data_path}")
        print("请确保Food-11数据集已正确放置在data/food11目录下")
        return
    
    # 创建数据集
    print("加载数据集...")
    train_set = FoodDataset(os.path.join(data_path, "training/labeled"), "train")
    val_set = FoodDataset(os.path.join(data_path, "validation"), "val")
    
    print(f"训练集: {len(train_set)} 样本")
    print(f"验证集: {len(val_set)} 样本")
    
    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = EfficientNet_B4_CBAM(num_classes=11)
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
        model, train_loader, val_loader, device, 30, optimizer, loss_fn, 
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
    metrics = calculate_metrics(model, val_loader, device, class_names)
    
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
        'model': 'EfficientNet_B4_CBAM',
        'epochs': 30,
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

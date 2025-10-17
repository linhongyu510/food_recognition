"""
优化的食物识别训练脚本 - 目标准确率90%+

主要改进：
1. 更好的数据增强策略
2. 优化的学习率调度
3. 早停机制
4. 更好的模型架构
5. 梯度裁剪
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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

# 导入混合精度训练
from torch.cuda.amp import autocast, GradScaler

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

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

seed_everything(42)  # 使用不同的随机种子

HW = 224  # 定义图片尺寸为224x224

# 改进的数据增强策略
train_transform = transforms.Compose([
    transforms.Resize(256),  # 先放大到256
    transforms.RandomResizedCrop(HW, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),  # 减少旋转角度
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证数据增强操作
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(HW),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 类别名称映射
class_names = [
    'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food',
    'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit'
]

# 自定义数据集类
class FoodDataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        if mode == "train":
            # 训练模式：从各个类别文件夹加载数据
            for label, class_name in enumerate(class_names):
                class_path = os.path.join(path, class_name)
                if os.path.exists(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.image_paths.append(os.path.join(class_path, img_file))
                            self.labels.append(label)
        elif mode == "val":
            # 验证模式：从evaluation文件夹加载数据，使用类别名称
            for label, class_name in enumerate(class_names):
                class_path = os.path.join(path, class_name)
                if os.path.exists(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.image_paths.append(os.path.join(class_path, img_file))
                            self.labels.append(label)
        
        self.labels = torch.LongTensor(self.labels)
        self.transform = train_transform if mode == "train" else val_transform
        
        print(f"加载了 {len(self.image_paths)} 个样本")
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
            image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            print(f"加载图片失败: {self.image_paths[idx]}, 错误: {e}")
            # 返回一个默认的黑色图片
            image = torch.zeros(3, HW, HW)
            return image, self.labels[idx]
    
    def __len__(self):
        return len(self.image_paths)


# 改进的CBAM模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
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


# 改进的ResNet50 + CBAM模型
class ResNet50_CBAM_Improved(nn.Module):
    def __init__(self, num_classes=11):
        super(ResNet50_CBAM_Improved, self).__init__()
        # 使用ResNet50作为backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 冻结早期层，只训练后面的层
        for param in self.backbone.conv1.parameters():
            param.requires_grad = False
        for param in self.backbone.bn1.parameters():
            param.requires_grad = False
        for param in self.backbone.layer1.parameters():
            param.requires_grad = False
        
        # ResNet50最后特征通道数是2048
        self.cbam = CBAM(2048)
        
        # 添加dropout和更好的分类头
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 替换原来的fc层
        self.backbone.fc = nn.Identity()
    
    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # 应用CBAM注意力
        x = self.cbam(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 早停类
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


# 改进的训练函数
def train_model_improved(model, train_loader, val_loader, device, epochs, optimizer, loss_fn, save_path):
    """改进的训练模型"""
    model.to(device)
    best_acc = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # 混合精度训练
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    # 学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 早停
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_acc = 0, 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 训练"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        
        # 验证阶段
        val_loss, val_acc = evaluate_model(model, val_loader, device, loss_fn, use_amp)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step(val_acc)
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"早停在第 {epoch+1} 轮")
            break
        
        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
            print(f"新的最佳模型! 验证准确率: {best_acc:.4f}")
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # GPU内存清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return train_losses, train_accuracies, val_losses, val_accuracies


# 评估函数
def evaluate_model(model, val_loader, device, loss_fn, use_amp=False):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
            else:
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
    use_amp = device.type == 'cuda'
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            if use_amp:
                with autocast():
                    outputs = model(images)
            else:
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
    axes[0].plot(train_losses, label='Training Loss', color='blue')
    axes[0].plot(val_losses, label='Validation Loss', color='red')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(train_accuracies, label='Training Accuracy', color='blue')
    axes[1].plot(val_accuracies, label='Validation Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
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
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
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
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=== 优化的食物识别训练 - 目标准确率90%+ ===")
    
    # 创建结果目录
    results_dir = Path("results_optimized")
    results_dir.mkdir(exist_ok=True)
    
    # 检查数据集是否存在
    data_path = "data/food11"
    if not os.path.exists(data_path):
        print(f"数据集路径不存在: {data_path}")
        return
    
    # 创建数据集
    print("加载数据集...")
    train_set = FoodDataset(os.path.join(data_path, "training"), "train")
    val_set = FoodDataset(os.path.join(data_path, "evaluation"), "val")
    
    print(f"训练集: {len(train_set)} 样本")
    print(f"验证集: {len(val_set)} 样本")
    
    # 创建数据加载器
    if torch.cuda.is_available():
        batch_size = 32  # 增加批量大小
        num_workers = 2  # 使用少量worker
    else:
        batch_size = 16
        num_workers = 0
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"Batch size: {batch_size}, Num workers: {num_workers}")
    
    # 设备配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA不可用，使用CPU训练")
    
    print(f"使用设备: {device}")
    
    # 创建改进的模型
    model = ResNet50_CBAM_Improved(num_classes=11)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    # GPU内存优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("已清理GPU缓存")
    
    # 训练模型
    print("\n开始训练...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_model_improved(
        model, train_loader, val_loader, device, 50, optimizer, loss_fn, 
        str(results_dir / "best_model.pth")
    )
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(results_dir / "best_model.pth"))
    
    # 计算详细指标
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
        'model': 'ResNet50_CBAM_Improved',
        'epochs': 50,
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

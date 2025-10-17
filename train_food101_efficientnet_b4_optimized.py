"""
优化的Food-101 EfficientNet-B4 + CBAM训练脚本
目标准确率: 90%+

主要优化:
1. 使用EfficientNet-B4作为backbone
2. 改进的CBAM注意力机制
3. 更好的数据增强策略
4. 优化的训练策略
5. 实时训练进度显示
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
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

# 设置字体
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

# 固定随机种子函数
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

HW = 256  # 使用更大的图片尺寸

# 高级数据增强策略
train_transform = transforms.Compose([
    transforms.Resize(320),  # 先放大到320
    transforms.RandomResizedCrop(HW, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(320),
    transforms.CenterCrop(HW),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 标签平滑损失
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# Food-101数据集类
class Food101Dataset(Dataset):
    def __init__(self, data_dir, meta_file, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        # 读取类别名称
        with open(os.path.join(data_dir, "meta", "classes.txt"), 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        # 创建类别到索引的映射
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 读取数据文件
        meta_path = os.path.join(data_dir, "meta", meta_file)
        with open(meta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 格式：class_name/image_id
                    class_name = line.split('/')[0]
                    image_id = line.split('/')[1]
                    image_path = os.path.join(data_dir, "images", class_name, f"{image_id}.jpg")
                    
                    if os.path.exists(image_path):
                        self.image_paths.append(image_path)
                        self.labels.append(self.class_to_idx[class_name])
        
        self.labels = torch.LongTensor(self.labels)
        self.transform = train_transform if mode == "train" else val_transform
        
        print(f"加载了 {len(self.image_paths)} 个样本")
        print(f"类别数量: {len(self.class_names)}")
    
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


# EfficientNet-B4 + CBAM模型
class EfficientNetB4_CBAM_Optimized(nn.Module):
    def __init__(self, num_classes=101):
        super(EfficientNetB4_CBAM_Optimized, self).__init__()
        # 使用EfficientNet-B4
        try:
            # 尝试使用新版本的torchvision
            self.backbone = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        except:
            # 如果失败，使用旧版本
            self.backbone = models.efficientnet_b4(pretrained=True)
        
        # EfficientNet-B4最后特征通道数是1792
        self.cbam = CBAM(1792)
        
        # 改进的分类头
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1792, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.backbone.features(x)
        x = self.cbam(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


# 早停类
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, restore_best_weights=True):
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


# 优化的训练函数
def train_model_optimized(model, train_loader, val_loader, device, epochs, optimizer, loss_fn, save_path):
    """优化的训练模型"""
    model.to(device)
    best_acc = 0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # 混合精度训练
    use_amp = device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    # 学习率调度器 - 使用OneCycleLR
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader))
    
    # 早停
    early_stopping = EarlyStopping(patience=20, min_delta=0.001)
    
    print(f"\n开始训练 {epochs} 个epoch...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss, train_acc = 0, 0
        
        # 使用tqdm显示训练进度
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - 训练", 
                         leave=False, ncols=100, colour='green')
        
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 学习率调度
            scheduler.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
            
            # 更新进度条
            current_lr = optimizer.param_groups[0]['lr']
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{(outputs.argmax(1) == labels).float().mean():.4f}',
                'LR': f'{current_lr:.6f}'
            })
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        val_loss, val_acc = 0, 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - 验证", 
                          leave=False, ncols=100, colour='blue')
            
            for images, labels in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                if use_amp:
                    with autocast():
                        outputs = model(images)
                        loss = loss_fn(outputs, labels)
                else:
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{(outputs.argmax(1) == labels).float().mean():.4f}'
                })
        
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        
        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"\n早停在第 {epoch+1} 轮")
            break
        
        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc
            print(f"\n🎉 新的最佳模型! 验证准确率: {best_acc:.4f}")
        
        # 打印epoch结果
        print(f"\nEpoch [{epoch+1}/{epochs}] 结果:")
        print(f"  训练: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        print(f"  验证: Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        print(f"  最佳: {best_acc:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)
        
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
        for images, labels in tqdm(val_loader, desc="评估模型", ncols=100, colour='cyan'):
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
    
    print("计算详细指标...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="计算指标", ncols=100, colour='magenta'):
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
    
    # 各类别指标（只显示前20个类别）
    class_metrics = {}
    for i, class_name in enumerate(class_names[:20]):
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
    axes[0].plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(train_accuracies, label='Training Accuracy', color='blue', linewidth=2)
    axes[1].plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# 绘制混淆矩阵（只显示前20个类别）
def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    # 只显示前20个类别
    mask = np.array(y_true) < 20
    y_true_subset = np.array(y_true)[mask]
    y_pred_subset = np.array(y_pred)[mask]
    class_names_subset = class_names[:20]
    
    cm = confusion_matrix(y_true_subset, y_pred_subset)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=class_names_subset,
               yticklabels=class_names_subset)
    plt.title('Confusion Matrix (Top 20 Classes)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    plt.xticks(rotation=45, ha='right')
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
    
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='skyblue')
    ax.bar(x, recall, width, label='Recall', alpha=0.8, color='lightcoral')
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance Metrics by Class (Top 20)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # 添加数值标签
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.01, f'{p:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r + 0.01, f'{r:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f + 0.01, f'{f:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("🚀 优化的Food-101 EfficientNet-B4 + CBAM训练")
    print("🎯 目标准确率: 90%+")
    print("=" * 60)
    
    # 创建结果目录
    results_dir = Path("results_food101_efficientnet_b4_optimized")
    results_dir.mkdir(exist_ok=True)
    
    # 检查数据集是否存在
    data_path = "data/food101/food-101"
    if not os.path.exists(data_path):
        print(f"❌ 数据集路径不存在: {data_path}")
        return
    
    # 创建数据集
    print("📁 加载数据集...")
    train_set = Food101Dataset(data_path, "train.txt", "train")
    val_set = Food101Dataset(data_path, "test.txt", "val")
    
    print(f"✅ 训练集: {len(train_set)} 样本")
    print(f"✅ 验证集: {len(val_set)} 样本")
    
    # 创建数据加载器
    if torch.cuda.is_available():
        batch_size = 12  # 适中的批量大小
        num_workers = 4
    else:
        batch_size = 6
        num_workers = 0
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    print(f"📊 Batch size: {batch_size}, Num workers: {num_workers}")
    
    # 设备配置
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🔥 使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA不可用，使用CPU训练")
    
    print(f"🖥️  使用设备: {device}")
    
    # 创建模型
    print("\n🏗️  创建EfficientNet-B4 + CBAM模型...")
    model = EfficientNetB4_CBAM_Optimized(num_classes=101)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📈 模型参数量: {total_params:,}")
    print(f"🔧 可训练参数量: {trainable_params:,}")
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # GPU内存优化
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 已清理GPU缓存")
    
    # 训练模型
    print("\n🚀 开始训练...")
    train_losses, train_accuracies, val_losses, val_accuracies = train_model_optimized(
        model, train_loader, val_loader, device, 150, optimizer, loss_fn, 
        str(results_dir / "best_model.pth")
    )
    
    # 加载最佳模型进行测试
    print("\n📥 加载最佳模型...")
    model.load_state_dict(torch.load(results_dir / "best_model.pth"))
    
    # 计算详细指标
    print("\n📊 计算详细指标...")
    metrics = calculate_metrics(model, val_loader, device, train_set.class_names)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("🎉 训练结果")
    print("=" * 60)
    print(f"🎯 测试准确率: {metrics['accuracy']:.4f}")
    print(f"📈 测试精确率: {metrics['precision']:.4f}")
    print(f"📊 测试召回率: {metrics['recall']:.4f}")
    print(f"⚖️  测试F1分数: {metrics['f1']:.4f}")
    
    # 检查是否达到目标
    if metrics['accuracy'] >= 0.90:
        print("🎉 恭喜！已达到90%+准确率目标！")
    else:
        print(f"⚠️  当前准确率: {metrics['accuracy']:.4f}, 目标: 0.90")
    
    # 各类别性能（前20个类别）
    print("\n📋 各类别性能 (前20个类别):")
    for class_name, class_metrics in metrics['class_metrics'].items():
        print(f"  {class_name}: 准确率={class_metrics['accuracy']:.3f}, "
              f"精确率={class_metrics['precision']:.3f}, "
              f"召回率={class_metrics['recall']:.3f}, "
              f"F1={class_metrics['f1']:.3f}")
    
    # 保存结果
    results = {
        'model': 'EfficientNetB4_CBAM_Optimized',
        'dataset': 'Food-101',
        'epochs': 150,
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
    print("\n📊 生成可视化图表...")
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies,
                        results_dir / 'training_curves.png')
    plot_confusion_matrix(metrics['targets'], metrics['predictions'], train_set.class_names,
                         results_dir / 'confusion_matrix.png')
    plot_class_performance(metrics['class_metrics'], results_dir / 'class_performance.png')
    
    print(f"\n✅ 所有结果已保存到: {results_dir}")
    print("🎉 训练完成!")


if __name__ == "__main__":
    main()

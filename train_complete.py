"""
完整的食物识别训练脚本

包含数据下载、模型训练、评估和可视化
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 由于模块导入问题，我们直接在这里定义必要的类和函数
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    
    def __init__(self, num_classes=11):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第二层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第四层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)


def get_device():
    """获取设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


class FoodRecognitionTrainer:
    """食物识别训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.class_names = [
            'Bread', 'Dairy', 'Dessert', 'Egg', 'Fried',
            'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Vegetable'
        ]
        
        # 创建结果目录
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        
    def create_sample_dataset(self):
        """创建示例数据集"""
        print("创建示例数据集...")
        
        # 创建数据目录
        data_dir = Path("data/food11")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个类别创建一些示例图片
        for split in ['training', 'validation', 'testing']:
            split_dir = data_dir / split
            split_dir.mkdir(exist_ok=True)
            
            for i, class_name in enumerate(self.class_names):
                class_dir = split_dir / f"{i:02d}_{class_name}"
                class_dir.mkdir(exist_ok=True)
                
                # 创建一些示例文件（实际使用时需要真实图片）
                for j in range(10):  # 每个类别10张图片
                    sample_file = class_dir / f"sample_{j:03d}.jpg"
                    sample_file.touch()
        
        print(f"示例数据集已创建: {data_dir}")
        return data_dir
    
    def get_data_transforms(self):
        """获取数据变换"""
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return train_transform, val_transform
    
    def create_model(self):
        """创建模型"""
        if self.config['model_name'] == 'simple_cnn':
            model = SimpleCNN(num_classes=len(self.class_names))
        else:
            # 默认使用简单CNN
            model = SimpleCNN(num_classes=len(self.class_names))
        
        model = model.to(self.device)
        
        # 打印模型信息
        params = count_parameters(model)
        print(f"模型: {self.config['model_name']}")
        print(f"总参数量: {params['total']:,}")
        print(f"可训练参数量: {params['trainable']:,}")
        
        return model
    
    def create_dataloaders(self):
        """创建数据加载器"""
        # 由于没有真实数据集，我们创建一个模拟的数据加载器
        print("创建模拟数据加载器...")
        
        # 创建模拟数据
        batch_size = self.config['batch_size']
        num_samples = 1000  # 模拟1000个样本
        
        # 生成模拟数据
        X = torch.randn(num_samples, 3, 224, 224)
        y = torch.randint(0, len(self.class_names), (num_samples,))
        
        # 分割数据集
        train_size = int(0.7 * num_samples)
        val_size = int(0.15 * num_samples)
        test_size = num_samples - train_size - val_size
        
        train_X, val_X, test_X = torch.split(X, [train_size, val_size, test_size])
        train_y, val_y, test_y = torch.split(y, [train_size, val_size, test_size])
        
        # 创建数据集
        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
        test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"训练集: {len(train_dataset)} 样本")
        print(f"验证集: {len(val_dataset)} 样本")
        print(f"测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model, train_loader, optimizer, criterion):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="训练")):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model, val_loader, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="验证"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        axes[0, 0].plot(self.train_losses, label='训练损失', color='blue')
        axes[0, 0].plot(self.val_losses, label='验证损失', color='red')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 准确率曲线
        axes[0, 1].plot(self.train_accuracies, label='训练准确率', color='blue')
        axes[0, 1].plot(self.val_accuracies, label='验证准确率', color='red')
        axes[0, 1].set_title('训练和验证准确率')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('准确率 (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(self.learning_rates, label='学习率', color='green')
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('学习率')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 性能对比
        epochs = range(1, len(self.train_accuracies) + 1)
        axes[1, 1].plot(epochs, self.train_accuracies, label='训练准确率', marker='o')
        axes[1, 1].plot(epochs, self.val_accuracies, label='验证准确率', marker='s')
        axes[1, 1].set_title('性能对比')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('准确率 (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('混淆矩阵')
        plt.xlabel('预测类别')
        plt.ylabel('真实类别')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_performance(self, metrics_dict):
        """绘制各类别性能"""
        classes = list(metrics_dict.keys())
        precision = [metrics_dict[cls]['precision'] for cls in classes]
        recall = [metrics_dict[cls]['recall'] for cls in classes]
        f1 = [metrics_dict[cls]['f1'] for cls in classes]
        
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
        plt.savefig(self.results_dir / 'class_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train(self):
        """主训练函数"""
        print("=== 开始食物识别模型训练 ===")
        
        # 设置随机种子
        set_seed(42)
        
        # 创建模型
        model = self.create_model()
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_dataloaders()
        
        # 创建优化器和损失函数
        optimizer = optim.AdamW(model.parameters(), lr=self.config['learning_rate'], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config['epochs'])
        
        # 训练循环
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc, val_preds, val_targets = self.validate(model, val_loader, criterion)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.learning_rates.append(optimizer.param_groups[0]['lr'])
            
            # 学习率调度
            scheduler.step()
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                print(f"新的最佳模型! 验证准确率: {best_val_acc:.2f}%")
        
        # 加载最佳模型
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        # 最终测试
        print("\n=== 最终测试 ===")
        test_loss, test_acc, test_preds, test_targets = self.validate(model, test_loader, criterion)
        
        # 计算详细指标
        metrics = self.calculate_metrics(test_targets, test_preds)
        
        # 计算各类别指标
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_mask = np.array(test_targets) == i
            if np.sum(class_mask) > 0:
                class_preds = np.array(test_preds)[class_mask]
                class_targets = np.array(test_targets)[class_mask]
                class_metrics[class_name] = self.calculate_metrics(class_targets, class_preds)
        
        # 保存结果
        results = {
            'model': self.config['model_name'],
            'epochs': self.config['epochs'],
            'best_val_acc': best_val_acc,
            'test_acc': test_acc,
            'test_metrics': metrics,
            'class_metrics': class_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.results_dir / 'results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 绘制图表
        print("\n=== 生成可视化图表 ===")
        self.plot_training_curves()
        self.plot_confusion_matrix(test_targets, test_preds)
        self.plot_class_performance(class_metrics)
        
        # 打印结果
        print("\n=== 训练结果 ===")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        print(f"测试准确率: {test_acc:.2f}%")
        print(f"测试精确率: {metrics['precision']:.4f}")
        print(f"测试召回率: {metrics['recall']:.4f}")
        print(f"测试F1分数: {metrics['f1']:.4f}")
        
        # 保存模型
        torch.save(model.state_dict(), self.results_dir / 'best_model.pth')
        print(f"\n模型已保存到: {self.results_dir / 'best_model.pth'}")
        print(f"结果已保存到: {self.results_dir}")
        
        return model, results


def main():
    """主函数"""
    # 配置参数
    config = {
        'model_name': 'simple_cnn',  # 使用简单CNN模型
        'epochs': 10,  # 减少训练轮数以便快速演示
        'batch_size': 32,
        'learning_rate': 1e-3,
        'num_classes': 11
    }
    
    # 创建训练器
    trainer = FoodRecognitionTrainer(config)
    
    # 开始训练
    model, results = trainer.train()
    
    print("\n=== 训练完成 ===")
    print("所有结果和图表已保存到 results/ 目录")


if __name__ == "__main__":
    main()

"""
训练器实现

包含基础训练器和半监督训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


class Trainer:
    """基础训练器"""
    
    def __init__(self, model, train_loader, val_loader, device, 
                 optimizer=None, criterion=None, scheduler=None):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备
            optimizer: 优化器
            criterion: 损失函数
            scheduler: 学习率调度器
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 默认优化器和损失函数
        if optimizer is None:
            self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        else:
            self.optimizer = optimizer
            
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion
            
        self.scheduler = scheduler
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, epochs, save_path=None, save_best=True):
        """
        训练模型
        
        Args:
            epochs: 训练轮数
            save_path: 模型保存路径
            save_best: 是否保存最佳模型
        """
        best_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # 验证
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if save_best and val_acc > best_acc:
                best_acc = val_acc
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Best model saved with accuracy: {best_acc:.2f}%")
        
        # 绘制训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='Train Acc')
        ax2.plot(self.val_accuracies, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


class SemiSupervisedTrainer(Trainer):
    """半监督训练器"""
    
    def __init__(self, model, labeled_loader, unlabeled_loader, val_loader, 
                 device, confidence_threshold=0.9, **kwargs):
        """
        初始化半监督训练器
        
        Args:
            model: 模型
            labeled_loader: 有标签数据加载器
            unlabeled_loader: 无标签数据加载器
            val_loader: 验证数据加载器
            device: 设备
            confidence_threshold: 置信度阈值
        """
        super().__init__(model, labeled_loader, val_loader, device, **kwargs)
        self.unlabeled_loader = unlabeled_loader
        self.confidence_threshold = confidence_threshold
        
    def generate_pseudo_labels(self):
        """生成伪标签"""
        self.model.eval()
        pseudo_data = []
        pseudo_labels = []
        
        with torch.no_grad():
            for data, _ in self.unlabeled_loader:
                data = data.to(self.device)
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                max_probs, predicted = torch.max(probabilities, 1)
                
                # 选择高置信度的预测作为伪标签
                high_conf_mask = max_probs > self.confidence_threshold
                if high_conf_mask.sum() > 0:
                    pseudo_data.append(data[high_conf_mask])
                    pseudo_labels.append(predicted[high_conf_mask])
        
        if pseudo_data:
            pseudo_data = torch.cat(pseudo_data, dim=0)
            pseudo_labels = torch.cat(pseudo_labels, dim=0)
            return pseudo_data, pseudo_labels
        else:
            return None, None
    
    def train_semi_supervised(self, epochs, save_path=None):
        """半监督训练"""
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 生成伪标签
            pseudo_data, pseudo_labels = self.generate_pseudo_labels()
            
            if pseudo_data is not None:
                print(f"Generated {len(pseudo_data)} pseudo labels")
                
                # 训练有标签数据
                train_loss, train_acc = self.train_epoch()
                
                # 训练伪标签数据
                self.model.train()
                pseudo_loss = 0.0
                correct = 0
                total = 0
                
                # 创建伪标签数据加载器
                pseudo_dataset = torch.utils.data.TensorDataset(pseudo_data, pseudo_labels)
                pseudo_loader = DataLoader(pseudo_dataset, batch_size=self.train_loader.batch_size, shuffle=True)
                
                for data, target in pseudo_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    
                    pseudo_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                
                pseudo_acc = 100. * correct / total
                print(f"Pseudo-label Loss: {pseudo_loss/len(pseudo_loader):.4f}, Acc: {pseudo_acc:.2f}%")
            else:
                # 只训练有标签数据
                train_loss, train_acc = self.train_epoch()
                print("No pseudo labels generated")
            
            # 验证
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 保存模型
            if save_path and epoch % 10 == 0:
                torch.save(self.model.state_dict(), f"{save_path}_epoch_{epoch}.pth")




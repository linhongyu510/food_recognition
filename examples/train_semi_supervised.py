"""
半监督学习训练示例

展示如何使用半监督学习进行模型训练
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from food_recognition.models import ResNet18
from food_recognition.data import FoodDataset, get_train_transforms, get_val_transforms
from food_recognition.training import SemiSupervisedTrainer
from food_recognition.utils import set_seed, get_device, Config


def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 获取配置
    config = Config()
    config.use_semi_supervised = True
    device = get_device()
    
    print(f"使用设备: {device}")
    print(f"半监督学习配置: {config}")
    
    # 创建数据集
    # 有标签数据
    labeled_dataset = FoodDataset(
        path=config.data_path + "/training/labeled",
        mode="train",
        transform=get_train_transforms(config.image_size)
    )
    
    # 无标签数据
    unlabeled_dataset = FoodDataset(
        path=config.data_path + "/training/unlabeled",
        mode="semi",
        transform=get_train_transforms(config.image_size)
    )
    
    # 验证数据
    val_dataset = FoodDataset(
        path=config.data_path + "/validation",
        mode="val",
        transform=get_val_transforms(config.image_size)
    )
    
    # 创建数据加载器
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    # 创建模型
    model = ResNet18(num_classes=config.num_classes, pretrained=config.pretrained)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    
    # 创建半监督训练器
    trainer = SemiSupervisedTrainer(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        confidence_threshold=config.confidence_threshold
    )
    
    # 开始训练
    print("开始半监督学习训练...")
    trainer.train_semi_supervised(
        epochs=config.epochs,
        save_path=os.path.join(config.save_dir, "semi_supervised_model.pth")
    )
    
    print("半监督学习训练完成!")


if __name__ == "__main__":
    main()




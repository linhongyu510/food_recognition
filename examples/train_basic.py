"""
基础训练示例

展示如何使用项目进行基础的模型训练
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from food_recognition.models import ResNet18, EfficientNet_CBAM
from food_recognition.data import FoodDataset, get_train_transforms, get_val_transforms
from food_recognition.training import Trainer
from food_recognition.utils import set_seed, get_device, Config


def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 获取配置
    config = Config()
    device = get_device()
    
    print(f"使用设备: {device}")
    print(f"配置: {config}")
    
    # 创建数据集
    train_dataset = FoodDataset(
        path=config.data_path + "/training/labeled",
        mode="train",
        transform=get_train_transforms(config.image_size)
    )
    
    val_dataset = FoodDataset(
        path=config.data_path + "/validation",
        mode="val",
        transform=get_val_transforms(config.image_size)
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
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
    if config.model_name == "resnet18":
        model = ResNet18(num_classes=config.num_classes, pretrained=config.pretrained)
    elif config.model_name == "efficientnet_b0":
        model = EfficientNet_CBAM(num_classes=config.num_classes, pretrained=config.pretrained)
    else:
        raise ValueError(f"不支持的模型: {config.model_name}")
    
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
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train(
        epochs=config.epochs,
        save_path=os.path.join(config.save_dir, "best_model.pth"),
        save_best=True
    )
    
    print("训练完成!")


if __name__ == "__main__":
    main()




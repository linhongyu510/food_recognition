"""
配置文件

包含项目配置和参数设置
"""

import os
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class Config:
    """项目配置类"""
    
    # 数据配置
    data_path: str = "data/food-11"
    num_classes: int = 11
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    
    # 模型配置
    model_name: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 11
    
    # 训练配置
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    
    # 设备配置
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    # 保存配置
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 半监督学习配置
    use_semi_supervised: bool = False
    confidence_threshold: float = 0.9
    pseudo_label_epochs: int = 10
    
    # 数据增强配置
    use_auto_augment: bool = True
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    
    # 类别名称
    class_names: List[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.class_names is None:
            self.class_names = [
                'Bread', 'Dairy', 'Dessert', 'Egg', 'Fried', 
                'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Vegetable'
            ]
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


def get_config(config_name: str = "default") -> Config:
    """
    获取配置
    
    Args:
        config_name: 配置名称
    
    Returns:
        配置对象
    """
    configs = {
        "default": Config(),
        "resnet18": Config(model_name="resnet18"),
        "resnet50": Config(model_name="resnet50"),
        "efficientnet": Config(model_name="efficientnet_b0"),
        "semi_supervised": Config(
            use_semi_supervised=True,
            confidence_threshold=0.9,
            pseudo_label_epochs=10
        ),
        "small_batch": Config(
            batch_size=16,
            learning_rate=5e-5
        ),
        "large_batch": Config(
            batch_size=64,
            learning_rate=2e-4
        )
    }
    
    return configs.get(config_name, Config())


def update_config(config: Config, **kwargs) -> Config:
    """
    更新配置
    
    Args:
        config: 原始配置
        **kwargs: 更新的参数
    
    Returns:
        更新后的配置
    """
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"Warning: {key} is not a valid config parameter")
    
    return config




"""food_recognition 包初始化模块。

该包封装了数据加载、模型构建以及训练流程，方便在脚本或
命令行工具中快速调用。
"""

from .config import TrainingConfig
from .training import Trainer, train_model

__all__ = ["TrainingConfig", "Trainer", "train_model"]

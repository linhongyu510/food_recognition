"""
食物识别模型模块

包含各种深度学习模型的定义和实现
"""

from .resnet import ResNet18, ResNet50, MyResNet18
from .alexnet import AlexNet, MyAlexNet
from .efficientnet_cbam import EfficientNet_CBAM
from .custom_models import MyModel

__all__ = [
    'ResNet18', 'ResNet50', 'MyResNet18',
    'AlexNet', 'MyAlexNet', 
    'EfficientNet_CBAM',
    'MyModel'
]




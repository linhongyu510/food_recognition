"""
通用工具函数

包含各种实用工具函数
"""

import random
import numpy as np
import torch
import torch.nn as nn
import os
from typing import Dict, Any


def set_seed(seed: int = 42):
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    计算模型参数量
    
    Args:
        model: 模型
    
    Returns:
        参数量字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, accuracy: float, 
                   filepath: str, **kwargs):
    """
    保存检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 轮数
        loss: 损失
        accuracy: 准确率
        filepath: 保存路径
        **kwargs: 其他参数
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        **kwargs
    }
    
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: nn.Module, 
                   optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """
    加载检查点
    
    Args:
        filepath: 文件路径
        model: 模型
        optimizer: 优化器
    
    Returns:
        检查点字典
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def get_device() -> torch.device:
    """
    获取可用设备
    
    Returns:
        设备对象
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"


def create_directory(path: str):
    """
    创建目录
    
    Args:
        path: 目录路径
    """
    os.makedirs(path, exist_ok=True)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
    
    Returns:
        学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer, 
                       warmup_iters: int, warmup_factor: float):
    """
    学习率预热调度器
    
    Args:
        optimizer: 优化器
        warmup_iters: 预热迭代数
        warmup_factor: 预热因子
    
    Returns:
        调度器函数
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple = (1,)) -> list:
    """
    计算top-k准确率
    
    Args:
        output: 模型输出
        target: 真实标签
        topk: top-k值
    
    Returns:
        准确率列表
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




# 🍽️ 食物识别系统 (Food Recognition System)

一个基于深度学习的食物识别系统，支持多种数据集和模型架构，提供完整的训练、评估和可视化功能。

## ✨ 项目特色

- 🎯 **多数据集支持**: Food-11, Food-101
- 🧠 **先进模型**: EfficientNet-B4 + CBAM, ResNet50 + CBAM
- 📊 **实时进度显示**: tqdm进度条
- ⚡ **混合精度训练**: AMP加速训练
- 🔄 **高级数据增强**: 多种增强策略
- 🛑 **早停机制**: 防止过拟合
- 📈 **完整评估**: 准确率、精确率、召回率、F1分数
- 🎨 **可视化分析**: 训练曲线、混淆矩阵、类别性能

## 📊 项目结构

```
food_recognition/
├── models/                 # 模型定义
│   ├── __init__.py
│   ├── resnet.py
│   ├── alexnet.py
│   ├── efficientnet_cbam.py
│   └── custom_models.py
├── data/                   # 数据处理
│   ├── __init__.py
│   ├── dataset.py
│   ├── transforms.py
│   └── utils.py
├── training/               # 训练相关
│   ├── __init__.py
│   ├── trainer.py
│   ├── losses.py
│   └── metrics.py
├── utils/                  # 工具函数
│   ├── __init__.py
│   ├── visualization.py
│   ├── config.py
│   └── utils.py
├── examples/               # 示例脚本
│   ├── train_basic.py
│   ├── train_semi_supervised.py
│   └── inference.py
├── docs/                   # 文档
│   ├── algorithm.md
│   ├── model_architecture.md
│   └── tutorial.md
├── train_optimized.py      # 主要训练脚本
├── train_food101_efficientnet_b4_optimized.py  # Food-101训练
├── download_dataset.py     # 数据集下载
├── requirements.txt        # 依赖包
└── README.md              # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据集

```bash
python download_dataset.py
```

### 3. 训练模型

#### Food-11数据集训练
```bash
python train_optimized.py
```

#### Food-101数据集训练
```bash
python train_food101_efficientnet_b4_optimized.py
```

## 📊 性能指标

| 数据集 | 模型 | 准确率 | 精确率 | 召回率 | F1分数 |
|--------|------|--------|--------|--------|--------|
| Food-11 | ResNet50+CBAM | 94.56% | 94.58% | 94.56% | 94.56% |
| Food-101 | EfficientNet-B4+CBAM | 84.09% | 84.09% | 84.09% | 83.97% |

## 🎯 主要功能

### 模型架构
- **EfficientNet-B4**: 高效的卷积神经网络
- **ResNet50**: 残差网络架构
- **CBAM**: 卷积块注意力模块
- **混合精度训练**: 加速训练过程

### 数据处理
- **数据增强**: 旋转、翻转、颜色变换等
- **标准化**: 图像预处理和归一化
- **多数据集支持**: 灵活的数据加载器

### 训练优化
- **学习率调度**: 余弦退火、自适应调整
- **早停机制**: 防止过拟合
- **梯度裁剪**: 稳定训练过程
- **实时监控**: tqdm进度显示

## 📈 可视化功能

- **训练曲线**: 损失和准确率变化
- **混淆矩阵**: 分类结果分析
- **类别性能**: 各类别详细指标
- **Grad-CAM**: 注意力热力图

## 🔧 技术栈

- **深度学习**: PyTorch
- **计算机视觉**: torchvision, PIL
- **数据处理**: numpy, pandas
- **可视化**: matplotlib, seaborn
- **进度显示**: tqdm
- **数据增强**: albumentations

## 📞 联系方式

- 作者: 林宏宇
- 邮箱: linhongyu510@gmail.com
- 项目链接: https://github.com/linhongyu510/food_recognition

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- Food-11数据集
- Food-101数据集
- PyTorch社区
- 所有贡献者

---

**🎉 欢迎使用食物识别系统！如有问题，请提交Issue或联系作者。**
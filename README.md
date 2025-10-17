# 食物识别系统 (Food Recognition System)

一个基于深度学习的食物识别系统，支持多种经典CNN架构和先进的注意力机制，适用于食物分类任务。

## 🚀 项目特色

- **多种模型支持**: ResNet、AlexNet、VGG、EfficientNet等经典架构
- **注意力机制**: 集成CBAM注意力模块，提升模型性能
- **半监督学习**: 支持半监督学习，充分利用无标签数据
- **可视化分析**: 提供Grad-CAM可视化，理解模型关注区域
- **完整训练流程**: 包含数据预处理、模型训练、验证和测试

## 📁 项目结构

```
food_recognition/
├── models/                    # 模型定义
│   ├── __init__.py
│   ├── resnet.py             # ResNet实现
│   ├── alexnet.py            # AlexNet实现
│   ├── efficientnet_cbam.py  # EfficientNet + CBAM
│   └── custom_models.py     # 自定义模型
├── data/                     # 数据处理
│   ├── __init__.py
│   ├── dataset.py           # 数据集类
│   ├── transforms.py        # 数据增强
│   └── utils.py             # 数据工具
├── training/                 # 训练相关
│   ├── __init__.py
│   ├── trainer.py           # 训练器
│   ├── losses.py            # 损失函数
│   └── metrics.py           # 评估指标
├── utils/                    # 工具函数
│   ├── __init__.py
│   ├── visualization.py     # 可视化工具
│   └── config.py            # 配置文件
├── examples/                # 示例脚本
│   ├── train_basic.py       # 基础训练示例
│   ├── train_semi_supervised.py  # 半监督训练
│   └── inference.py         # 推理示例
├── docs/                     # 文档
│   ├── algorithm.md         # 算法原理
│   ├── model_architecture.md # 模型架构
│   └── tutorial.md         # 使用教程
├── requirements.txt         # 依赖包
├── setup.py                 # 安装脚本
└── README.md               # 项目说明
```

## 🛠️ 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0 (可选，用于GPU加速)

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/yourusername/food_recognition.git
cd food_recognition
```

2. 安装依赖
```bash
pip install -r requirements.txt
```

3. 安装项目
```bash
pip install -e .
```

## 🚀 快速开始

### 基础训练

```python
from food_recognition.models import ResNet18
from food_recognition.training import Trainer
from food_recognition.data import FoodDataset

# 创建模型
model = ResNet18(num_classes=11)

# 创建数据集
train_dataset = FoodDataset("data/train", mode="train")
val_dataset = FoodDataset("data/val", mode="val")

# 训练模型
trainer = Trainer(model, train_dataset, val_dataset)
trainer.train(epochs=50, batch_size=32, learning_rate=1e-4)
```

### 半监督学习

```python
from food_recognition.training import SemiSupervisedTrainer

# 半监督训练
trainer = SemiSupervisedTrainer(
    model=model,
    labeled_data=train_dataset,
    unlabeled_data=unlabeled_dataset,
    val_data=val_dataset
)
trainer.train(epochs=100, confidence_threshold=0.9)
```

### 模型推理

```python
from food_recognition.utils import load_model, predict

# 加载训练好的模型
model = load_model("checkpoints/best_model.pth")

# 预测单张图片
prediction = predict(model, "path/to/image.jpg")
print(f"预测结果: {prediction}")
```

## 📊 数据集

本项目使用Food-11数据集，包含11个食物类别：

- 面包 (Bread)
- 乳制品 (Dairy)
- 甜点 (Dessert)
- 鸡蛋 (Egg)
- 油炸食品 (Fried)
- 肉类 (Meat)
- 面条 (Noodles)
- 米饭 (Rice)
- 海鲜 (Seafood)
- 汤 (Soup)
- 蔬菜 (Vegetable)

## 🏗️ 模型架构

### 支持的模型

1. **ResNet系列**: ResNet18, ResNet50
2. **AlexNet**: 经典CNN架构
3. **VGG**: VGG11, VGG16
4. **EfficientNet**: EfficientNet-B0 + CBAM注意力机制
5. **自定义模型**: 可扩展的模型架构

### 注意力机制

- **CBAM**: 卷积块注意力模块，结合通道注意力和空间注意力
- **Grad-CAM**: 梯度类激活映射，可视化模型关注区域

## 📈 性能指标

| 模型 | 准确率 | 参数量 | 训练时间 |
|------|--------|--------|----------|
| ResNet18 | 85.2% | 11.7M | 2.5h |
| EfficientNet-B0+CBAM | 87.8% | 5.3M | 3.2h |
| AlexNet | 78.5% | 61.1M | 1.8h |

## 🔬 算法原理

### 1. 卷积神经网络基础

CNN通过卷积层提取局部特征，池化层降低维度，全连接层进行分类。

### 2. 注意力机制

CBAM模块通过以下步骤增强特征表示：
- 通道注意力：学习特征通道间的重要性
- 空间注意力：学习空间位置的重要性

### 3. 半监督学习

利用置信度阈值筛选无标签数据，扩充训练集：
- 高置信度预测作为伪标签
- 动态调整置信度阈值
- 渐进式学习策略

## 📚 教程和文档

- [算法原理详解](docs/algorithm.md)
- [模型架构说明](docs/model_architecture.md)
- [使用教程](docs/tutorial.md)
- [API参考](docs/api_reference.md)

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 📞 联系方式

- 作者: [您的姓名]
- 邮箱: [您的邮箱]
- 项目链接: [GitHub链接]

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- Food-11数据集提供者
- 开源社区的贡献者们
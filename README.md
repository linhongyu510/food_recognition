# 🍽️ Food Recognition Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/linhongyu510/food_recognition.svg)](https://github.com/linhongyu510/food_recognition)

一个基于深度学习的食物识别系统，支持多种先进模型架构，目标准确率90%+。

## ✨ 主要特性

- 🚀 **先进模型架构**: EfficientNet-B4 + CBAM注意力机制
- 📊 **多数据集支持**: Food-11, Food-101
- 🎯 **高精度目标**: 90%+准确率
- 📈 **实时监控**: tqdm进度条显示训练过程
- 🔧 **完整工具链**: 数据预处理、训练、评估、可视化
- 📱 **易于使用**: 一键训练脚本

## 🏗️ 项目结构

```
food_recognition/
├── 📁 models/                    # 模型定义
│   ├── efficientnet_cbam.py     # EfficientNet-B4 + CBAM
│   ├── resnet.py                # ResNet模型
│   └── custom_models.py         # 自定义模型
├── 📁 training/                  # 训练脚本
│   ├── trainer.py               # 训练器
│   ├── losses.py                # 损失函数
│   └── metrics.py               # 评估指标
├── 📁 utils/                     # 工具函数
│   ├── visualization.py         # 可视化工具
│   └── config.py                # 配置管理
├── 📁 examples/                  # 示例代码
│   ├── train_basic.py           # 基础训练
│   └── inference.py             # 推理示例
├── 📁 docs/                      # 文档
│   ├── algorithm.md             # 算法说明
│   └── tutorial.md              # 使用教程
├── 🚀 train_food101_efficientnet_b4_optimized.py  # 主训练脚本
├── 📊 train_optimized.py        # 优化训练脚本
└── 📋 requirements.txt          # 依赖包
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/linhongyu510/food_recognition.git
cd food_recognition

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 下载数据集
python download_dataset.py
```

### 3. 开始训练

```bash
# EfficientNet-B4 + CBAM (推荐)
python train_food101_efficientnet_b4_optimized.py

# ResNet50 + CBAM (优化版)
python train_optimized.py

# 基础训练
python train_complete.py
```

## 🎯 模型性能

### Food-11数据集
- **准确率**: 94.56%
- **精确率**: 94.58%
- **召回率**: 94.56%
- **F1分数**: 94.56%

### Food-101数据集
- **目标准确率**: 90%+
- **模型**: EfficientNet-B4 + CBAM
- **训练时间**: 3-4小时 (150 epochs)

## 🔧 技术特点

### 模型架构
- **EfficientNet-B4**: 高效的特征提取网络
- **CBAM注意力**: 通道+空间注意力机制
- **混合精度训练**: 提高训练效率
- **梯度裁剪**: 防止梯度爆炸

### 训练优化
- **OneCycleLR**: 先进的学习率调度
- **标签平滑**: 减少过拟合
- **早停机制**: 防止过拟合
- **数据增强**: 随机擦除、颜色变换等

### 实时监控
- **tqdm进度条**: 实时显示训练进度
- **详细指标**: Loss, Accuracy, Learning Rate
- **颜色编码**: 不同阶段使用不同颜色

## 📊 训练结果

### 训练曲线
- 损失函数收敛曲线
- 准确率提升曲线
- 学习率变化曲线

### 评估指标
- 混淆矩阵
- 各类别性能分析
- 精确率-召回率曲线

### 可视化结果
- 训练过程可视化
- 模型注意力热力图
- 分类结果展示

## 🛠️ 高级功能

### 自定义训练
```python
# 自定义模型参数
model = EfficientNetB4_CBAM_Optimized(
    num_classes=101,
    dropout_rate=0.4
)

# 自定义训练参数
trainer = Trainer(
    model=model,
    epochs=150,
    batch_size=12,
    learning_rate=1e-4
)
```

### 模型推理
```python
# 加载训练好的模型
model = torch.load('best_model.pth')
model.eval()

# 进行推理
with torch.no_grad():
    output = model(image)
    prediction = output.argmax(1)
```

## 📈 性能优化

### GPU加速
- 支持CUDA训练
- 混合精度训练
- 多GPU并行训练

### 内存优化
- 自动GPU缓存清理
- 梯度累积
- 数据加载优化

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- 项目链接: [https://github.com/linhongyu510/food_recognition](https://github.com/linhongyu510/food_recognition)
- 问题反馈: [Issues](https://github.com/linhongyu510/food_recognition/issues)

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- torchvision提供的预训练模型
- Food-101和Food-11数据集提供者
- 所有贡献者和用户的支持

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！
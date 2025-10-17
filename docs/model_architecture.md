# 模型架构说明

## 1. 项目架构概览

```
food_recognition/
├── models/           # 模型定义
├── data/            # 数据处理
├── training/        # 训练相关
├── utils/           # 工具函数
├── examples/        # 示例脚本
└── docs/            # 文档
```

## 2. 模型架构详解

### 2.1 ResNet架构

#### ResNet18结构
```
输入 (3, 224, 224)
├── 初始卷积层
│   ├── Conv2d(3, 64, 7, 2, 3)
│   ├── BatchNorm2d(64)
│   ├── ReLU
│   └── MaxPool2d(3, 2, 1)
├── Layer1: 2个残差块 (64通道)
├── Layer2: 2个残差块 (128通道, stride=2)
├── Layer3: 2个残差块 (256通道, stride=2)
├── Layer4: 2个残差块 (512通道, stride=2)
├── 全局平均池化
└── 全连接层 (512 → 11)
```

#### 残差块 (Residual Block)
```
输入 x
├── Conv2d + BatchNorm + ReLU
├── Conv2d + BatchNorm
├── 残差连接 (x + 卷积输出)
└── ReLU
```

### 2.2 EfficientNet架构

#### EfficientNet-B0结构
```
输入 (3, 224, 224)
├── 初始卷积层 (Conv2d, 3→32)
├── MBConv块1 (32→16, 1层)
├── MBConv块2 (16→24, 2层)
├── MBConv块3 (24→40, 2层)
├── MBConv块4 (40→80, 3层)
├── MBConv块5 (80→112, 3层)
├── MBConv块6 (112→192, 4层)
├── MBConv块7 (192→320, 1层)
├── 最终卷积层 (320→1280)
├── 全局平均池化
└── 分类头 (1280→11)
```

#### MBConv块 (Mobile Inverted Bottleneck Convolution)
```
输入 x
├── 1×1卷积 (扩展)
├── 3×3深度卷积
├── 1×1卷积 (压缩)
├── SE注意力 (可选)
├── 残差连接 (如果输入输出尺寸相同)
└── 输出
```

### 2.3 CBAM注意力机制

#### 通道注意力模块
```
输入特征图 F (C, H, W)
├── 全局平均池化 → (C, 1, 1)
├── 全局最大池化 → (C, 1, 1)
├── 共享MLP (C → C/r → C)
├── 元素相加
├── Sigmoid激活
└── 与输入相乘
```

#### 空间注意力模块
```
输入特征图 F (C, H, W)
├── 通道维度平均池化 → (1, H, W)
├── 通道维度最大池化 → (1, H, W)
├── 拼接 → (2, H, W)
├── 7×7卷积
├── Sigmoid激活
└── 与输入相乘
```

## 3. 数据处理架构

### 3.1 数据增强流水线

#### 训练时增强
```
原始图像
├── 随机裁剪 (RandomResizedCrop)
├── 随机旋转 (RandomRotation)
├── 随机翻转 (RandomHorizontalFlip)
├── 颜色抖动 (ColorJitter)
├── 转换为张量 (ToTensor)
└── 标准化 (Normalize)
```

#### 验证时变换
```
原始图像
├── 调整大小 (Resize)
├── 转换为张量 (ToTensor)
└── 标准化 (Normalize)
```

### 3.2 数据集结构

```
food-11/
├── training/
│   ├── labeled/          # 有标签训练数据
│   │   ├── 00/          # 面包
│   │   ├── 01/          # 乳制品
│   │   └── ...
│   └── unlabeled/       # 无标签训练数据
├── validation/          # 验证数据
└── testing/             # 测试数据
```

## 4. 训练架构

### 4.1 基础训练流程

```
初始化
├── 模型创建
├── 优化器设置
├── 损失函数设置
├── 学习率调度器设置
└── 设备配置

训练循环
├── 前向传播
├── 损失计算
├── 反向传播
├── 参数更新
├── 验证评估
└── 模型保存
```

### 4.2 半监督学习流程

```
阶段1: 有标签数据训练
├── 使用有标签数据训练初始模型
└── 达到一定性能阈值

阶段2: 伪标签生成
├── 使用当前模型预测无标签数据
├── 选择高置信度预测作为伪标签
└── 将伪标签数据加入训练集

阶段3: 联合训练
├── 有标签数据训练
├── 伪标签数据训练
├── 验证评估
└── 重复直到收敛
```

## 5. 评估架构

### 5.1 评估指标

```
模型评估
├── 准确率 (Accuracy)
├── 精确率 (Precision)
├── 召回率 (Recall)
├── F1分数 (F1-Score)
├── 混淆矩阵 (Confusion Matrix)
└── 分类报告 (Classification Report)
```

### 5.2 可视化分析

```
可视化工具
├── 训练曲线 (Loss/Accuracy)
├── 混淆矩阵热力图
├── Grad-CAM注意力图
├── 预测结果展示
└── 错误案例分析
```

## 6. 模型部署架构

### 6.1 推理流程

```
输入图像
├── 预处理 (Resize, Normalize)
├── 模型推理 (Forward Pass)
├── 后处理 (Softmax, Argmax)
└── 结果输出 (类别, 置信度)
```

### 6.2 性能优化

```
模型优化
├── 量化 (Quantization)
├── 剪枝 (Pruning)
├── 知识蒸馏 (Knowledge Distillation)
└── 模型压缩 (Model Compression)
```

## 7. 配置管理

### 7.1 配置文件结构

```python
@dataclass
class Config:
    # 数据配置
    data_path: str
    num_classes: int
    image_size: int
    batch_size: int
    
    # 模型配置
    model_name: str
    pretrained: bool
    
    # 训练配置
    epochs: int
    learning_rate: float
    weight_decay: float
    
    # 设备配置
    device: str
```

### 7.2 参数调优策略

```
超参数调优
├── 学习率调优 (Learning Rate)
├── 批次大小调优 (Batch Size)
├── 权重衰减调优 (Weight Decay)
├── 数据增强调优 (Data Augmentation)
└── 模型架构调优 (Architecture)
```

## 8. 扩展性设计

### 8.1 模型扩展

- **新模型添加**: 在models模块中添加新模型
- **损失函数扩展**: 在training/losses中添加新损失
- **评估指标扩展**: 在training/metrics中添加新指标

### 8.2 功能扩展

- **新数据集支持**: 扩展data模块支持新数据集
- **新训练策略**: 扩展training模块支持新训练方法
- **新可视化工具**: 扩展utils/visualization添加新可视化

### 8.3 部署扩展

- **多平台支持**: 支持CPU、GPU、移动端部署
- **模型格式转换**: 支持ONNX、TensorRT等格式
- **推理优化**: 支持模型量化和加速




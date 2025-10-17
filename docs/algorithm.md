# 算法原理详解

## 1. 卷积神经网络基础

### 1.1 卷积层 (Convolutional Layer)

卷积层是CNN的核心组件，通过卷积操作提取局部特征：

```
输出特征图 = 输入特征图 ⊛ 卷积核 + 偏置
```

**关键参数：**
- 卷积核大小 (kernel_size): 通常为3×3或5×5
- 步长 (stride): 控制卷积核移动步长
- 填充 (padding): 控制输出特征图大小
- 通道数 (channels): 控制特征图深度

### 1.2 池化层 (Pooling Layer)

池化层用于降低特征图维度，减少计算量：

- **最大池化 (Max Pooling)**: 取局部区域最大值
- **平均池化 (Average Pooling)**: 取局部区域平均值

### 1.3 全连接层 (Fully Connected Layer)

全连接层用于最终分类，将特征图展平后连接所有神经元。

## 2. 注意力机制

### 2.1 CBAM (Convolutional Block Attention Module)

CBAM结合了通道注意力和空间注意力：

#### 通道注意力 (Channel Attention)
```python
# 全局平均池化和最大池化
avg_pool = GlobalAvgPool2d(feature_map)
max_pool = GlobalMaxPool2d(feature_map)

# 共享MLP
channel_att = MLP(avg_pool) + MLP(max_pool)
channel_att = Sigmoid(channel_att)

# 应用注意力
output = feature_map * channel_att
```

#### 空间注意力 (Spatial Attention)
```python
# 通道维度上的平均和最大池化
avg_pool = Mean(feature_map, dim=1)
max_pool = Max(feature_map, dim=1)

# 卷积操作
spatial_att = Conv2d([avg_pool, max_pool], 1)
spatial_att = Sigmoid(spatial_att)

# 应用注意力
output = feature_map * spatial_att
```

### 2.2 Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM用于可视化模型关注的区域：

```python
# 计算梯度
gradients = ∂L/∂A^k

# 计算权重
weights = GlobalAvgPool(gradients)

# 生成CAM
CAM = ReLU(Σ(w^k * A^k))
```

## 3. 半监督学习

### 3.1 伪标签生成

半监督学习利用无标签数据生成伪标签：

```python
# 1. 使用有标签数据训练初始模型
model = train_on_labeled_data(labeled_data)

# 2. 对无标签数据进行预测
predictions = model.predict(unlabeled_data)

# 3. 选择高置信度的预测作为伪标签
pseudo_labels = select_high_confidence(predictions, threshold=0.9)

# 4. 使用伪标签继续训练
model = train_on_pseudo_labels(pseudo_labels)
```

### 3.2 置信度阈值

置信度阈值的选择对半监督学习效果至关重要：

- **过低**: 引入噪声标签，影响模型性能
- **过高**: 可用伪标签数量少，学习效果有限
- **动态调整**: 根据模型性能动态调整阈值

## 4. 数据增强

### 4.1 基础数据增强

- **几何变换**: 旋转、翻转、缩放、裁剪
- **颜色变换**: 亮度、对比度、饱和度调整
- **噪声添加**: 高斯噪声、椒盐噪声

### 4.2 高级数据增强

- **AutoAugment**: 自动搜索最优增强策略
- **Mixup**: 混合不同样本生成新样本
- **CutMix**: 随机裁剪和粘贴操作

## 5. 损失函数

### 5.1 交叉熵损失 (Cross Entropy Loss)

标准分类损失函数：

```
L = -Σ(y_i * log(p_i))
```

### 5.2 Focal Loss

解决类别不平衡问题：

```
FL = -α(1-p_t)^γ * log(p_t)
```

其中：
- α: 权重参数
- γ: 聚焦参数
- p_t: 预测概率

### 5.3 标签平滑 (Label Smoothing)

防止过拟合，提高泛化能力：

```
y_smooth = (1-ε) * y + ε/K
```

其中：
- ε: 平滑参数
- K: 类别数

## 6. 优化算法

### 6.1 AdamW优化器

AdamW是Adam的改进版本，更好地处理权重衰减：

```
m_t = β₁ * m_{t-1} + (1-β₁) * g_t
v_t = β₂ * v_{t-1} + (1-β₂) * g_t²
θ_t = θ_{t-1} - lr * m_t / (√v_t + ε) - lr * λ * θ_{t-1}
```

### 6.2 学习率调度

- **余弦退火**: 学习率按余弦函数衰减
- **步长衰减**: 每隔一定epoch降低学习率
- **预热策略**: 训练初期使用较小学习率

## 7. 模型评估

### 7.1 准确率 (Accuracy)

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### 7.2 精确率和召回率

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

### 7.3 混淆矩阵

混淆矩阵展示各类别的预测情况，帮助分析模型性能。

## 8. 模型架构选择

### 8.1 ResNet

- **残差连接**: 解决梯度消失问题
- **深度**: 支持更深的网络结构
- **性能**: 在ImageNet上表现优异

### 8.2 EfficientNet

- **复合缩放**: 同时缩放深度、宽度和分辨率
- **效率**: 在相同计算量下获得更好性能
- **可扩展**: 支持不同规模的模型

### 8.3 模型选择策略

1. **数据量**: 小数据集选择预训练模型
2. **计算资源**: 资源有限选择轻量级模型
3. **精度要求**: 高精度要求选择复杂模型
4. **推理速度**: 实时应用选择快速模型




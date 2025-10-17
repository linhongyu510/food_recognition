# 使用教程

## 1. 环境准备

### 1.1 系统要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.0 (可选，用于GPU加速)

### 1.2 安装依赖

```bash
# 克隆项目
git clone https://github.com/yourusername/food_recognition.git
cd food_recognition

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

### 1.3 数据准备

1. 下载Food-11数据集
2. 解压到项目根目录
3. 确保目录结构如下：

```
food_recognition/
├── data/
│   └── food-11/
│       ├── training/
│       │   ├── labeled/
│       │   └── unlabeled/
│       ├── validation/
│       └── testing/
```

## 2. 快速开始

### 2.1 基础训练

```python
from food_recognition.models import ResNet18
from food_recognition.training import Trainer
from food_recognition.data import FoodDataset, get_train_transforms, get_val_transforms
from food_recognition.utils import set_seed, get_device

# 设置随机种子
set_seed(42)

# 创建数据集
train_dataset = FoodDataset("data/food-11/training/labeled", "train", get_train_transforms())
val_dataset = FoodDataset("data/food-11/validation", "val", get_val_transforms())

# 创建模型
model = ResNet18(num_classes=11, pretrained=True)

# 创建训练器
trainer = Trainer(model, train_dataset, val_dataset, get_device())

# 开始训练
trainer.train(epochs=50, save_path="checkpoints/best_model.pth")
```

### 2.2 半监督学习

```python
from food_recognition.training import SemiSupervisedTrainer

# 创建半监督训练器
trainer = SemiSupervisedTrainer(
    model=model,
    labeled_loader=labeled_loader,
    unlabeled_loader=unlabeled_loader,
    val_loader=val_loader,
    device=device,
    confidence_threshold=0.9
)

# 开始半监督训练
trainer.train_semi_supervised(epochs=100)
```

### 2.3 模型推理

```python
from food_recognition.examples.inference import FoodPredictor

# 创建预测器
predictor = FoodPredictor("checkpoints/best_model.pth", "resnet18")

# 预测单张图片
result = predictor.predict("path/to/image.jpg")
print(f"预测类别: {result['class_name']}")
print(f"置信度: {result['confidence']:.4f}")
```

## 3. 详细使用指南

### 3.1 数据加载

#### 基础数据加载

```python
from food_recognition.data import FoodDataset, get_dataloader

# 创建数据集
dataset = FoodDataset("data/food-11/training/labeled", mode="train")

# 创建数据加载器
dataloader = get_dataloader("data/food-11/training/labeled", "train", batch_size=32)
```

#### 自定义数据变换

```python
from food_recognition.data import get_train_transforms, get_val_transforms
import torchvision.transforms as transforms

# 自定义训练变换
custom_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用自定义变换
dataset = FoodDataset("data/food-11/training/labeled", "train", custom_train_transform)
```

### 3.2 模型选择

#### 预训练模型

```python
from food_recognition.models import ResNet18, ResNet50, EfficientNet_CBAM

# ResNet18
model = ResNet18(num_classes=11, pretrained=True)

# ResNet50
model = ResNet50(num_classes=11, pretrained=True)

# EfficientNet + CBAM
model = EfficientNet_CBAM(num_classes=11, pretrained=True)
```

#### 自定义模型

```python
from food_recognition.models import MyModel

# 自定义CNN模型
model = MyModel(num_classes=11)

# 查看模型参数量
from food_recognition.utils import count_parameters
params = count_parameters(model)
print(f"总参数量: {params['total']}")
print(f"可训练参数量: {params['trainable']}")
```

### 3.3 训练配置

#### 基础训练配置

```python
from food_recognition.training import Trainer
from food_recognition.utils import Config
import torch.optim as optim

# 获取配置
config = Config()

# 创建优化器
optimizer = optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# 创建学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    optimizer=optimizer,
    scheduler=scheduler
)
```

#### 高级训练配置

```python
from food_recognition.training import FocalLoss, LabelSmoothingLoss

# 使用Focal Loss处理类别不平衡
criterion = FocalLoss(alpha=1, gamma=2)

# 使用标签平滑
criterion = LabelSmoothingLoss(num_classes=11, smoothing=0.1)

# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    optimizer=optimizer,
    criterion=criterion
)
```

### 3.4 评估和可视化

#### 模型评估

```python
from food_recognition.training import Accuracy, Precision, Recall, F1Score, ConfusionMatrix

# 创建评估指标
accuracy = Accuracy()
precision = Precision()
recall = Recall()
f1_score = F1Score()
confusion_matrix = ConfusionMatrix()

# 评估模型
model.eval()
with torch.no_grad():
    for data, target in val_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        
        accuracy.update(pred, target)
        precision.update(pred, target)
        recall.update(pred, target)
        f1_score.update(pred, target)
        confusion_matrix.update(pred, target)

# 计算指标
print(f"准确率: {accuracy.compute():.4f}")
print(f"精确率: {precision.compute():.4f}")
print(f"召回率: {recall.compute():.4f}")
print(f"F1分数: {f1_score.compute():.4f}")

# 绘制混淆矩阵
confusion_matrix.plot()
```

#### 可视化分析

```python
from food_recognition.utils import GradCAM, visualize_predictions

# Grad-CAM可视化
gradcam = GradCAM(model, model.layer4[-1])  # 选择目标层
gradcam.visualize(input_tensor, save_path="gradcam.png")

# 预测结果可视化
visualize_predictions(model, val_loader, device, num_samples=9)
```

### 3.5 模型保存和加载

#### 保存模型

```python
from food_recognition.utils import save_checkpoint

# 保存检查点
save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    loss=val_loss,
    accuracy=val_acc,
    filepath="checkpoints/checkpoint.pth"
)

# 保存最佳模型
torch.save(model.state_dict(), "checkpoints/best_model.pth")
```

#### 加载模型

```python
from food_recognition.utils import load_checkpoint

# 加载检查点
checkpoint = load_checkpoint("checkpoints/checkpoint.pth", model, optimizer)
epoch = checkpoint['epoch']
best_acc = checkpoint['accuracy']

# 加载模型权重
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
```

## 4. 高级功能

### 4.1 半监督学习

```python
from food_recognition.training import SemiSupervisedTrainer

# 创建半监督训练器
trainer = SemiSupervisedTrainer(
    model=model,
    labeled_loader=labeled_loader,
    unlabeled_loader=unlabeled_loader,
    val_loader=val_loader,
    device=device,
    confidence_threshold=0.9
)

# 开始半监督训练
trainer.train_semi_supervised(
    epochs=100,
    save_path="checkpoints/semi_supervised_model.pth"
)
```

### 4.2 数据增强

```python
from food_recognition.data import get_albumentations_transforms

# 使用Albumentations增强
transform = get_albumentations_transforms(image_size=224)

# 创建数据集
dataset = FoodDataset("data/food-11/training/labeled", "train", transform)
```

### 4.3 模型集成

```python
# 创建多个模型
models = [
    ResNet18(num_classes=11, pretrained=True),
    ResNet50(num_classes=11, pretrained=True),
    EfficientNet_CBAM(num_classes=11, pretrained=True)
]

# 加载权重
for i, model in enumerate(models):
    model.load_state_dict(torch.load(f"checkpoints/model_{i}.pth"))

# 集成预测
def ensemble_predict(models, input_tensor):
    predictions = []
    for model in models:
        with torch.no_grad():
            output = model(input_tensor)
            predictions.append(F.softmax(output, dim=1))
    
    # 平均预测结果
    ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
    return ensemble_pred.argmax(dim=1)
```

## 5. 常见问题

### 5.1 内存不足

```python
# 减少批次大小
config.batch_size = 16

# 使用梯度累积
accumulation_steps = 4
for i, (data, target) in enumerate(train_loader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 5.2 训练速度慢

```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in train_loader:
    optimizer.zero_grad()
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 5.3 过拟合

```python
# 使用正则化
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# 使用Dropout
model = nn.Sequential(
    nn.Linear(512, 256),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(256, 11)
)

# 使用早停
early_stopping = EarlyStopping(patience=10, min_delta=0.001)
```

## 6. 性能优化

### 6.1 模型优化

```python
# 模型量化
import torch.quantization as quantization

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# 静态量化
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
quantized_model = torch.quantization.prepare(model)
quantized_model = torch.quantization.convert(quantized_model)
```

### 6.2 推理优化

```python
# 模型编译
model = torch.jit.script(model)

# 使用ONNX
import torch.onnx

torch.onnx.export(
    model, input_tensor, "model.onnx",
    export_params=True, opset_version=11
)
```

## 7. 部署指南

### 7.1 本地部署

```python
# 创建预测服务
class FoodRecognitionService:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)
        self.transform = get_val_transforms()
    
    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0][prediction].item()
        
        return prediction, confidence
```

### 7.2 云端部署

```python
# 使用Flask创建API
from flask import Flask, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)
service = FoodRecognitionService("checkpoints/best_model.pth")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = base64.b64decode(data['image'])
    image = Image.open(BytesIO(image_data))
    
    prediction, confidence = service.predict(image)
    
    return jsonify({
        'prediction': prediction,
        'confidence': confidence
    })
```

## 8. 贡献指南

### 8.1 代码规范

- 使用PEP 8代码风格
- 添加类型注解
- 编写文档字符串
- 添加单元测试

### 8.2 提交规范

```bash
# 功能开发
git commit -m "feat: add new model architecture"

# 修复bug
git commit -m "fix: resolve memory leak in training"

# 文档更新
git commit -m "docs: update tutorial with new examples"
```

### 8.3 测试

```python
# 单元测试
import unittest

class TestFoodDataset(unittest.TestCase):
    def test_dataset_creation(self):
        dataset = FoodDataset("data/food-11/training/labeled", "train")
        self.assertGreater(len(dataset), 0)
    
    def test_data_loading(self):
        dataset = FoodDataset("data/food-11/training/labeled", "train")
        image, label = dataset[0]
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIsInstance(label, torch.Tensor)
```

## 9. 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持ResNet、AlexNet、EfficientNet模型
- 实现CBAM注意力机制
- 支持半监督学习
- 提供完整的训练和推理流程

### v1.1.0 (2024-01-15)
- 添加更多数据增强方法
- 支持模型集成
- 优化训练速度
- 添加可视化工具

### v1.2.0 (2024-02-01)
- 支持模型量化和优化
- 添加部署指南
- 完善文档和教程
- 修复已知问题




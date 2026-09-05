```python
import random
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 固定随机种子函数
# 作用：确保实验的可重复性，设置PyTorch、Python的随机种子以及相关配置
def seed_everything(seed):
    torch.manual_seed(seed)  # 设置CPU随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU随机种子
    torch.backends.cudnn.benchmark = False  # 关闭cuDNN的自动调优功能，确保每次运行结果一致
    torch.backends.cudnn.deterministic = True  # 使cuDNN操作具有确定性
    random.seed(seed)  # 设置Python内置的random模块的随机种子
    np.random.seed(seed)  # 设置NumPy的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python的哈希种子

seed_everything(0)  # 固定随机种子为0

HW = 224  # 定义图片尺寸为224x224

# 训练数据增强操作
# 包含随机裁剪、随机旋转和转换为张量
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(HW),  # 随机裁剪并调整大小为HWxHW
    transforms.RandomRotation(50),  # 随机旋转图片，角度范围为[-50, 50]
    transforms.ToTensor()  # 将图片转换为张量
])

# 验证数据增强操作
# 仅包含调整大小和转换为张量
val_transform = transforms.Compose([
    transforms.Resize((HW, HW)),  # 调整图片大小为HWxHW
    transforms.ToTensor()  # 将图片转换为张量
])

# 自定义数据集类
class FoodDataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode  # 数据集模式，如"train"、"val"、"semi"
        self.image_paths = []  # 存储图片路径
        self.labels = []  # 存储图片标签

        if mode == "semi":  # 如果是半监督模式
            self.image_paths = [os.path.join(path, img) for img in os.listdir(path)]  # 获取路径下所有图片路径
        else:  # 否则是有监督模式
            for label in range(11):  # 遍历11个类别
                folder = os.path.join(path, f"{label:02d}")  # 构建类别文件夹路径
                for img in os.listdir(folder):  # 遍历类别文件夹下的所有图片
                    self.image_paths.append(os.path.join(folder, img))  # 添加图片路径
                    self.labels.append(label)  # 添加图片标签
            self.labels = torch.LongTensor(self.labels)  # 将标签转换为LongTensor

        self.transform = train_transform if mode == "train" else val_transform  # 根据模式选择数据增强方式

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")  # 打开图片并转换为RGB格式
        image = self.transform(image)  # 应用数据增强

        if self.mode == "semi":  # 如果是半监督模式
            return image, self.image_paths[idx]  # 返回图片和图片路径
        else:  # 否则是有监督模式
            return image, self.labels[idx]  # 返回图片和标签

    def __len__(self):
        return len(self.image_paths)  # 返回数据集大小


# CBAM模块（卷积块注意力模块）
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 自适应平均池化，将特征图大小变为1x1
            nn.Conv2d(channels, channels // reduction, 1, bias=False),  # 1x1卷积降维
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(channels // reduction, channels, 1, bias=False),  # 1x1卷积升维
            nn.Sigmoid()  # Sigmoid激活函数，输出通道注意力权重
        )
        # 空间注意力机制
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),  # 卷积操作，输入通道为2，输出通道为1，卷积核大小为7x7
            nn.Sigmoid()  # Sigmoid激活函数，输出空间注意力权重
        )

    def forward(self, x):
        # 通道注意力
        x_channel = self.channel_att(x) * x  # 将通道注意力权重与输入特征图相乘
        # 空间注意力
        x_spatial = torch.cat([torch.mean(x_channel, dim=1, keepdim=True),
                               torch.max(x_channel, dim=1, keepdim=True)[0]], dim=1)  # 拼接平均池化和最大池化结果
        x_spatial = self.spatial_att(x_spatial) * x_channel  # 将空间注意力权重与通道注意力后的特征图相乘
        return x_spatial


# EfficientNet-B0 + CBAM模型
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_CBAM, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)  # 加载预训练的EfficientNet-B0模型
        self.cbam = CBAM(1280)  # 初始化CBAM模块，EfficientNet-B0最后特征通道数是1280
        self.fc = nn.Linear(1280, num_classes)  # 全连接层，输出类别数

    def forward(self, x):
        x = self.backbone.features(x)  # 获取EfficientNet-B0的特征图
        x = self.cbam(x)  # 应用CBAM模块
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # 自适应平均池化并展平特征图
        x = self.fc(x)  # 全连接层进行分类
        return x


# Grad-CAM可视化函数
# 作用：生成Grad-CAM可视化图，显示模型关注的区域
def generate_grad_cam(model, image_tensor, target_layer):
    model.eval()  # 设置模型为评估模式

    def forward_hook(module, input, output):
        feature_maps.append(output)  # 保存前向传播的特征图

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])  # 保存反向传播的梯度

    feature_maps = []  # 存储特征图
    gradients = []  # 存储梯度

    hook_forward = target_layer.register_forward_hook(forward_hook)  # 注册前向传播钩子
    hook_backward = target_layer.register_backward_hook(backward_hook)  # 注册反向传播钩子

    image_tensor = image_tensor.unsqueeze(0).cuda()  # 添加批次维度并移动到GPU
    output = model(image_tensor)  # 前向传播
    class_idx = output.argmax().item()  # 获取预测的类别索引

    model.zero_grad()  # 清空梯度
    output[:, class_idx].backward()  # 反向传播计算梯度

    hook_forward.remove()  # 移除前向传播钩子
    hook_backward.remove()  # 移除反向传播钩子

    grad = gradients[0].cpu().data.numpy()[0]  # 将梯度转换为numpy数组并移动到CPU
    fmap = feature_maps[0].cpu().data.numpy()[0]  # 将特征图转换为numpy数组并移动到CPU

    weights = np.mean(grad, axis=(1, 2))  # 计算梯度的平均值，作为每个特征图的权重
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)  # 初始化Grad-CAM图

    for i, w in enumerate(weights):
        cam += w * fmap[i]  # 计算Grad-CAM图

    cam = np.maximum(cam, 0)  # 应用ReLU激活函数
    cam = cv2.resize(cam, (HW, HW))  # 调整Grad-CAM图大小
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化Grad-CAM图

    return cam


# 训练与验证函数
def train_model(model, train_loader, val_loader, device, epochs, optimizer, loss_fn, save_path):
    model.to(device)  # 将模型移动到指定设备（如GPU）
    best_acc = 0  # 记录最佳验证准确率

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        train_loss, train_acc = 0, 0  # 初始化训练损失和准确率

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备
            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播
            loss = loss_fn(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            train_loss += loss.item()  # 累加训练损失
            train_acc += (outputs.argmax(1) == labels).sum().item()  # 累加正确预测的数量

        train_acc /= len(train_loader.dataset)  # 计算训练准确率
        val_acc = evaluate_model(model, val_loader, device)  # 计算验证准确率

        if val_acc > best_acc:  # 如果验证准确率高于当前最佳准确率
            torch.save(model.state_dict(), save_path)  # 保存模型参数
            best_acc = val_acc  # 更新最佳准确率

        print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")  # 打印训练和验证准确率


# 评估函数
def evaluate_model(model, val_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化正确预测的数量

    with torch.no_grad():  # 不计算梯度
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到指定设备
            outputs = model(images)  # 前向传播
            correct += (outputs.argmax(1) == labels).sum().item()  # 累加正确预测的数量

    return correct / len(val_loader.dataset)  # 计算验证准确率


# 训练部分
train_set = FoodDataset("food-11/training/labeled", "train")  # 创建训练数据集
val_set = FoodDataset("food-11/validation", "val")  # 创建验证数据集
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)  # 创建训练数据加载器
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)  # 创建验证数据加载器

device = torch.device("cuda:0")  # 使用GPU 0，若没有GPU则使用CPU
model = EfficientNet_CBAM().to(device)  # 初始化模型并移动到指定设备
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # 初始化优化器
loss_fn = nn.CrossEntropyLoss()  # 初始化损失函数
train_model(model, train_loader, val_loader, device, 50, optimizer, loss_fn, "best_model.pth")  # 训练模型
```


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

# 固定随机种子
def seed_everything(seed):
    torch.manual_seed(seed)  # 设置CPU的随机种子
    torch.cuda.manual_seed(seed)  # 设置当前GPU的随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有GPU的随机种子
    torch.backends.cudnn.benchmark = False  # 关闭cuDNN的自动寻找最优算法模式，保证结果可复现
    torch.backends.cudnn.deterministic = True  # 使cuDNN的行为固定，保证结果可复现
    random.seed(seed)  # 设置Python内置的random库的随机种子
    np.random.seed(seed)  # 设置numpy库的随机种子
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python的哈希种子，保证哈希结果可复现

seed_everything(0)

HW = 224  # 定义图片尺寸

# 数据增强函数
def mosaic_augmentation(image, size=HW):
    # 获取原图的宽高
    w, h = image.size
    # 创建一个全黑的空白图像，大小为指定的size
    mosaic_image = Image.new('RGB', (size, size), (0, 0, 0))

    # 随机选择 4 张图片块位置
    boxes = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
        (w // 2, h // 2, w, h)
    ]

    for box in boxes:
        # 从原图中裁剪出图片块
        patch = image.crop(box)
        # 将图片块调整为指定大小的一半
        patch = patch.resize((size // 2, size // 2))
        # 将调整后的图片块粘贴到空白图像的相应位置
        mosaic_image.paste(patch, (box[0] * size // w, box[1] * size // h))

    return mosaic_image

def random_occlusion(image, size=HW):
    # 随机遮挡：用黑色矩形遮挡图像中的一部分
    w, h = image.size
    # 随机生成遮挡矩形的宽度
    occlusion_width = random.randint(40, 100)
    # 随机生成遮挡矩形的高度
    occlusion_height = random.randint(40, 100)
    # 随机生成遮挡矩形左上角的x坐标偏移量
    x_offset = random.randint(0, w - occlusion_width)
    # 随机生成遮挡矩形左上角的y坐标偏移量
    y_offset = random.randint(0, h - occlusion_height)
    
    # 在图像上粘贴一个黑色矩形进行遮挡
    image.paste((0, 0, 0), [x_offset, y_offset, x_offset + occlusion_width, y_offset + occlusion_height])
    return image

# 数据增强
train_transform = transforms.Compose([
    # 随机裁剪并调整大小为HW
    transforms.RandomResizedCrop(HW),
    # 随机旋转图像，旋转角度范围是[-50, 50]度
    transforms.RandomRotation(50),
    # 以0.5的概率应用Mosaic增强
    transforms.Lambda(lambda img: mosaic_augmentation(img) if random.random() > 0.5 else img),  
    # 以0.5的概率应用随机遮挡增强
    transforms.Lambda(lambda img: random_occlusion(img) if random.random() > 0.5 else img),  
    # 动态调整图像的亮度、对比度、饱和度和色调
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
    # 将图像转换为PyTorch的张量格式
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    # 将图像调整为指定大小HW x HW
    transforms.Resize((HW, HW)),
    # 将图像转换为PyTorch的张量格式
    transforms.ToTensor()
])

# 自定义数据集
class FoodDataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        # 如果是半监督学习模式（semi）
        if mode == "semi":
            # 将路径下的所有图像路径添加到image_paths列表中
            self.image_paths = [os.path.join(path, img) for img in os.listdir(path)]
        else:
            # 对于每个标签（0到10）
            for label in range(11):
                # 构建标签对应的文件夹路径
                folder = os.path.join(path, f"{label:02d}")
                # 遍历文件夹中的所有图像文件
                for img in os.listdir(folder):
                    # 将图像路径添加到image_paths列表中
                    self.image_paths.append(os.path.join(folder, img))
                    # 将标签添加到labels列表中
                    self.labels.append(label)
            # 将labels列表转换为PyTorch的LongTensor类型
            self.labels = torch.LongTensor(self.labels)

        # 根据不同模式选择不同的数据增强方式
        self.transform = train_transform if mode == "train" else val_transform

    def __getitem__(self, idx):
        # 打开图像并转换为RGB模式
        image = Image.open(self.image_paths[idx]).convert("RGB")
        # 对图像应用数据增强和转换
        image = self.transform(image)
        
        # 如果是半监督学习模式，返回图像和图像路径
        if self.mode == "semi":
            return image, self.image_paths[idx]
        else:
            # 否则返回图像和对应的标签
            return image, self.labels[idx]

    def __len__(self):
        # 返回数据集中图像的数量
        return len(self.image_paths)


# CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力模块
        self.channel_att = nn.Sequential(
            # 全局平均池化，将特征图大小变为1x1
            nn.AdaptiveAvgPool2d(1),
            # 卷积层，减少通道数为原来的1/reduction
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            # ReLU激活函数
            nn.ReLU(),
            # 卷积层，恢复通道数为原来的数量
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            # Sigmoid激活函数，生成通道注意力权重
            nn.Sigmoid()
        )
        # 空间注意力模块
        self.spatial_att = nn.Sequential(
            # 卷积层，输入通道数为2（平均池化和最大池化结果的拼接），输出通道数为1
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            # Sigmoid激活函数，生成空间注意力权重
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力模块的计算
        x_channel = self.channel_att(x) * x
        # 空间注意力模块的输入，拼接平均池化和最大池化的结果
        x_spatial = torch.cat([torch.mean(x_channel, dim=1, keepdim=True),
                               torch.max(x_channel, dim=1, keepdim=True)[0]], dim=1)
        # 空间注意力模块的计算
        x_spatial = self.spatial_att(x_spatial) * x_channel
        return x_spatial


# EfficientNet-B0 + CBAM
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_CBAM, self).__init__()
        # 使用预训练的EfficientNet-B0模型作为骨干网络
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        # 添加CBAM模块，输入通道数为EfficientNet-B0最后特征的通道数1280
        self.cbam = CBAM(1280)  
        # 全连接层，将特征映射到指定的类别数
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        # 获取EfficientNet-B0的特征图
        x = self.backbone.features(x)  
        # 将特征图通过CBAM模块进行处理
        x = self.cbam(x)  
        # 对特征图进行自适应平均池化，将其展平为一维向量
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  
        # 通过全连接层进行分类
        x = self.fc(x)  
        return x


# Grad-CAM 可视化
def generate_grad_cam(model, image_tensor, target_layer):
    model.eval()  # 设置模型为评估模式
    
    def forward_hook(module, input, output):
        # 前向传播钩子函数，将目标层的输出特征图保存到feature_maps列表中
        feature_maps.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        # 反向传播钩子函数，将目标层的梯度保存到gradients列表中
        gradients.append(grad_out[0])
    
    feature_maps = []
    gradients = []

    # 注册前向传播钩子
    hook_forward = target_layer.register_forward_hook(forward_hook)
    # 注册反向传播钩子
    hook_backward = target_layer.register_backward_hook(backward_hook)

    # 将图像张量添加一个维度并移动到GPU上
    image_tensor = image_tensor.unsqueeze(0).cuda()
    # 模型前向传播得到输出
    output = model(image_tensor)
    # 获取预测的类别索引
    class_idx = output.argmax().item()

    model.zero_grad()  # 清空模型的梯度
    # 计算指定类别的梯度
    output[:, class_idx].backward()

    hook_forward.remove()  # 移除前向传播钩子
    hook_backward.remove()  # 移除反向传播钩子

    # 将梯度转换为numpy数组并取出第一个样本的梯度
    grad = gradients[0].cpu().data.numpy()[0]
    # 将特征图转换为numpy数组并取出第一个样本的特征图
    fmap = feature_maps[0].cpu().data.numpy()[0]

    # 计算每个通道的权重，取梯度在空间维度上的平均值
    weights = np.mean(grad, axis=(1, 2))  
    # 初始化CAM（Class Activation Map）为全零数组
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        # 计算CAM，将每个通道的特征图乘以对应的权重并累加
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)  # 应用ReLU函数，将负值设为0
    # 将CAM调整为指定大小
    cam = cv2.resize(cam, (HW, HW))
    # 对CAM进行归一化处理
    cam = (cam - cam.min()) / (cam.max() - cam.min())  

    return cam


# 训练与验证
def train_model(model, train_loader, val_loader, device, epochs, optimizer, loss_fn, save_path):
    model.to(device)  # 将模型移动到指定设备（GPU或CPU）上
    best_acc = 0  # 记录最佳验证准确率

    for epoch in range(epochs):
        model.train()  # 设置模型为训练模式
        train_loss, train_acc = 0, 0  # 初始化训练损失和训练准确率

        for images, labels in train_loader:
            # 将图像和标签移动到指定设备上
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # 清空优化器的梯度
            outputs = model(images)  # 模型前向传播得到输出
            loss = loss_fn(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数
            
            train_loss += loss.item()  # 累加训练损失
            train_acc += (outputs.argmax(1) == labels).sum().item()  # 累加正确预测的数量

        train_acc /= len(train_loader.dataset)  # 计算平均训练准确率
        val_acc = evaluate_model(model, val_loader, device)  # 计算验证准确率

        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)  # 保存最佳模型参数
            best_acc = val_acc

        print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# 评估
def evaluate_model(model, val_loader, device):
    model.eval()  # 设置模型为评估模式
    correct = 0  # 初始化正确预测的数量

    with torch.no_grad():  # 不计算梯度
        for images, labels in val_loader:
            # 将图像和标签移动到指定设备上
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 模型前向传播得到输出
            correct += (outputs.argmax(1) == labels).sum().item()  # 累加正确预测的数量
    
    return correct / len(val_loader.dataset)  # 计算验证准确率


# 训练
train_set = FoodDataset("food-11/training/labeled", "train")  # 创建训练数据集
val_set = FoodDataset("food-11/validation", "val")  # 创建验证数据集
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)  # 创建训练数据加载器
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)  # 创建验证数据加载器

device = torch.device("cuda:0")  # 指定设备为GPU 0
model = EfficientNet_CBAM().to(device)  # 创建模型并移动到指定设备上
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # 创建优化器
loss_fn = nn.CrossEntropyLoss()  # 创建损失函数
train_model(model, train_loader, val_loader, device, 50, optimizer, loss_fn, "best_model.pth")  # 训练模型
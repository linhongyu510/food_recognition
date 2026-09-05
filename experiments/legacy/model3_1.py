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
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(0)

HW = 224  # 图片尺寸

# 数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(HW),
    transforms.RandomRotation(50),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((HW, HW)),
    transforms.ToTensor()
])

# 自定义数据集
class FoodDataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode
        self.image_paths = []
        self.labels = []
        
        if mode == "semi":
            self.image_paths = [os.path.join(path, img) for img in os.listdir(path)]
        else:
            for label in range(11):
                folder = os.path.join(path, f"{label:02d}")
                for img in os.listdir(folder):
                    self.image_paths.append(os.path.join(folder, img))
                    self.labels.append(label)
            self.labels = torch.LongTensor(self.labels)

        self.transform = train_transform if mode == "train" else val_transform

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        
        if self.mode == "semi":
            return image, self.image_paths[idx]
        else:
            return image, self.labels[idx]

    def __len__(self):
        return len(self.image_paths)


# CBAM 模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        x_channel = self.channel_att(x) * x
        # 空间注意力
        x_spatial = torch.cat([torch.mean(x_channel, dim=1, keepdim=True),
                               torch.max(x_channel, dim=1, keepdim=True)[0]], dim=1)
        x_spatial = self.spatial_att(x_spatial) * x_channel
        return x_spatial


# EfficientNet-B0 + CBAM
class EfficientNet_CBAM(nn.Module):
    def __init__(self, num_classes=11):
        super(EfficientNet_CBAM, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.cbam = CBAM(1280)  # EfficientNet-B0 最后特征通道数是 1280
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.backbone.features(x)  # 获取 EfficientNet-B0 特征
        x = self.cbam(x)  # CBAM 处理
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)  # 平均池化
        x = self.fc(x)  # 分类
        return x


# Grad-CAM 可视化
def generate_grad_cam(model, image_tensor, target_layer):
    model.eval()
    
    def forward_hook(module, input, output):
        feature_maps.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    feature_maps = []
    gradients = []

    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_backward_hook(backward_hook)

    image_tensor = image_tensor.unsqueeze(0).cuda()
    output = model(image_tensor)
    class_idx = output.argmax().item()

    model.zero_grad()
    output[:, class_idx].backward()

    hook_forward.remove()
    hook_backward.remove()

    grad = gradients[0].cpu().data.numpy()[0]
    fmap = feature_maps[0].cpu().data.numpy()[0]

    weights = np.mean(grad, axis=(1, 2))  
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * fmap[i]

    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (HW, HW))
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # 归一化

    return cam


# 训练与验证
def train_model(model, train_loader, val_loader, device, epochs, optimizer, loss_fn, save_path):
    model.to(device)
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()

        train_acc /= len(train_loader.dataset)
        val_acc = evaluate_model(model, val_loader, device)

        if val_acc > best_acc:
            torch.save(model.state_dict(), save_path)
            best_acc = val_acc

        print(f"Epoch [{epoch+1}/{epochs}] - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# 评估
def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
    
    return correct / len(val_loader.dataset)


# 训练
train_set = FoodDataset("food-11/training/labeled", "train")
val_set = FoodDataset("food-11/validation", "val")
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

device = torch.device("cuda:0")
model = EfficientNet_CBAM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()
train_model(model, train_loader, val_loader, device, 50, optimizer, loss_fn, "best_model.pth")

import os
import torch
import torch.utils.data
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 自定义 UECFOOD256 数据集类
class UECFoodDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        """
        参数:
          root: 数据集根目录，目录下应包含类别子文件夹，存放图片。
          transforms: 图像预处理变换（例如 ToTensor）。
        """
        self.root = root
        self.transforms = transforms
        self.imgs = []
        self.labels = []
        
        # 遍历所有类别子文件夹并加载图片
        for class_folder in sorted(os.listdir(root)):
            class_folder_path = os.path.join(root, class_folder)
            if os.path.isdir(class_folder_path):
                for img_name in os.listdir(class_folder_path):
                    if img_name.endswith('.jpg') or img_name.endswith('.png'):  # 假设图片是 .jpg 或 .png 格式
                        self.imgs.append(os.path.join(class_folder_path, img_name))
                        self.labels.append(int(class_folder))  # 类别号作为标签
        
    def __getitem__(self, idx):
        # 加载图片
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert("RGB")
        
        # 生成虚拟标注：在图片中央放一个随机大小的框
        width, height = img.size
        xmin = random.randint(0, width // 2)
        ymin = random.randint(0, height // 2)
        xmax = random.randint(width // 2, width)
        ymax = random.randint(height // 2, height)
        
        # 标签是当前类的索引
        label = self.labels[idx]

        boxes = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
        labels = torch.tensor([label], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms is not None:
            img = self.transforms(image=np.array(img))['image']
            
        return img, target

    def __len__(self):
        return len(self.imgs)

# 图像预处理变换：引入 Albumentations 进行动态光照调整与仿射变换
def get_transform():
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.2),  # 随机光照调整
        A.Affine(scale=(0.8, 1.2), rotate=(-20, 20), p=0.2),  # 随机仿射变换
        A.HorizontalFlip(p=0.5),  # 水平翻转
        A.RandomSizedCrop(min_max_height=(200, 400), height=300, width=300, p=0.3),  # 随机裁剪
        ToTensorV2()  # 转换为 Tensor
    ])
    return transform

# 构造 Faster R-CNN 模型，并替换分类头
def get_model(num_classes):
    # 使用 COCO 预训练的 Faster R-CNN 模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # 替换分类头，输入特征数保持不变
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    # 数据集路径，请确保你已经下载并整理好数据集
    dataset_path = './data/UECFOOD256'  # 修改为相对路径
    if not os.path.exists(dataset_path):
        print("数据集未找到，请下载 UECFOOD256 数据集并放置在目录：", dataset_path)
        return

    # 定义预处理变换
    transforms = get_transform()
    # 创建数据集实例
    dataset = UECFoodDataset(dataset_path, transforms=transforms)
    
    # 划分训练集和测试集（80% 训练，20% 测试）
    indices = torch.randperm(len(dataset)).tolist()
    train_split = int(0.8 * len(dataset))
    dataset_train = torch.utils.data.Subset(dataset, indices[:train_split])
    dataset_test = torch.utils.data.Subset(dataset, indices[train_split:])
    
    # DataLoader：设置 collate_fn 保证 batch 内元素格式一致
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # 设置类别数：
    # UECFOOD256 包含 256 个菜品类别，背景为 0，因此总类别数 = 256 + 1 = 257
    num_classes = 257
    model = get_model(num_classes)
    
    # 设备配置（优先使用 GPU）
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    num_epochs = 10  # 根据数据量调整训练轮数
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        lr_scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
        
        # 简单测试：打印一个 batch 的预测结果
        model.eval()
        with torch.no_grad():
            for images, _ in data_loader_test:
                images = list(img.to(device) for img in images)
                outputs = model(images)
                print("预测结果：", outputs)
                break

    # 保存训练好的模型
    torch.save(model.state_dict(), "uecfood256_detection_model.pth")
    print("训练完成，模型已保存。")

if __name__ == '__main__':
    main()

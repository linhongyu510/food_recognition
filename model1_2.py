import random
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image  # 读取图片数据
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torchvision import transforms, models
from torch.optim.lr_scheduler import StepLR
import time
import matplotlib.pyplot as plt
from torchvision.models import VGG11_BN_Weights


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 设置随机种子，确保每次结果一致
seed_everything(0)

HW = 224

# 数据增强
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(50),
        transforms.ToTensor()
    ]
)

val_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor()
    ]
)


class food_Dataset(Dataset):
    def __init__(self, path, mode="train"):
        self.mode = mode
        if mode == "semi":
            self.X = self.read_file(path)
        else:
            self.X, self.Y = self.read_file(path)
            self.Y = torch.LongTensor(self.Y)  # 标签转为长整形

        if mode == "train":
            self.transform = train_transform
        else:
            self.transform = val_transform

    def read_file(self, path):
        if self.mode == "semi":
            file_list = os.listdir(path)
            xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
            for j, img_name in enumerate(file_list):
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path)
                img = img.resize((HW, HW))
                xi[j, ...] = img
            print(f"读到了{len(xi)}个数据")
            return xi
        else:
            for i in tqdm(range(11)):
                file_dir = path + f"/%02d" % i
                file_list = os.listdir(file_dir)

                xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)
                yi = np.zeros(len(file_list), dtype=np.uint8)

                for j, img_name in enumerate(file_list):
                    img_path = os.path.join(file_dir, img_name)
                    img = Image.open(img_path)
                    img = img.resize((HW, HW))
                    xi[j, ...] = img
                    yi[j] = i

                if i == 0:
                    X = xi
                    Y = yi
                else:
                    X = np.concatenate((X, xi), axis=0)
                    Y = np.concatenate((Y, yi), axis=0)
            print(f"读到了{len(Y)}个数据")
            return X, Y

    def __getitem__(self, item):
        if self.mode == "semi":
            return self.transform(self.X[item]), self.X[item]
        else:
            return self.transform(self.X[item]), self.Y[item]

    def __len__(self):
        return len(self.X)


class myModel(nn.Module):
    def __init__(self, num_class):
        super(myModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)    # 64*224*224
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)   # 64*112*112

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),    # 128*112*112
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 128*56*56
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 256*28*28
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 512*14*14
        )

        self.pool2 = nn.MaxPool2d(2)    # 512*7*7
        self.fc1 = nn.Linear(25088, 1000)   # 25088->1000
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(1000, num_class)  # 1000-11

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        return x


def train_val(model, train_loader, val_loader, no_label_loader, device, epochs, optimizer, loss, thres, save_path):
    model = model.to(device)
    max_acc = 0.0
    patience = 60
    decrease_lr_patience = 30
    best_loss = float('inf')
    epoch_no_improvement = 0
    epoch_lr_decrease = 0

    plt_train_loss = []
    plt_val_loss = []
    plt_train_acc = []
    plt_val_acc = []

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0

        start_time = time.time()

        model.train()
        for batch_x, batch_y in train_loader:
            x, target = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            train_bat_loss = loss(pred, target)
            train_bat_loss.backward()
            optimizer.step()
            train_loss += train_bat_loss.cpu().item()
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        plt_train_loss.append(train_loss / len(train_loader))
        plt_train_acc.append(train_acc / len(train_loader.dataset))

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                x, target = batch_x.to(device), batch_y.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target)
                val_loss += val_bat_loss.cpu().item()
                val_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

        plt_val_loss.append(val_loss / len(val_loader.dataset))
        plt_val_acc.append(val_acc / len(val_loader.dataset))

        if epoch % 3 == 0 and plt_val_acc[-1] > 0.6:
            pass  # 可以在这里添加半监督训练的代码

        if val_acc > max_acc:
            torch.save(model, save_path)
            max_acc = val_acc
            epoch_no_improvement = 0
        else:
            epoch_no_improvement += 1

        if epoch_no_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch_no_improvement >= decrease_lr_patience:
            scheduler.step()

        print(f"[{epoch}/{epochs}] {time.time() - start_time:.2f}s "
              f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader.dataset):.4f} "
              f"Train Acc: {train_acc/len(train_loader.dataset):.4f} | Val Acc: {val_acc/len(val_loader.dataset):.4f}")

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("Loss")
    plt.legend(["Train", "Val"])
    plt.show()

    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("Accuracy")
    plt.legend(["Train", "Val"])
    plt.show()


# 数据路径
train_path = "food-11/training/labeled"
val_path = "food-11/validation"
no_label_path = "food-11_sample/training/unlabeled/00"

train_set = food_Dataset(train_path, "train")
val_set = food_Dataset(val_path, "val")
no_label_set = food_Dataset(no_label_path, "semi")

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader = DataLoader(val_set, batch_size=16, shuffle=True)
no_label_loader = DataLoader(no_label_set, batch_size=16, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.vgg11_bn(weights=VGG11_BN_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(4096, 11)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss = nn.CrossEntropyLoss()

train_val(model, train_loader, val_loader, no_label_loader, device, epochs=50, optimizer=optimizer, loss=loss,
          thres=0.6, save_path="model_best.pth")

from torchvision.datasets import FashionMNIST, CIFAR10, MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np


train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.RandomResizedCrop(HW),
    # transforms.RandomHorizontalFlip(),
    # autoaugment.AutoAugment(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])                   # 训练集需要做各种变换。   效果参见https://pytorch.org/vision/stable/transforms.html



train_set = FashionMNIST(root="FashionMnist", train=True, download=False, transform=train_transform)  #数据集名字， 训练集， 第一次downloadtrue

test_set = FashionMNIST(root="FashionMnist", train=False, download=False, transform=train_transform)

train_loader = DataLoader(train_set, batch_size=16)

rows = 4
cols = 4
for batch in train_loader:
    print(batch)
    fig2, axs2 = plt.subplots(nrows=rows,ncols=cols,squeeze=False)

    for i in range(rows):
        for j in range(cols):
            ax = axs2[i, j]
            ax.imshow(np.array(batch[0][i*4+j].permute(1,2,0)))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            ax.set_title(batch[1][i*4+j].numpy())
    plt.show()
    plt.tight_layout()







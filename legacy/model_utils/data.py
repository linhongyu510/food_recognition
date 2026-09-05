import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
from sklearn.model_selection import train_test_split
import os
import cv2
from torchvision.transforms import transforms,autoaugment
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt

HW = 224
imagenet_norm = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

test_transform = transforms.Compose([
    transforms.ToTensor(),
])              # 测试集只需要转为张量

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(HW),
    transforms.RandomHorizontalFlip(),
    autoaugment.AutoAugment(),
    transforms.ToTensor(),
    # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])                   # 训练集需要做各种变换。   效果参见https://pytorch.org/vision/stable/transforms.html



class foodDataset(Dataset):                      #数据集三要素： init ， getitem ， len
    def __init__(self, path, mode):
        y = None
        self.transform = None
        self.mode = mode

        pathDict = {'train':'training/labeled','train_unl':'training/unlabeled', 'val':'validation', 'test':'testing'}
        imgPaths = path +'/'+ pathDict[mode]                       # 定义路径

        if mode == 'test':
            x = self._readfile(imgPaths,label=False)
            self.transform = test_transform                         #从文件读数据,测试机和无标签数据没有标签， trans方式也不一样
        elif mode == 'train':
            x, y =self._readfile(imgPaths,label=True)
            self.transform = train_transform
        elif mode == 'val':
            x, y =self._readfile(imgPaths,label=True)
            self.transform = test_transform
        elif mode == 'train_unl':
            x = self._readfile(imgPaths,label=False)
            self.transform = train_transform

        if y is not None:                                    # 注意， 分类的标签必须转换为长整型： int64.
            y = torch.LongTensor(y)
        self.x, self.y = x, y

    def __getitem__(self, index):                        # getitem 用于根据标签取数据
        orix = self.x[index]                              # 取index的图片

        if self.transform == None:
            xT = torch.tensor(orix).float()
        else:
            xT = self.transform(orix)                     # 如果规定了transformer， 则需要transf

        if self.y is not None:                       # 有标签， 则需要返回标签。 这里额外返回了原图， 方便后面画图。
            y = self.y[index]
            return xT, y, orix
        else:
            return xT, orix

    def _readfile(self,path, label=True):                   # 定义一个读文件的函数
        if label:                                             # 有无标签， 文件结构是不一样的。
            x, y = [], []
            for i in tqdm(range(11)):                           # 有11类
                label = '/%02d/'%i                                 # %02必须为两位。 符合文件夹名字
                imgDirpath = path+label
                imglist = os.listdir(imgDirpath)                    # listdir 可以列出文件夹下所有文件。
                xi = np.zeros((len(imglist), HW, HW, 3), dtype=np.uint8)
                yi = np.zeros((len(imglist)), dtype=np.uint8)           # 先把放数据的格子打好。 x的维度是 照片数量*H*W*3
                for j, each in enumerate(imglist):
                    imgpath = imgDirpath + each
                    img = Image.open(imgpath)                  # 用image函数读入照片， 并且resize。
                    img = img.resize((HW, HW))
                    xi[j,...] = img                           #在第j个位置放上数据和标签。
                    yi[j] = i
                if i == 0:
                    x = xi
                    y = yi
                else:
                    x = np.concatenate((x, xi), axis=0)             # 将11个文件夹的数据合在一起。
                    y = np.concatenate((y, yi), axis=0)
            print('读入有标签数据%d个 '%len(x))
            return x, y
        else:
            imgDirpath = path + '/00/'
            imgList = os.listdir(imgDirpath)
            x = np.zeros((len(imgList), HW, HW ,3),dtype=np.uint8)
            for i, each in enumerate(imgList):
                imgpath = imgDirpath + each
                img = Image.open(imgpath)
                img = img.resize((HW, HW))
                x[i,...] = img
            return x

    def __len__(self):                      # len函数 负责返回长度。
        return len(self.x)

class noLabDataset(Dataset):
    def __init__(self,dataloader, model, device, thres=0.85):
        super(noLabDataset, self).__init__()
        self.model = model      #模型也要传入进来
        self.device = device
        self.thres = thres      #这里置信度阈值 我设置的 0.99
        x, y = self._model_pred(dataloader)        #核心， 获得新的训练数据
        if x == []:                            # 如果没有， 就不启用这个数据集
            self.flag = False
        else:
            self.flag = True
            self.x = np.array(x)
            self.y = torch.LongTensor(y)
        # self.x = np.concatenate((np.array(x), train_dataset.x),axis=0)
        # self.y = torch.cat(((torch.LongTensor(y),train_dataset.y)),dim=0)
        self.transformers = train_transform

    def _model_pred(self, dataloader):
        model = self.model
        device = self.device
        thres = self.thres
        pred_probs = []
        labels = []
        x = []
        y = []
        with torch.no_grad():                                  # 不训练， 要关掉梯度
            for data in dataloader:                            # 取数据
                imgs = data[0].to(device)
                pred = model(imgs)                              #预测
                soft = torch.nn.Softmax(dim=1)             #softmax 可以返回一个概率分布
                pred_p = soft(pred)
                pred_max, preds = pred_p.max(1)          #得到最大值 ，和最大值的位置 。 就是置信度和标签。
                pred_probs.extend(pred_max.cpu().numpy().tolist())
                labels.extend(preds.cpu().numpy().tolist())        #把置信度和标签装起来

        for index, prob in enumerate(pred_probs):
            if prob > thres:                                  #如果置信度超过阈值， 就转化为可信的训练数据
                x.append(dataloader.dataset[index][1])
                y.append(labels[index])
        return x, y

    def __getitem__(self, index):                          # getitem 和len
        x = self.x[index]
        x= self.transformers(x)
        y = self.y[index]
        return x, y

    def __len__(self):
        return len(self.x)

def get_semi_loader(dataloader,model, device, thres):
    semi_set = noLabDataset(dataloader, model, device, thres)
    if semi_set.flag:                                                    #不可用时返回空
        dataloader = DataLoader(semi_set, batch_size=dataloader.batch_size,shuffle=True)
        return dataloader
    else:
        return None


def getDataLoader(path, mode, batchSize):
    assert mode in ['train', 'train_unl', 'val', 'test']
    dataset = foodDataset(path, mode)
    if mode in ['test','train_unl']:
        shuffle = False
    else:
        shuffle = True
    loader = DataLoader(dataset,batchSize,shuffle=shuffle)                      #装入loader
    return loader


def samplePlot(dataset, isloader=True, isbat=False,ori=None):           #画图， 此函数不需要掌握。
    if isloader:
        dataset = dataset.dataset
    rows = 3
    cols = 3
    num = rows*cols
    # if isbat:
    #     dataset = dataset * 225
    datalen = len(dataset)
    randomNum = []
    while len(randomNum) < num:
        temp = random.randint(0,datalen-1)
        if temp not in randomNum:
            randomNum.append(temp)
    fig, axs = plt.subplots(nrows=rows,ncols=cols,squeeze=False)
    index = 0
    for i in range(rows):
        for j in range(cols):
            ax = axs[i, j]
            if isbat:
                ax.imshow(np.array(dataset[randomNum[index]].cpu().permute(1,2,0)))
            else:
                ax.imshow(np.array(dataset[randomNum[index]][0].cpu().permute(1,2,0)))
            index += 1
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.show()
    plt.tight_layout()
    if ori != None:
        fig2, axs2 = plt.subplots(nrows=rows,ncols=cols,squeeze=False)
        index = 0
        for i in range(rows):
            for j in range(cols):
                ax = axs2[i, j]
                if isbat:
                    ax.imshow(np.array(dataset[randomNum[index]][-1]))
                else:
                    ax.imshow(np.array(dataset[randomNum[index]][-1]))
                index += 1
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.show()
        plt.tight_layout()





if __name__ == '__main__':   #运行的模块，  如果你运行的模块是当前模块
    print("你运行的是data.py文件")
    filepath = '../food-11_sample'
    train_loader = getDataLoader(filepath, 'train', 8)
    for i in range(3):
        samplePlot(train_loader,True,isbat=False,ori=True)
    val_loader = getDataLoader(filepath, 'val', 8)
    for i in range(100):
        samplePlot(val_loader,True,isbat=False,ori=True)
    ##########################


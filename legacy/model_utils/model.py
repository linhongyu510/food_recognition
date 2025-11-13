import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import PatchEmbed, Block
import torchvision.models as models



def set_parameter_requires_grad(model, linear_probing):
    if linear_probing:
        for param in model.parameters():
            param.requires_grad = False                             # 一个参数的requires_grad设为false， 则训练时就会不更新

class MyModel(nn.Module):                  #自己的模型
    def __init__(self,numclass = 2):
        super(MyModel, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #112*112
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #56*56
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #28*28
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )  #14*14
        self.pool1 = nn.MaxPool2d(2)#7*7
        self.fc = nn.Linear(25088, 512)
        # self.drop = nn.Dropout(0.5)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, numclass)
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool1(x)
        x = x.view(x.size()[0],-1)       #view 类似于reshape  这里指定了第一维度为batch大小，第二维度为适应的，即剩多少， 就是多少维。
                                        # 这里就是将特征展平。  展为 B*N  ，N为特征维度。
        x = self.fc(x)
        # x = self.drop(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x



# def model_Datapara(model, device,  pre_path=None):
#     model = torch.nn.DataParallel(model).to(device)
# 
#     model_dict = torch.load(pre_path).module.state_dict()
#     model.module.load_state_dict(model_dict)
#     return model

#传入模型名字，和分类数， 返回你想要的模型
def initialize_model(model_name, num_classes, linear_prob=False, use_pretrained=True):
    # 初始化将在此if语句中设置的这些变量。
    # 每个变量都是模型特定的。
    model_ft = None
    input_size = 0
    if model_name =="MyModel":
        if use_pretrained == True:
            model_ft = torch.load('model_save/MyModel')
        else:
            model_ft = MyModel(num_classes)
        input_size = 224

    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)            # 从网络下载模型  pretrain true 使用参数和架构， false 仅使用架构。
        set_parameter_requires_grad(model_ft, linear_prob)            # 是否为线性探测，线性探测： 固定特征提取器不训练。
        num_ftrs = model_ft.fc.in_features  #分类头的输入维度
        model_ft.fc = nn.Linear(num_ftrs, num_classes)            # 删掉原来分类头， 更改最后一层为想要的分类数的分类头。
        input_size = 224
        
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "googlenet":
        """ googlenet
        """
        model_ft = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "alexnet":
        """ Alexnet
 """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
 """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
 """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
 """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
 Be careful, expects (299,299) sized images and has auxiliary output
 """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, linear_prob)
        # 处理辅助网络
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # 处理主要网络
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model_utils name, exiting...")
        exit()

    return model_ft, input_size


def prilearn_para(model_ft,linear_prob):
    # 将模型发送到GPU
    device = torch.device("cuda:0")
    model_ft = model_ft.to(device)

    # 在此运行中收集要优化/更新的参数。
    # 如果我们正在进行微调，我们将更新所有参数。
    # 但如果我们正在进行特征提取方法，我们只会更新刚刚初始化的参数，即`requires_grad`的参数为True。
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if linear_prob:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    #
    # # 观察所有参数都在优化
    # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)




def init_para(model):
    def weights_init(model):
        classname = model.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)
    model.apply(weights_init)
    return model











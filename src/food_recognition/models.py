"""模型构建相关逻辑。"""

from __future__ import annotations

from typing import Tuple

import torch.nn as nn
import torchvision.models as models


def set_parameter_requires_grad(model: nn.Module, linear_probe: bool) -> None:
    """当进行线性探测时冻结除分类头之外的参数。"""
    if not linear_probe:
        return
    for param in model.parameters():
        param.requires_grad = False


class MyConvNet(nn.Module):
    """示例自定义卷积网络。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def initialize_model(
    model_name: str, num_classes: int, linear_probe: bool = False, use_pretrained: bool = True
) -> Tuple[nn.Module, int]:
    """根据名称构建并初始化模型，返回模型和期望输入尺寸。"""
    model_name = model_name.lower()
    input_size = 224

    if model_name == "mymodel":
        model = MyConvNet(num_classes)
        return model, input_size

    if model_name == "resnet18":
        model = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model, linear_probe)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model, input_size

    if model_name == "resnet50":
        model = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model, linear_probe)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model, input_size

    if model_name == "googlenet":
        googlenet = models.googlenet(pretrained=use_pretrained)
        set_parameter_requires_grad(googlenet, linear_probe)
        in_features = googlenet.fc.in_features
        googlenet.fc = nn.Linear(in_features, num_classes)
        return googlenet, input_size

    if model_name == "alexnet":
        alexnet = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(alexnet, linear_probe)
        in_features = alexnet.classifier[6].in_features
        alexnet.classifier[6] = nn.Linear(in_features, num_classes)
        return alexnet, input_size

    if model_name == "vgg11_bn":
        vgg = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(vgg, linear_probe)
        in_features = vgg.classifier[6].in_features
        vgg.classifier[6] = nn.Linear(in_features, num_classes)
        return vgg, input_size

    if model_name == "squeezenet":
        squeezenet = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(squeezenet, linear_probe)
        squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        squeezenet.num_classes = num_classes
        return squeezenet, input_size

    if model_name == "densenet121":
        densenet = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(densenet, linear_probe)
        in_features = densenet.classifier.in_features
        densenet.classifier = nn.Linear(in_features, num_classes)
        return densenet, input_size

    if model_name == "inception_v3":
        input_size = 299
        inception = models.inception_v3(pretrained=use_pretrained, aux_logits=True)
        set_parameter_requires_grad(inception, linear_probe)
        in_features = inception.AuxLogits.fc.in_features
        inception.AuxLogits.fc = nn.Linear(in_features, num_classes)
        in_features = inception.fc.in_features
        inception.fc = nn.Linear(in_features, num_classes)
        return inception, input_size

    raise ValueError(f"不支持的模型名称: {model_name}")


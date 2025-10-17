"""
推理示例

展示如何使用训练好的模型进行推理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from food_recognition.models import ResNet18, EfficientNet_CBAM
from food_recognition.utils import get_device, set_seed


class FoodPredictor:
    """食物预测器"""
    
    def __init__(self, model_path, model_name="resnet18", num_classes=11):
        """
        初始化预测器
        
        Args:
            model_path: 模型路径
            model_name: 模型名称
            num_classes: 类别数
        """
        self.device = get_device()
        self.class_names = [
            'Bread', 'Dairy', 'Dessert', 'Egg', 'Fried', 
            'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Vegetable'
        ]
        
        # 加载模型
        self.model = self._load_model(model_path, model_name, num_classes)
        self.model.eval()
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path, model_name, num_classes):
        """加载模型"""
        if model_name == "resnet18":
            model = ResNet18(num_classes=num_classes, pretrained=False)
        elif model_name == "efficientnet_b0":
            model = EfficientNet_CBAM(num_classes=num_classes, pretrained=False)
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 加载权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        
        return model
    
    def predict(self, image_path):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
        
        Returns:
            预测结果
        """
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            prediction = outputs.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': prediction,
            'class_name': self.class_names[prediction],
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
    
    def predict_batch(self, image_paths):
        """
        批量预测
        
        Args:
            image_paths: 图片路径列表
        
        Returns:
            预测结果列表
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                results.append(result)
            except Exception as e:
                print(f"预测 {image_path} 时出错: {e}")
                results.append(None)
        
        return results


def main():
    """主函数"""
    # 设置随机种子
    set_seed(42)
    
    # 模型配置
    model_path = "checkpoints/best_model.pth"
    model_name = "resnet18"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型或检查模型路径")
        return
    
    # 创建预测器
    predictor = FoodPredictor(model_path, model_name)
    
    # 示例图片路径（请替换为实际路径）
    image_paths = [
        "data/test/bread_001.jpg",
        "data/test/meat_001.jpg",
        "data/test/vegetable_001.jpg"
    ]
    
    # 检查图片是否存在
    valid_paths = [path for path in image_paths if os.path.exists(path)]
    
    if not valid_paths:
        print("没有找到有效的测试图片")
        print("请将测试图片放在 data/test/ 目录下")
        return
    
    # 进行预测
    print("开始预测...")
    results = predictor.predict_batch(valid_paths)
    
    # 显示结果
    for i, (path, result) in enumerate(zip(valid_paths, results)):
        if result is not None:
            print(f"\n图片 {i+1}: {path}")
            print(f"预测类别: {result['class_name']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"所有类别概率:")
            for j, (class_name, prob) in enumerate(zip(predictor.class_names, result['probabilities'])):
                print(f"  {class_name}: {prob:.4f}")
        else:
            print(f"\n图片 {i+1}: {path} - 预测失败")


if __name__ == "__main__":
    main()




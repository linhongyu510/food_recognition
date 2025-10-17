"""
数据集下载脚本

下载常用的食物识别数据集
"""

import os
import requests
import zipfile
import tarfile
from pathlib import Path
import shutil


def download_file(url, filename):
    """下载文件"""
    print(f"正在下载 {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"下载完成: {filename}")


def extract_zip(zip_path, extract_to):
    """解压zip文件"""
    print(f"正在解压 {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"解压完成: {extract_to}")


def extract_tar(tar_path, extract_to):
    """解压tar文件"""
    print(f"正在解压 {tar_path}...")
    with tarfile.open(tar_path, 'r:*') as tar_ref:
        tar_ref.extractall(extract_to)
    print(f"解压完成: {extract_to}")


def download_food101():
    """下载Food-101数据集"""
    data_dir = Path("data/food101")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Food-101数据集下载链接（这里使用一个示例链接，实际使用时需要替换）
    # 由于Food-101数据集较大，这里提供一个模拟的下载过程
    print("Food-101数据集下载说明:")
    print("1. 请访问 https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/")
    print("2. 下载 food-101.tar.gz 文件")
    print("3. 将文件放在 data/food101/ 目录下")
    print("4. 运行以下命令解压:")
    print("   tar -xzf data/food101/food-101.tar.gz -C data/food101/")
    
    # 创建示例目录结构
    (data_dir / "images").mkdir(exist_ok=True)
    (data_dir / "meta").mkdir(exist_ok=True)
    
    return data_dir


def download_food11():
    """下载Food-11数据集（模拟）"""
    data_dir = Path("data/food11")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建Food-11数据集结构
    categories = [
        'Bread', 'Dairy', 'Dessert', 'Egg', 'Fried',
        'Meat', 'Noodles', 'Rice', 'Seafood', 'Soup', 'Vegetable'
    ]
    
    for split in ['training/labeled', 'training/unlabeled', 'validation', 'testing']:
        split_dir = data_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        if split in ['training/labeled', 'validation']:
            for i, category in enumerate(categories):
                category_dir = split_dir / f"{i:02d}"
                category_dir.mkdir(exist_ok=True)
                print(f"创建目录: {category_dir}")
    
    print("Food-11数据集结构已创建")
    return data_dir


def download_sample_images():
    """下载示例图片"""
    print("正在下载示例图片...")
    
    # 这里可以添加一些示例图片的下载逻辑
    # 由于实际图片下载需要具体的URL，这里提供一个框架
    
    sample_dir = Path("data/samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print("示例图片目录已创建: data/samples")
    return sample_dir


def setup_data_structure():
    """设置数据结构"""
    print("设置数据结构...")
    
    # 创建主要目录
    directories = [
        "data",
        "data/food101",
        "data/food11", 
        "data/samples",
        "checkpoints",
        "logs",
        "results",
        "results/plots",
        "results/models"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"创建目录: {directory}")
    
    print("数据结构设置完成")


def main():
    """主函数"""
    print("=== 食物识别数据集下载和设置 ===")
    
    # 设置数据结构
    setup_data_structure()
    
    # 下载Food-101数据集
    print("\n1. 设置Food-101数据集...")
    food101_dir = download_food101()
    
    # 下载Food-11数据集
    print("\n2. 设置Food-11数据集...")
    food11_dir = download_food11()
    
    # 下载示例图片
    print("\n3. 设置示例图片...")
    sample_dir = download_sample_images()
    
    print("\n=== 数据集设置完成 ===")
    print("请按照以下步骤完成数据集准备:")
    print("1. 下载Food-101数据集并解压到 data/food101/")
    print("2. 准备训练图片并放入相应的类别文件夹")
    print("3. 运行训练脚本开始训练")


if __name__ == "__main__":
    main()




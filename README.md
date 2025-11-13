# Food Recognition

整理后的项目提供了一个可配置、可扩展的食物识别训练流程，支持半监督学习与多种 torchvision 模型选择，同时保留历史实验脚本以供参考。

## 目录结构

- `src/food_recognition/`：核心库代码，包含配置、数据加载、模型与训练逻辑。
- `configs/`：训练配置文件，默认提供 `default.yaml`。
- `scripts/`：命令行工具，如 `train.py`（训练入口）、`download_dataset.py`（数据集下载示例）。
- `experiments/`：示例与扩展示例，`legacy/` 子目录存档旧的单文件实验脚本。
- `legacy/`：旧版 `model_utils` 模块，若需要复现历史版本，可按需引用。
- `reports/`：训练过程生成的图表或日志示例。

## 快速开始

1. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```
2. **准备数据集**  
   可参考 `scripts/download_dataset.py` 或手动下载 `food-11` / `food-11_sample` 数据集，并保持与配置文件中一致的目录结构。
3. **运行训练**
   ```bash
   python scripts/train.py --config configs/default.yaml
   ```
   常用覆盖参数示例：
   ```bash
   python scripts/train.py --epochs 20 --model-name resnet50 --use-pretrained
   ```

## 配置说明

`configs/default.yaml` 中的关键字段：

| 字段 | 作用 |
| ---- | ---- |
| `model_name` | 选择使用的模型（如 `resnet18`、`resnet50`、`vgg11_bn` 等） |
| `train_dir` / `val_dir` / `unlabeled_dir` | 分别指向有标签训练集、验证集、无标签数据目录 |
| `batch_size` / `epochs` / `learning_rate` | 训练超参数 |
| `semi_supervised.enabled` | 是否启用半监督训练流程 |
| `semi_supervised.confidence_threshold` | 生成伪标签时的置信度阈值 |

配置文件路径可以通过 `--config` 指定；命令行参数可在运行时覆盖 YAML 中的设置。

## 半监督流程

当验证集准确率达到 `semi_supervised.activation_threshold` 且满足刷新间隔时，会对无标签数据生成伪标签，并在同一 epoch 内追加一次训练，以提升模型表现。

## 历史代码

- 旧的 `model_utils` 与分散脚本现统一归档在 `legacy/` 与 `experiments/legacy/`，默认训练流程无需依赖。
- 如需参考或复现早期实验，可在调用脚本中将 `legacy` 目录加入 `PYTHONPATH`。

## 其他

- 目标检测实验示例迁移至 `experiments/object_detection.py`，涉及 `albumentations`、`torchvision` 检测模型等扩展依赖。
- 运行 `python main.py` 与调用 `scripts/train.py` 等价。

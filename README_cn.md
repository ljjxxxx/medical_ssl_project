简体中文 | [English](README.md)
# 🫁 胸部X光疾病分类项目 🔬

这个项目使用深度学习技术自动分析胸部X光片，可以检测14种常见的胸部疾病。该项目包含完整的训练、评估和应用部署流程，采用了自监督学习(SSL)和监督学习相结合的方法来提高模型性能。让AI成为医生的好帮手！✨

## ✨ 主要特点

- 🧠 **自监督预训练**: 使用SimCLR对比学习方法对模型进行预训练，让机器自己学会"看"X光片
- 🏷️ **多标签分类**: 同时检测多种胸部疾病，一次扫描全面诊断
- 🖥️ **交互式界面**: 基于Gradio的用户友好Web应用程序，点几下鼠标就能用上AI诊断
- 📊 **详细评估**: 提供精确度、召回率、F1分数和AUC等多种性能指标，让结果一目了然
- 📈 **可视化支持**: 训练过程和评估结果的图表可视化，数据不再枯燥
- 🍎 **针对Mac进行了优化**: Mac的MPS显卡终于不再吃灰啦

## 📂 项目结构

```
medical_ssl_project/
│
├── datasets/                  # 数据集目录
│
├── src/                       # 源代码目录
│   ├── models/                # 模型相关代码
│   │   ├── __init__.py
│   │   ├── backbones.py       # 主干网络
│   │   ├── ssl_model.py       # 自监督学习模型
│   │   └── classifier.py      # 疾病分类器
│   │
│   ├── data/                  # 数据处理相关代码
│   │   ├── __init__.py
│   │   ├── dataset.py         # 数据集类
│   │   ├── transforms.py      # 数据转换和增强
│   │   └── download.py        # 数据下载函数
│   │
│   ├── utils/                 # 通用工具函数
│   │   ├── __init__.py
│   │   ├── device.py          # 设备选择工具
│   │   ├── visualization.py   # 图表生成和可视化
│   │   └── metrics.py         # 评估指标
│   │
│   ├── training/              # 训练相关代码
│   │   ├── __init__.py
│   │   ├── ssl_trainer.py     # SSL训练器
│   │   └── classifier_trainer.py  # 分类器训练器
│   │
│   └── app/                   # Web应用相关代码
│       ├── __init__.py
│       ├── interface.py       # Gradio界面
│       └── disease_info.py    # 疾病信息和翻译
│
├── scripts/                   # 脚本目录
│   ├── train_ssl.py           # SSL训练脚本
│   ├── train_classifier.py    # 分类器训练脚本
│   ├── evaluate.py            # 评估脚本
│   └── regenerate_plots.py    # 重新生成图表脚本
│
├── examples/                  # 示例X光图像目录
│
├── checkpoints/               # 模型检查点目录
│
├── app.py                     # 应用入口
├── requirements.txt           # 项目依赖
├── setup.py                   # 安装脚本
└── README.md                  # 项目说明文档
```

## 📊 数据集

该项目使用NIH ChestX-ray14数据集，包含超过100,000张有标注的胸部X光片，涵盖14种常见病理。这么大的数据集，够机器学个够！🤓风扇呼呼吹！💨

下载地址：https://www.kaggle.com/datasets/nih-chest-xrays/data

## 🚀 使用方法

### 🏋️‍♀️ 训练模型

#### 1. 自监督学习预训练 (让模型自己学习特征):

```bash
python scripts/train_ssl.py --epochs 50 --batch-size 32
```

可用参数（调参大师，请开始你的表演）:

- `--device`: 训练设备 (auto, cuda, mps, 或 cpu)
- `--batch-size`: 批量大小 (默认: 32)
- `--epochs`: 训练轮数 (默认: 50)
- `--checkpoint-dir`: 检查点保存目录 (默认: ./checkpoints)
- `--num-workers`: 数据加载的工作进程数 (默认: 4)
- `--resume`: 从上次中断处继续训练（被打断了没关系，接着来！）
- `--lr`: 学习率 (默认: 0.0003)

#### 2. 分类器训练 (教会模型识别疾病):

```bash
python scripts/train_classifier.py --epochs 30 --ssl-model ./checkpoints/best_ssl_model.pt
```

可用参数:

- `--device`: 训练设备 (auto, cuda, mps, 或 cpu)
- `--batch-size`: 批量大小 (默认: 32)
- `--epochs`: 训练轮数 (默认: 30)
- `--checkpoint-dir`: 检查点保存目录 (默认: ./checkpoints)
- `--ssl-model`: 预训练SSL模型路径 (如果不提供，将使用预训练的ResNet)
- `--num-workers`: 数据加载的工作进程数 (默认: 4)
- `--resume`: 从上次中断处继续训练
- `--lr`: 学习率 (默认: 0.0001)
- `--freeze-backbone`: 是否冻结骨干网络（冻住参数，省心省力）

### 🔍 评估模型

评估模型性能:

```bash
python scripts/evaluate.py --checkpoint-path ./checkpoints/best_classifier_model.pt
```

可用参数:

- `--checkpoint-path`: 模型检查点文件路径 (默认: ./checkpoints/best_classifier_model.pt)
- `--data-dir`: 数据集目录路径 (默认: ./datasets/chestxray14)
- `--batch-size`: 评估批量大小 (默认: 32)
- `--output-dir`: 保存评估结果的目录 (默认: ./evaluation_results)
- `--device`: 评估设备 (cuda, mps, 或 cpu)
- `--num-workers`: 数据加载的工作进程数 (默认: 4)
- `--strict-loading`: 使用严格的state_dict加载 (设为False以跳过不匹配的键)

### 📈 重新生成训练图表

重新生成训练过程图表:

```bash
python scripts/regenerate_plots.py
```

可用参数:

- `--checkpoint-dir`: 包含训练状态JSON文件的目录 (默认: ./checkpoints)
- `--output-dir`: 保存重新生成的图表的目录 (默认: 与checkpoint-dir相同)
- `--ssl-only`: 仅重新生成SSL训练曲线
- `--classifier-only`: 仅重新生成分类器训练曲线

### 🌐 启动Web应用

启动交互式Web界面:

```bash
python app.py
```

可用参数:

- `--model`: 训练好的模型检查点路径 (默认: ./checkpoints/best_classifier_model.pt)
- `--device`: 运行设备 (auto, mps, cuda, 或 cpu)
- `--port`: Gradio服务器端口 (默认: 7860)

应用将在`http://localhost:7860`上可访问。打开浏览器就能用啦！🎉

## 📊 性能指标

模型在测试集上的性能:

- **整体准确率**: 0.9494 🎯
- **整体精确率**: 0.5492 🔍
- **整体召回率**: 0.1111 🕵️
- **整体F1分数**: 0.1848 ⚖️
- **平均AUC**: 0.8217 📈

各疾病AUC和F1分数可以在`evaluation_results/test_metrics.txt`中查看，或通过以下命令重新生成:

```bash
python scripts/evaluate.py
```

## 🐍 Python 版本

本项目使用的python版本为3.8 (不是最新但很稳定！)

## 🙏 致谢

- 👩‍⚕️ 感谢NIH提供ChestX-ray14数据集
- 🧠 本项目的对比学习部分参考了SimCLR论文
- 🤖 本项目使用了AI提升开发效率（AI帮AI，效率加倍！）
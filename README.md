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

## 💻 页面效果
![img.png](assets%2Fimg.png)
![img_1.png](assets%2Fimg_1.png)
![img_2.png](assets%2Fimg_2.png)
![img_3.png](assets%2Fimg_3.png)
![img_4.png](assets%2Fimg_4.png)
![img_5.png](assets%2Fimg_5.png)
![img_6.png](assets%2Fimg_6.png)


## 🙏 致谢

- 👩‍⚕️ 感谢NIH提供ChestX-ray14数据集
- 🧠 本项目的对比学习部分参考了SimCLR论文
- 🤖 本项目使用了AI提升开发效率（AI帮AI，效率加倍！）


------------

# 🫁 Chest X-Ray Disease Classification Project 🔬

This project uses deep learning technology to automatically analyze chest X-rays and detect 14 common chest diseases. The project includes complete training, evaluation, and deployment processes, using a combination of self-supervised learning (SSL) and supervised learning methods to improve model performance. Let AI become a doctor's helpful assistant! ✨

## ✨ Key Features

- 🧠 **Self-Supervised Pre-training**: Uses SimCLR contrastive learning method to pre-train the model, allowing the machine to learn to "see" X-rays on its own
- 🏷️ **Multi-Label Classification**: Simultaneously detects multiple chest diseases for comprehensive diagnosis in one scan
- 🖥️ **Interactive Interface**: User-friendly web application based on Gradio, AI diagnosis with just a few clicks
- 📊 **Detailed Evaluation**: Provides multiple performance metrics including precision, recall, F1 score, and AUC for clear results
- 📈 **Visualization Support**: Chart visualization of training process and evaluation results, making data more engaging
- 🍎 **Optimized for Mac**: Mac's MPS GPU finally gets to shine

## 📂 Project Structure

```
medical_ssl_project/
│
├── datasets/                  # Dataset directory
│
├── src/                       # Source code directory
│   ├── models/                # Model-related code
│   │   ├── __init__.py
│   │   ├── backbones.py       # Backbone networks
│   │   ├── ssl_model.py       # Self-supervised learning model
│   │   └── classifier.py      # Disease classifier
│   │
│   ├── data/                  # Data processing related code
│   │   ├── __init__.py
│   │   ├── dataset.py         # Dataset classes
│   │   ├── transforms.py      # Data transformations and augmentation
│   │   └── download.py        # Data download functions
│   │
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── device.py          # Device selection utilities
│   │   ├── visualization.py   # Chart generation and visualization
│   │   └── metrics.py         # Evaluation metrics
│   │
│   ├── training/              # Training related code
│   │   ├── __init__.py
│   │   ├── ssl_trainer.py     # SSL trainer
│   │   └── classifier_trainer.py  # Classifier trainer
│   │
│   └── app/                   # Web application related code
│       ├── __init__.py
│       ├── interface.py       # Gradio interface
│       └── disease_info.py    # Disease information and translations
│
├── scripts/                   # Scripts directory
│   ├── train_ssl.py           # SSL training script
│   ├── train_classifier.py    # Classifier training script
│   ├── evaluate.py            # Evaluation script
│   └── regenerate_plots.py    # Plot regeneration script
│
├── examples/                  # Example X-ray images directory
│
├── checkpoints/               # Model checkpoints directory
│
├── app.py                     # Application entry point
├── requirements.txt           # Project dependencies
├── setup.py                   # Installation script
└── README.md                  # Project documentation
```

## 📊 Dataset

This project uses the NIH ChestX-ray14 dataset, which contains over 100,000 annotated chest X-rays covering 14 common pathologies. Such a large dataset is perfect for machine learning! 🤓 Fans whirring! 💨

Download link: https://www.kaggle.com/datasets/nih-chest-xrays/data

## 🚀 Usage

### 🏋️‍♀️ Training Models

#### 1. Self-Supervised Learning Pre-training (Let the model learn features on its own):

```bash
python scripts/train_ssl.py --epochs 50 --batch-size 32
```

Available parameters (Parameter tuning masters, show us your skills):

- `--device`: Training device (auto, cuda, mps, or cpu)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--checkpoint-dir`: Checkpoint save directory (default: ./checkpoints)
- `--num-workers`: Number of data loading workers (default: 4)
- `--resume`: Resume training from last checkpoint (Interrupted? No worries, continue!)
- `--lr`: Learning rate (default: 0.0003)

#### 2. Classifier Training (Teaching the model to recognize diseases):

```bash
python scripts/train_classifier.py --epochs 30 --ssl-model ./checkpoints/best_ssl_model.pt
```

Available parameters:

- `--device`: Training device (auto, cuda, mps, or cpu)
- `--batch-size`: Batch size (default: 32)
- `--epochs`: Number of training epochs (default: 30)
- `--checkpoint-dir`: Checkpoint save directory (default: ./checkpoints)
- `--ssl-model`: Pre-trained SSL model path (if not provided, will use pre-trained ResNet)
- `--num-workers`: Number of data loading workers (default: 4)
- `--resume`: Resume training from last checkpoint
- `--lr`: Learning rate (default: 0.0001)
- `--freeze-backbone`: Whether to freeze backbone network (Freeze parameters, save effort)

### 🔍 Model Evaluation

Evaluate model performance:

```bash
python scripts/evaluate.py --checkpoint-path ./checkpoints/best_classifier_model.pt
```

Available parameters:

- `--checkpoint-path`: Model checkpoint file path (default: ./checkpoints/best_classifier_model.pt)
- `--data-dir`: Dataset directory path (default: ./datasets/chestxray14)
- `--batch-size`: Evaluation batch size (default: 32)
- `--output-dir`: Directory to save evaluation results (default: ./evaluation_results)
- `--device`: Evaluation device (cuda, mps, or cpu)
- `--num-workers`: Number of data loading workers (default: 4)
- `--strict-loading`: Use strict state_dict loading (set to False to skip mismatched keys)

### 📈 Regenerate Training Plots

Regenerate training process plots:

```bash
python scripts/regenerate_plots.py
```

Available parameters:

- `--checkpoint-dir`: Directory containing training state JSON files (default: ./checkpoints)
- `--output-dir`: Directory to save regenerated plots (default: same as checkpoint-dir)
- `--ssl-only`: Only regenerate SSL training curves
- `--classifier-only`: Only regenerate classifier training curves

### 🌐 Launch Web Application

Launch the interactive web interface:

```bash
python app.py
```

Available parameters:

- `--model`: Trained model checkpoint path (default: ./checkpoints/best_classifier_model.pt)
- `--device`: Running device (auto, mps, cuda, or cpu)
- `--port`: Gradio server port (default: 7860)

The application will be accessible at `http://localhost:7860`. Just open your browser and you're ready to go! 🎉

## 📊 Performance Metrics

Model performance on the test set:

- **Overall Accuracy**: 0.9494 🎯
- **Overall Precision**: 0.5492 🔍
- **Overall Recall**: 0.1111 🕵️
- **Overall F1 Score**: 0.1848 ⚖️
- **Average AUC**: 0.8217 📈

AUC and F1 scores for each disease can be found in `evaluation_results/test_metrics.txt`, or regenerated using:

```bash
python scripts/evaluate.py
```

## 🐍 Python Version

This project uses Python version 3.8 (Not the latest but very stable!)

## 🙏 Acknowledgments

- 👩‍⚕️ Thanks to NIH for providing the ChestX-ray14 dataset
- 🧠 The contrastive learning part of this project is based on the SimCLR paper
- 🤖 This project used AI to enhance development efficiency (AI helping AI, double the efficiency!)
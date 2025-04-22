English | [简体中文](README_cn.md)
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
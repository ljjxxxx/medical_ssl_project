import torch
import logging
import time
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from src.utils.visualization import plot_training_curves
from src.utils.metrics import calculate_metrics

# 设置后端以在没有GUI的服务器上生成图表
matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_classifier(model, train_loader, val_loader, optimizer, criterion, device, config, disease_labels,
                     resume=False):
    """
    训练分类器模型

    Args:
        model: 分类器模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 训练设备
        config: 配置参数
        disease_labels: 疾病标签列表
        resume: 是否从上次中断处继续训练

    Returns:
        model: 训练后的模型
    """
    logger.info("Starting classifier training")

    epochs = config['finetune_epochs']
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoint_dir / 'best_classifier_model.pt'
    last_model_path = checkpoint_dir / 'last_classifier_model.pt'
    train_state_path = checkpoint_dir / 'classifier_train_state.json'

    train_losses = []
    val_losses = []
    val_metrics = []
    best_val_metric = 0
    start_epoch = 0

    if resume and last_model_path.exists() and train_state_path.exists():
        logger.info(f"Resuming training from {last_model_path}")

        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with open(train_state_path, 'r') as f:
            train_state = json.load(f)

        start_epoch = train_state['epoch']
        train_losses = train_state['train_losses']
        val_losses = train_state['val_losses']
        val_metrics = train_state['val_metrics']
        best_val_metric = train_state['best_val_metric']

        logger.info(f"Resuming training from epoch {start_epoch}/{epochs}")
    else:
        logger.info("Starting training from scratch")

    start_time = time.time()

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for batch in pbar:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(images)
                    logits = outputs['logits']

                    loss = criterion(logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            all_probs = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(images)
                    logits = outputs['logits']

                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()

                    all_probs.append(probs.cpu().numpy())
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            all_probs = np.vstack(all_probs)

            metrics, per_disease_metrics = calculate_metrics(all_labels, all_preds, all_probs, disease_labels)

            metrics['per_disease'] = per_disease_metrics
            val_metrics.append(metrics)

            current_metric = metrics['f1']

            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}, "
                        f"Val F1: {metrics['f1']:.4f}, "
                        f"Val AUC: {metrics['auc']:.4f}")

            if current_metric > best_val_metric:
                best_val_metric = current_metric
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1_score': metrics['f1'],
                    'auc': metrics['auc']
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f"Saved best model, validation F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': metrics['f1'],
                'auc': metrics['auc']
            }
            torch.save(checkpoint, last_model_path)

            train_state = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_metrics': val_metrics,
                'best_val_metric': best_val_metric,
                'time_elapsed': time.time() - start_time
            }

            with open(train_state_path, 'w') as f:
                json.dump(train_state, f)

            if (epoch + 1) % 5 == 0:
                torch.save(checkpoint, checkpoint_dir / f'classifier_model_epoch_{epoch + 1}.pt')

    except KeyboardInterrupt:
        logger.info("Training interrupted by user! Saving current state for later resumption...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1_score': metrics['f1'] if 'metrics' in locals() else 0,
            'auc': metrics['auc'] if 'metrics' in locals() else 0
        }
        torch.save(checkpoint, last_model_path)

        train_state = {
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_metrics': val_metrics,
            'best_val_metric': best_val_metric,
            'time_elapsed': time.time() - start_time
        }

        with open(train_state_path, 'w') as f:
            json.dump(train_state, f)

        logger.info(f"Saved interrupt checkpoint, can resume training with --resume flag")
        return model

    # 绘制训练曲线
    plot_training_curves(train_state, checkpoint_dir, plot_type='classifier')

    with open(checkpoint_dir / 'disease_metrics.txt', 'w') as f:
        f.write("Disease Metrics Summary (Best Model):\n\n")
        f.write(f"Overall F1 Score: {best_val_metric:.4f}\n")
        f.write(f"Overall AUC: {val_metrics[-1]['auc']:.4f}\n\n")
        f.write("Metrics for each disease:\n")
        for disease, metric in val_metrics[-1]['per_disease'].items():
            f.write(f"{disease}:\n")
            f.write(f"  Precision: {metric['precision']:.4f}\n")
            f.write(f"  Recall: {metric['recall']:.4f}\n")
            f.write(f"  F1 Score: {metric['f1']:.4f}\n\n")

    training_time = time.time() - start_time
    logger.info(f"Classifier training completed in {training_time / 60:.2f} minutes")

    return model
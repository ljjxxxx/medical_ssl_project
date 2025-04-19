import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import logging
from pathlib import Path
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    average_precision_score

from models import SimCLREncoder, ChestXrayClassifier
from data_utils import download_chestxray14_dataset, create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_available_device():
    mps_available = False
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            try:
                test_tensor = torch.zeros(1, device="mps")
                mps_available = True
                logger.info("MPS加速可用且正常工作")
                return torch.device("mps")
            except RuntimeError as e:
                logger.warning(f"MPS报告可用但失败: {e}")
                logger.warning("回退到CPU")
        else:
            if torch.backends.mps.is_available():
                logger.warning("MPS可用但PyTorch未使用MPS支持构建")
            else:
                logger.info("MPS加速在此系统上不可用")
    except AttributeError:
        logger.warning("PyTorch版本不支持MPS")

    if torch.cuda.is_available():
        logger.info("CUDA加速可用")
        return torch.device("cuda")

    logger.info("使用CPU进行计算（无GPU加速可用）")
    return torch.device("cpu")


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ssl_model(model, train_loader, val_loader, optimizer, device, config, resume=False):
    logger.info("开始SSL模型训练")

    epochs = config['ssl_epochs']
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = checkpoint_dir / 'best_ssl_model.pt'
    last_model_path = checkpoint_dir / 'last_ssl_model.pt'
    train_state_path = checkpoint_dir / 'ssl_train_state.json'

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    start_epoch = 0

    if resume and last_model_path.exists() and train_state_path.exists():
        logger.info(f"正在从 {last_model_path} 恢复训练")

        checkpoint = torch.load(last_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        with open(train_state_path, 'r') as f:
            train_state = json.load(f)

        start_epoch = train_state['epoch']
        train_losses = train_state['train_losses']
        val_losses = train_state['val_losses']
        best_val_loss = train_state['best_val_loss']

        logger.info(f"恢复训练从epoch {start_epoch}/{epochs}")
    else:
        logger.info("从头开始训练")

    model.train()
    start_time = time.time()

    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0

            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
                for batch in pbar:
                    views = batch['views']
                    view1, view2 = views[0].to(device), views[1].to(device)

                    _, proj1 = model(view1)
                    _, proj2 = model(view2)

                    loss = model.contrastive_loss(proj1, proj2)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    views = batch['views']
                    view1, view2 = views[0].to(device), views[1].to(device)

                    _, proj1 = model(view1)
                    _, proj2 = model(view2)

                    loss = model.contrastive_loss(proj1, proj2)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }
                torch.save(checkpoint, best_model_path)
                logger.info(f"保存了最佳模型，验证损失: {best_val_loss:.4f}")

            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }
            torch.save(checkpoint, last_model_path)

            train_state = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss,
                'time_elapsed': time.time() - start_time
            }

            with open(train_state_path, 'w') as f:
                json.dump(train_state, f)

            if (epoch + 1) % 10 == 0:
                torch.save(checkpoint, checkpoint_dir / f'ssl_model_epoch_{epoch + 1}.pt')

    except KeyboardInterrupt:
        logger.info("训练被用户中断！保存当前状态以便稍后恢复...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss if 'avg_val_loss' in locals() else float('inf'),
        }
        torch.save(checkpoint, last_model_path)

        train_state = {
            'epoch': epoch,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'time_elapsed': time.time() - start_time
        }

        with open(train_state_path, 'w') as f:
            json.dump(train_state, f)

        logger.info(f"已保存中断检查点，可以使用 --resume 参数恢复训练")
        return model

    plt.switch_backend('agg')

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('SSL训练损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(checkpoint_dir / 'ssl_training_curve.png')
    plt.close()

    torch.save(model.state_dict(), checkpoint_dir / 'final_ssl_model.pt')

    training_time = time.time() - start_time
    logger.info(f"SSL训练完成，用时 {training_time / 60:.2f} 分钟")

    return model


def train_classifier(model, train_loader, val_loader, optimizer, criterion, device, config, disease_labels,
                     resume=False):
    logger.info("开始分类器训练")

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
        logger.info(f"正在从 {last_model_path} 恢复训练")

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

        logger.info(f"恢复训练从epoch {start_epoch}/{epochs}")
    else:
        logger.info("从头开始训练")

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

                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)

            metrics = {}

            metrics['accuracy'] = float(accuracy_score(all_labels.flatten(), all_preds.flatten()))
            metrics['precision'] = float(precision_score(all_labels.flatten(), all_preds.flatten(), zero_division=0))
            metrics['recall'] = float(recall_score(all_labels.flatten(), all_preds.flatten(), zero_division=0))
            metrics['f1'] = float(f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0))

            aucs = []
            for i in range(all_labels.shape[1]):
                if np.any(all_labels[:, i] == 1):
                    try:
                        auc = float(roc_auc_score(all_labels[:, i], all_preds[:, i]))
                        aucs.append(auc)
                    except ValueError:
                        pass

            metrics['auc'] = float(np.mean(aucs) if aucs else 0)

            per_disease_metrics = {}
            for i, disease in enumerate(disease_labels):
                per_disease_metrics[disease] = {
                    'precision': float(precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)),
                    'recall': float(recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)),
                    'f1': float(f1_score(all_labels[:, i], all_preds[:, i], zero_division=0))
                }

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
                logger.info(f"保存了最佳模型，验证F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

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
        logger.info("训练被用户中断！保存当前状态以便稍后恢复...")
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

        logger.info(f"已保存中断检查点，可以使用 --resume 参数恢复训练")
        return model

    plt.switch_backend('agg')

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot([m['f1'] for m in val_metrics], label='F1分数')
    plt.plot([m['accuracy'] for m in val_metrics], label='准确率')
    plt.xlabel('Epoch')
    plt.ylabel('分数')
    plt.title('F1和准确率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot([m['precision'] for m in val_metrics], label='精确率')
    plt.plot([m['recall'] for m in val_metrics], label='召回率')
    plt.xlabel('Epoch')
    plt.ylabel('分数')
    plt.title('精确率和召回率')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot([m['auc'] for m in val_metrics], label='AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('ROC曲线下面积')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(checkpoint_dir / 'classifier_training_curve.png')
    plt.close()

    with open(checkpoint_dir / 'disease_metrics.txt', 'w') as f:
        f.write("疾病指标汇总（最佳模型）:\n\n")
        f.write(f"整体F1分数: {best_val_metric:.4f}\n")
        f.write(f"整体AUC: {val_metrics[-1]['auc']:.4f}\n\n")
        f.write("每个疾病的指标:\n")
        for disease, metric in val_metrics[-1]['per_disease'].items():
            f.write(f"{disease}:\n")
            f.write(f"  精确率: {metric['precision']:.4f}\n")
            f.write(f"  召回率: {metric['recall']:.4f}\n")
            f.write(f"  F1分数: {metric['f1']:.4f}\n\n")

    training_time = time.time() - start_time
    logger.info(f"分类器训练完成，用时 {training_time / 60:.2f} 分钟")

    return model


def evaluate_model(model, test_loader, device, disease_labels):
    logger.info("在测试集上评估模型")

    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            logits = outputs['logits']

            # 计算概率和预测
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    metrics = {}
    metrics['accuracy'] = accuracy_score(all_labels.flatten(), all_preds.flatten())
    metrics['precision'] = precision_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    metrics['recall'] = recall_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)
    metrics['f1'] = f1_score(all_labels.flatten(), all_preds.flatten(), zero_division=0)

    disease_metrics = {}
    avg_auc = 0
    num_diseases = len(disease_labels)

    for i, disease in enumerate(disease_labels):
        disease_metrics[disease] = {
            'precision': precision_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            'recall': recall_score(all_labels[:, i], all_preds[:, i], zero_division=0),
            'f1': f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        }

        try:
            if np.any(all_labels[:, i] == 1):
                auc = roc_auc_score(all_labels[:, i], all_probs[:, i])
                disease_metrics[disease]['auc'] = auc
                avg_auc += auc
            else:
                disease_metrics[disease]['auc'] = float('nan')
        except ValueError:
            disease_metrics[disease]['auc'] = float('nan')

    metrics['avg_auc'] = avg_auc / num_diseases

    logger.info(f"测试准确率: {metrics['accuracy']:.4f}")
    logger.info(f"测试精确率: {metrics['precision']:.4f}")
    logger.info(f"测试召回率: {metrics['recall']:.4f}")
    logger.info(f"测试F1分数: {metrics['f1']:.4f}")
    logger.info(f"测试平均AUC: {metrics['avg_auc']:.4f}")

    logger.info("每个疾病的指标:")
    for disease, metric in disease_metrics.items():
        logger.info(f"{disease}:")
        logger.info(f"  精确率: {metric['precision']:.4f}")
        logger.info(f"  召回率: {metric['recall']:.4f}")
        logger.info(f"  F1分数: {metric['f1']:.4f}")
        logger.info(f"  AUC: {metric['auc']:.4f}")

    return {
        'overall': metrics,
        'per_disease': disease_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='胸部X光疾病分类项目')
    parser.add_argument('--mode', type=str, choices=['ssl', 'finetune', 'all', 'evaluate'],
                        default='all', help='运行模式')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (auto, mps, cuda, 或 cpu)')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--ssl-epochs', type=int, default=50, help='SSL训练轮数')
    parser.add_argument('--finetune-epochs', type=int, default=30, help='微调轮数')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='保存检查点的目录')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载的工作进程数')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断的地方恢复训练')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='保存检查点的频率（epochs）')
    args = parser.parse_args()

    if args.device is None or args.device == 'auto':
        device = get_available_device()
    else:
        try:
            device = torch.device(args.device)
            if args.device == 'mps':
                try:
                    test_tensor = torch.zeros(1, device=device)
                except RuntimeError:
                    logger.warning("指定了 MPS 但不可用，将使用 CPU。")
                    device = torch.device("cpu")
        except:
            logger.warning(f"无效的设备 '{args.device}'，将自动检测设备。")
            device = get_available_device()

    logger.info(f"使用设备: {device}")

    set_seed(42)

    if str(device) == 'mps':
        logger.info("正在为Apple Silicon (M1/M2)优化参数")
        num_workers = min(os.cpu_count() - 1, args.num_workers)
        logger.info(f"使用 {num_workers} 个数据加载器工作进程")
    else:
        num_workers = args.num_workers

    config = {
        'ssl_epochs': args.ssl_epochs,
        'finetune_epochs': args.finetune_epochs,
        'batch_size': args.batch_size,
        'checkpoint_dir': args.checkpoint_dir,
        'backbone': 'resnet18',
        'projection_dim': 128,
        'temperature': 0.1
    }

    dataset_path = download_chestxray14_dataset()

    dataloaders_with_labels = create_dataloaders(
        data_dir=dataset_path,
        batch_size=config['batch_size'],
        num_workers=num_workers
    )

    dataloaders = {
        'ssl': dataloaders_with_labels['ssl'],
        'supervised': dataloaders_with_labels['supervised']
    }
    disease_labels = dataloaders_with_labels['disease_labels']

    config['num_classes'] = len(disease_labels)

    logger.info(f"疾病标签: {disease_labels}")
    logger.info(f"类别数量: {config['num_classes']}")

    if args.mode in ['ssl', 'all']:
        ssl_model = SimCLREncoder(
            backbone_name=config['backbone'],
            projection_dim=config['projection_dim'],
            temperature=config['temperature']
        ).to(device)

        lr = 0.001 if str(device) == 'mps' else 0.0003
        ssl_optimizer = optim.Adam(ssl_model.parameters(), lr=lr, weight_decay=1e-4)

        ssl_model = train_ssl_model(
            model=ssl_model,
            train_loader=dataloaders['ssl']['train'],
            val_loader=dataloaders['ssl']['val'],
            optimizer=ssl_optimizer,
            device=device,
            config=config,
            resume=args.resume
        )

    if args.mode in ['finetune', 'all']:
        if args.mode == 'all' and 'ssl_model' in locals():
            logger.info("使用刚训练好的SSL模型进行微调")
        else:
            ssl_model = None
            last_model_path = Path(config['checkpoint_dir']) / 'last_ssl_model.pt'
            best_model_path = Path(config['checkpoint_dir']) / 'best_ssl_model.pt'

            if last_model_path.exists() and args.resume:
                checkpoint_path = last_model_path
            elif best_model_path.exists():
                checkpoint_path = best_model_path
            else:
                checkpoint_path = None

            if checkpoint_path:
                ssl_model = SimCLREncoder(
                    backbone_name=config['backbone'],
                    projection_dim=config['projection_dim'],
                    temperature=config['temperature']
                ).to(device)
                checkpoint = torch.load(checkpoint_path, map_location=device)
                ssl_model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"从 {checkpoint_path} 加载了SSL模型")
            else:
                logger.warning(f"未找到SSL模型检查点，将使用预训练的ResNet")

        classifier = ChestXrayClassifier(
            ssl_model=ssl_model,
            backbone_name=config['backbone'],
            pretrained=True if ssl_model is None else False,
            num_classes=config['num_classes'],
            freeze_backbone=False
        ).to(device)

        lr = 0.0005 if str(device) == 'mps' else 0.0001
        classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

        criterion = nn.BCEWithLogitsLoss()

        classifier = train_classifier(
            model=classifier,
            train_loader=dataloaders['supervised']['train'],
            val_loader=dataloaders['supervised']['val'],
            optimizer=classifier_optimizer,
            criterion=criterion,
            device=device,
            config=config,
            disease_labels=disease_labels,
            resume=args.resume
        )

        if args.mode in ['evaluate', 'all', 'finetune']:
            if args.mode == 'evaluate':
                best_model_path = Path(config['checkpoint_dir']) / 'best_classifier_model.pt'
                if best_model_path.exists():
                    classifier = ChestXrayClassifier(
                        backbone_name=config['backbone'],
                        pretrained=False,
                        num_classes=config['num_classes']
                    ).to(device)
                    checkpoint = torch.load(best_model_path, map_location=device)
                    classifier.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"从 {best_model_path} 加载了分类器")
                else:
                    logger.error(f"在 {best_model_path} 未找到分类器检查点")
                    return

            metrics = evaluate_model(
                model=classifier,
                test_loader=dataloaders['supervised']['test'],
                device=device,
                disease_labels=disease_labels
            )

            logger.info("训练和评估完成！")

            metrics_path = Path(config['checkpoint_dir']) / 'test_metrics.txt'
            with open(metrics_path, 'w') as f:
                f.write("测试集评估指标:\n\n")
                f.write(f"整体准确率: {metrics['overall']['accuracy']:.4f}\n")
                f.write(f"整体精确率: {metrics['overall']['precision']:.4f}\n")
                f.write(f"整体召回率: {metrics['overall']['recall']:.4f}\n")
                f.write(f"整体F1分数: {metrics['overall']['f1']:.4f}\n")
                f.write(f"平均AUC: {metrics['overall']['avg_auc']:.4f}\n\n")

                f.write("每个疾病的指标:\n")
                for disease, metric in metrics['per_disease'].items():
                    f.write(f"{disease}:\n")
                    f.write(f"  精确率: {metric['precision']:.4f}\n")
                    f.write(f"  召回率: {metric['recall']:.4f}\n")
                    f.write(f"  F1分数: {metric['f1']:.4f}\n")
                    f.write(f"  AUC: {metric['auc']:.4f}\n\n")


if __name__ == '__main__':
    main()

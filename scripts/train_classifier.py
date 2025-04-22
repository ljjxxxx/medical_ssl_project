import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.data import download_chestxray14_dataset, create_dataloaders
from src.models import SimCLREncoder, ChestXrayClassifier
from src.training import train_classifier
from src.utils import get_available_device

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """
    设置随机种子以确保可重复性

    Args:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description='胸部X光分类器训练')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (auto, mps, cuda, 或 cpu)')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='模型检查点保存目录')
    parser.add_argument('--ssl-model', type=str, default=None,
                        help='预训练SSL模型路径 (如果不提供，将使用预训练的ResNet)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载的工作进程数')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断处继续训练')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--freeze-backbone', action='store_true',
                        help='是否冻结骨干网络')
    args = parser.parse_args()

    # 设置设备
    if args.device is None or args.device == 'auto':
        device = get_available_device()
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")

    # 设置随机种子
    set_seed(42)

    # 优化Apple Silicon (M1/M2)性能
    if str(device) == 'mps':
        logger.info("为Apple Silicon (M1/M2)优化参数")
        num_workers = min(os.cpu_count() - 1, args.num_workers)
        logger.info(f"使用 {num_workers} 个数据加载工作进程")
        lr = 0.0005  # MPS设备的更高学习率
    else:
        num_workers = args.num_workers
        lr = args.lr

    # 配置参数
    config = {
        'finetune_epochs': args.epochs,
        'batch_size': args.batch_size,
        'checkpoint_dir': args.checkpoint_dir,
        'backbone': 'resnet50',
    }

    # 创建检查点目录
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 下载和准备数据集
    dataset_path = download_chestxray14_dataset()

    dataloaders_with_labels = create_dataloaders(
        data_dir=dataset_path,
        batch_size=config['batch_size'],
        num_workers=num_workers
    )

    disease_labels = dataloaders_with_labels['disease_labels']
    config['num_classes'] = len(disease_labels)

    # 加载SSL模型（如果有）
    ssl_model = None
    if args.ssl_model:
        ssl_model_path = Path(args.ssl_model)
        if ssl_model_path.exists():
            logger.info(f"加载SSL模型: {ssl_model_path}")
            ssl_model = SimCLREncoder(
                backbone_name=config['backbone'],
                projection_dim=128,
                temperature=0.1
            ).to(device)

            checkpoint = torch.load(ssl_model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                ssl_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                ssl_model.load_state_dict(checkpoint)

            logger.info("SSL模型加载成功")
        else:
            logger.warning(f"未找到SSL模型: {ssl_model_path}")

    # 创建分类器
    classifier = ChestXrayClassifier(
        ssl_model=ssl_model,
        backbone_name=config['backbone'],
        pretrained=True if ssl_model is None else False,
        num_classes=config['num_classes'],
        freeze_backbone=args.freeze_backbone
    ).to(device)

    # 创建优化器和损失函数
    classifier_optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 训练分类器
    classifier = train_classifier(
        model=classifier,
        train_loader=dataloaders_with_labels['supervised']['train'],
        val_loader=dataloaders_with_labels['supervised']['val'],
        optimizer=classifier_optimizer,
        criterion=criterion,
        device=device,
        config=config,
        disease_labels=disease_labels,
        resume=args.resume
    )

    logger.info("分类器训练完成!")


if __name__ == '__main__':
    main()
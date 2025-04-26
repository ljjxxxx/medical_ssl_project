import os
import torch
import numpy as np
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # 设置后端以在没有GUI的服务器上生成图表

from src.models import ChestXrayClassifier
from src.data import create_dataloaders
from src.utils import calculate_metrics

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, test_loader, device, disease_labels, output_dir="./evaluation_results"):
    """
    评估模型在测试数据集上的性能并保存结果

    Args:
        model: 训练好的分类器模型
        test_loader: 测试数据集的DataLoader
        device: 运行评估的设备
        disease_labels: 疾病名称列表
        output_dir: 保存评估结果的目录

    Returns:
        metrics_dict: 包含评估指标的字典
    """
    logger.info("在测试集上评估模型")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 定义每种疾病的自定义阈值
    disease_thresholds = {
        'Consolidation': 0.12,  # 肺实变，临床重要性高
        'Edema': 0.12,  # 肺水肿，可能危及生命
        'Fibrosis': 0.04,  # 肺纤维化，慢性严重疾病
        'Pneumonia': 0.02,  # 肺炎，高度临床相关
        'Pleural_Thickening': 0.07,  # 胸膜增厚，需要关注
        'Nodule': 0.13,  # 肺结节，癌症风险，不能漏诊
        'Cardiomegaly': 0.15,  # 心脏肥大，心脏疾病指标
        'Atelectasis': 0.25,  # 肺不张
        'Infiltration': 0.25,  # 肺浸润
        'Hernia': 0.05,  # 疝气
        'Mass': 0.20,  # 肿块，可能恶性
        'Pneumothorax': 0.12,  # 气胸，医疗紧急情况
        'Emphysema': 0.15,  # 肺气肿，性能相对较好
        'Effusion': 0.30,  # 胸腔积液，当前最佳性能疾病
    }

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

            # 计算概率
            probs = torch.sigmoid(logits)

            # 使用自定义阈值进行预测
            preds = torch.zeros_like(probs)
            for i, disease in enumerate(disease_labels):
                threshold = disease_thresholds.get(disease, 0.3)  # 默认使用0.3作为阈值
                preds[:, i] = (probs[:, i] >= threshold).float()

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 计算评估指标
    metrics, disease_metrics = calculate_metrics(all_labels, all_preds, all_probs, disease_labels)

    # 打印整体指标
    logger.info(f"测试集准确率: {metrics['accuracy']:.4f}")
    logger.info(f"测试集精确率: {metrics['precision']:.4f}")
    logger.info(f"测试集召回率: {metrics['recall']:.4f}")
    logger.info(f"测试集F1分数: {metrics['f1']:.4f}")
    logger.info(f"测试集平均AUC: {metrics['avg_auc']:.4f}")

    # 打印每种疾病的指标
    logger.info("每种疾病的指标:")
    for disease, metric in disease_metrics.items():
        logger.info(f"{disease}:")
        logger.info(f"  精确率: {metric['precision']:.4f}")
        logger.info(f"  召回率: {metric['recall']:.4f}")
        logger.info(f"  F1分数: {metric['f1']:.4f}")
        logger.info(f"  AUC: {metric['auc']:.4f}")

    # 将指标保存到文件
    metrics_path = output_path / 'test_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("测试集评估指标:\n\n")
        f.write(f"整体准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"整体精确率: {metrics['precision']:.4f}\n")
        f.write(f"整体召回率: {metrics['recall']:.4f}\n")
        f.write(f"整体F1分数: {metrics['f1']:.4f}\n")
        f.write(f"平均AUC: {metrics['avg_auc']:.4f}\n\n")

        # 记录使用的疾病阈值
        f.write("使用的疾病阈值:\n")
        for disease in disease_labels:
            threshold = disease_thresholds.get(disease, 0.3)
            f.write(f"{disease}: {threshold:.2f}\n")
        f.write("\n")

        f.write("每种疾病的指标:\n")
        for disease, metric in disease_metrics.items():
            f.write(f"{disease}:\n")
            f.write(f"  精确率: {metric['precision']:.4f}\n")
            f.write(f"  召回率: {metric['recall']:.4f}\n")
            f.write(f"  F1分数: {metric['f1']:.4f}\n")
            f.write(f"  AUC: {metric['auc']:.4f}\n\n")

    # 创建F1分数柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(disease_labels, [disease_metrics[d]['f1'] for d in disease_labels])
    plt.xticks(rotation=90)
    plt.ylabel('F1分数')
    plt.title('按疾病的F1分数')
    plt.tight_layout()
    plt.savefig(output_path / 'disease_f1_scores.png')
    plt.close()

    # 创建AUC柱状图
    plt.figure(figsize=(8, 6))
    auc_values = []
    valid_diseases = []
    for disease in disease_labels:
        if not np.isnan(disease_metrics[disease]['auc']):
            auc_values.append(disease_metrics[disease]['auc'])
            valid_diseases.append(disease)

    if auc_values:
        plt.bar(valid_diseases, auc_values)
        plt.xticks(rotation=90)
        plt.ylabel('AUC')
        plt.title('按疾病的AUC')
        plt.tight_layout()
        plt.savefig(output_path / 'disease_auc_scores.png')
    plt.close()

    # 创建召回率柱状图
    plt.figure(figsize=(8, 6))
    plt.bar(disease_labels, [disease_metrics[d]['recall'] for d in disease_labels])
    plt.xticks(rotation=90)
    plt.ylabel('召回率')
    plt.title('按疾病的召回率')
    plt.tight_layout()
    plt.savefig(output_path / 'disease_recall_scores.png')
    plt.close()

    logger.info(f"评估结果已保存到 {output_path}")

    return {
        'overall': metrics,
        'per_disease': disease_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='评估胸部X光疾病分类模型')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/best_classifier_model.pt',
                        help='模型检查点文件路径')
    parser.add_argument('--data-dir', type=str, default='./datasets/chestxray14',
                        help='数据集目录路径')
    parser.add_argument('--batch-size', type=int, default=32, help='评估批量大小')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results',
                        help='保存评估结果的目录')
    parser.add_argument('--device', type=str, default=None,
                        help='运行评估的设备 (cuda, mps, 或 cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载的工作进程数')
    parser.add_argument('--strict-loading', action='store_true',
                        help='使用严格的state_dict加载 (设为False以跳过不匹配的键)')
    args = parser.parse_args()

    # 设置设备
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else
                              "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")

    # 检查检查点是否存在
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"未找到检查点文件: {checkpoint_path}")
        return

    # 加载数据
    logger.info(f"从 {args.data_dir} 加载数据集")
    try:
        dataloaders_with_labels = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        test_loader = dataloaders_with_labels['supervised']['test']
        disease_labels = dataloaders_with_labels['disease_labels']

        num_classes = len(disease_labels)
        logger.info(f"加载了具有 {num_classes} 种疾病类别的数据集")

    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        return

    # 创建并加载模型
    logger.info(f"从检查点加载模型: {checkpoint_path}")
    try:
        model = ChestXrayClassifier(
            backbone_name='resnet50',  # 使用ResNet50作为骨干网络
            pretrained=False,
            num_classes=num_classes
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 根据需要使用strict=False进行部分加载
        strict_loading = args.strict_loading
        logger.info(f"使用严格加载: {strict_loading}")

        model.load_state_dict(checkpoint['model_state_dict'], strict=strict_loading)
        logger.info("模型加载成功")

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # 评估模型
    try:
        evaluate_model(
            model=model,
            test_loader=test_loader,
            device=device,
            disease_labels=disease_labels,
            output_dir=args.output_dir
        )
        logger.info("评估成功完成")

    except Exception as e:
        logger.error(f"评估过程中出错: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    main()
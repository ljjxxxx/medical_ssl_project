import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def calculate_metrics(all_labels, all_preds, all_probs, disease_labels):
    """
    计算模型评估指标

    Args:
        all_labels: 真实标签
        all_preds: 预测标签
        all_probs: 预测概率
        disease_labels: 疾病标签列表

    Returns:
        metrics: 整体评估指标
        disease_metrics: 每种疾病的评估指标
    """
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

    return metrics, disease_metrics
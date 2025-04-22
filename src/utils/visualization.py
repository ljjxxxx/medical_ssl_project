import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path

# 设置后端以在没有GUI的服务器上生成图表
matplotlib.use('Agg')
# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False


def create_bar_chart(predictions):
    """
    创建疾病预测概率条形图

    Args:
        predictions: 疾病预测概率字典

    Returns:
        fig: 条形图Figure对象
    """
    # 延迟导入，避免循环依赖
    from src.app.disease_info import get_disease_info

    diseases = list(predictions.keys())
    probs = [predictions[d] for d in diseases]
    sorted_indices = np.argsort(probs)[::-1]  # 降序排序

    sorted_diseases_en = [diseases[i] for i in sorted_indices]
    sorted_diseases = [get_disease_info(disease)['translation'] for disease in sorted_diseases_en]
    sorted_probs = [probs[i] for i in sorted_indices]

    # 多级风险颜色编码
    colors = []
    for prob in sorted_probs:
        if prob >= 0.7:  # 高风险 - 深红色
            colors.append('#b91c1c')
        elif prob >= 0.5:  # 中高风险 - 红色
            colors.append('#ef4444')
        elif prob >= 0.3:  # 中度风险 - 橙色
            colors.append('#f97316')
        else:  # 低风险 - 绿色
            colors.append('#10b981')

    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(sorted_diseases, sorted_probs, color=colors, alpha=0.85)

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{sorted_probs[i]:.3f}',
            va='center',
            fontweight='bold' if sorted_probs[i] >= 0.3 else 'normal'  # 降低加粗阈值
        )

    # 添加三个阈值线
    ax.axvline(x=0.7, color='#b91c1c', linestyle='--', alpha=0.7, label='高风险 (0.7)')
    ax.axvline(x=0.5, color='#ef4444', linestyle='--', alpha=0.7, label='中高风险 (0.5)')
    ax.axvline(x=0.3, color='#f97316', linestyle='--', alpha=0.7, label='中度风险 (0.3)')

    # 区域着色
    ax.axvspan(0.7, 1.0, facecolor='#b91c1c', alpha=0.1)
    ax.axvspan(0.5, 0.7, facecolor='#ef4444', alpha=0.1)
    ax.axvspan(0.3, 0.5, facecolor='#f97316', alpha=0.1)

    ax.set_xlabel('概率', fontsize=12, fontweight='bold')
    ax.set_ylabel('发现', fontsize=12, fontweight='bold')
    ax.set_title('胸部X光疾病筛查结果 (筛查模式)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right')

    plt.tight_layout()
    return fig


def plot_training_curves(train_state, output_dir, plot_type='ssl'):
    """
    绘制训练曲线

    Args:
        train_state: 训练状态字典
        output_dir: 输出目录
        plot_type: 绘图类型 ('ssl' 或 'classifier')

    Returns:
        output_path: 保存的图表路径
    """
    output_path = Path(output_dir)

    if plot_type == 'ssl':
        train_losses = train_state.get('train_losses', [])
        val_losses = train_state.get('val_losses', [])

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SSL Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_file = output_path / 'ssl_training_curve.png'
        plt.savefig(output_file)
        plt.close()

    else:  # classifier
        train_losses = train_state.get('train_losses', [])
        val_losses = train_state.get('val_losses', [])
        val_metrics = train_state.get('val_metrics', [])

        plt.figure(figsize=(12, 8))

        # Plot 1: Training and Validation Loss
        plt.subplot(2, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: F1 Score and Accuracy
        plt.subplot(2, 2, 2)
        if val_metrics:
            plt.plot([m.get('f1', 0) for m in val_metrics], label='F1 Score')
            plt.plot([m.get('accuracy', 0) for m in val_metrics], label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('F1 and Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: Precision and Recall
        plt.subplot(2, 2, 3)
        if val_metrics:
            plt.plot([m.get('precision', 0) for m in val_metrics], label='Precision')
            plt.plot([m.get('recall', 0) for m in val_metrics], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.title('Precision and Recall')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: AUC
        plt.subplot(2, 2, 4)
        if val_metrics:
            plt.plot([m.get('auc', 0) for m in val_metrics], label='AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Area Under ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        output_file = output_path / 'classifier_training_curve.png'
        plt.savefig(output_file)
        plt.close()

    return output_file

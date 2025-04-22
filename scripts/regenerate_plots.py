import os
import json
import matplotlib.pyplot as plt
import matplotlib
import argparse
from pathlib import Path

# 设置后端以在没有GUI的服务器上生成图表
matplotlib.use('Agg')

from src.utils.visualization import plot_training_curves


def regenerate_ssl_plot(json_file, output_dir):
    """
    重新生成SSL训练曲线，使用英文标签

    Args:
        json_file: 训练状态JSON文件
        output_dir: 输出目录

    Returns:
        bool: 是否成功
    """
    print(f"从 {json_file} 读取SSL训练数据")
    try:
        with open(json_file, 'r') as f:
            train_state = json.load(f)

        output_path = plot_training_curves(train_state, output_dir, plot_type='ssl')
        print(f"SSL训练曲线已保存到 {output_path}")
        return True

    except Exception as e:
        print(f"重新生成SSL图表时出错: {e}")
        return False


def regenerate_classifier_plot(json_file, output_dir):
    """
    重新生成分类器训练曲线，使用英文标签

    Args:
        json_file: 训练状态JSON文件
        output_dir: 输出目录

    Returns:
        bool: 是否成功
    """
    print(f"从 {json_file} 读取分类器训练数据")
    try:
        with open(json_file, 'r') as f:
            train_state = json.load(f)

        output_path = plot_training_curves(train_state, output_dir, plot_type='classifier')
        print(f"分类器训练曲线已保存到 {output_path}")
        return True

    except Exception as e:
        print(f"重新生成分类器图表时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='使用英文标签重新生成训练曲线图')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='包含训练状态JSON文件的目录')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='保存重新生成的图表的目录 (默认为checkpoint-dir)')
    parser.add_argument('--ssl-only', action='store_true',
                        help='仅重新生成SSL训练曲线')
    parser.add_argument('--classifier-only', action='store_true',
                        help='仅重新生成分类器训练曲线')
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir if args.output_dir else args.checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查训练状态文件
    ssl_json = checkpoint_dir / 'ssl_train_state.json'
    classifier_json = checkpoint_dir / 'classifier_train_state.json'

    if not args.classifier_only and ssl_json.exists():
        regenerate_ssl_plot(ssl_json, output_dir)
    elif not args.classifier_only:
        print(f"在 {ssl_json} 未找到SSL训练状态文件")

    if not args.ssl_only and classifier_json.exists():
        regenerate_classifier_plot(classifier_json, output_dir)
    elif not args.ssl_only:
        print(f"在 {classifier_json} 未找到分类器训练状态文件")

    print("图表重新生成完成")


if __name__ == '__main__':
    main()
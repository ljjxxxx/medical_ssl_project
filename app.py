import os
import torch
import argparse
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

from models import ChestXrayClassifier
from data_utils import get_eval_transforms, download_chestxray14_dataset

# 设置matplotlib支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

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
        logger.warning("PyTorch版本不支持MPS（需要PyTorch 1.12+）")

    if torch.cuda.is_available():
        logger.info("CUDA加速可用")
        return torch.device("cuda")

    logger.info("使用CPU进行计算（无GPU加速可用）")
    return torch.device("cpu")


def load_model(model_path, num_classes=14, device=None):
    if device is None:
        device = get_available_device()

    model = ChestXrayClassifier(
        backbone_name='resnet50',
        pretrained=False,
        num_classes=num_classes
    ).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logger.info(f"从 {model_path} 加载了模型")
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        raise

    model.eval()

    return model, device


def predict_image(model, image_path, disease_labels, device, transform=None):
    if transform is None:
        transform = get_eval_transforms()

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            logits = outputs['logits'].squeeze(0)

        predictions = model.predict_diseases(logits, disease_labels)

        return predictions, img

    except Exception as e:
        logger.error(f"预测过程中出错: {e}")
        raise


def create_bar_chart(predictions):
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


def get_disease_info(english_name):
    """返回每种疾病的描述和翻译"""
    disease_dict = {
        'Atelectasis': {
            'name': 'Atelectasis',
            'translation': '肺不张',
            'description': '肺部或部分肺部的塌陷或闭合'
        },
        'Consolidation': {
            'name': 'Consolidation',
            'translation': '肺实变',
            'description': '肺组织被液体而非空气填充'
        },
        'Infiltration': {
            'name': 'Infiltration',
            'translation': '肺浸润',
            'description': '肺部存在不应出现的物质'
        },
        'Pneumothorax': {
            'name': 'Pneumothorax',
            'translation': '气胸',
            'description': '胸腔内有空气导致肺部塌陷'
        },
        'Edema': {
            'name': 'Edema',
            'translation': '肺水肿',
            'description': '肺组织中过多的液体'
        },
        'Emphysema': {
            'name': 'Emphysema',
            'translation': '肺气肿',
            'description': '肺部气泡（肺泡）受损'
        },
        'Fibrosis': {
            'name': 'Fibrosis',
            'translation': '肺纤维化',
            'description': '肺组织疤痕形成'
        },
        'Effusion': {
            'name': 'Effusion',
            'translation': '胸腔积液',
            'description': '肺部和胸腔之间的液体'
        },
        'Pneumonia': {
            'name': 'Pneumonia',
            'translation': '肺炎',
            'description': '肺部感染'
        },
        'Pleural_Thickening': {
            'name': 'Pleural Thickening',
            'translation': '胸膜增厚',
            'description': '胸膜（肺部周围的衬里）增厚'
        },
        'Cardiomegaly': {
            'name': 'Cardiomegaly',
            'translation': '心脏肥大',
            'description': '心脏扩大'
        },
        'Nodule': {
            'name': 'Nodule',
            'translation': '肺结节',
            'description': '肺部的小圆形生长物'
        },
        'Mass': {
            'name': 'Mass',
            'translation': '肺肿块',
            'description': '肺部较大的生长物或肿瘤'
        },
        'Hernia': {
            'name': 'Hernia',
            'translation': '疝气',
            'description': '器官通过腔壁的开口突出'
        }
    }

    return disease_dict.get(english_name, {'name': english_name, 'translation': '', 'description': ''})


def format_html_results(predictions):
    sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

    html = "<div style='max-width: 800px; margin: 0 auto;'>"

    # 筛查模式说明
    html += f"""
    <div style='background-color: #f3f4f6; border-left: 4px solid #3b82f6; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
        <h3 style='margin-top: 0; color: #1d4ed8;'>筛查模式</h3>
        <p style='margin-bottom: 0;'>此模式优先考虑敏感性，使用较低的阈值(0.3)来减少漏诊风险。由于阈值降低，可能会出现一些假阳性结果，最终诊断需专业医生判断。</p>
    </div>
    """

    # 风险级别分类
    high_risk = []
    medium_high_risk = []
    medium_risk = []

    for disease, prob in sorted_preds:
        info = get_disease_info(disease)
        if prob >= 0.7:
            high_risk.append(f"{info['translation']} ({info['name']}): {prob:.3f}")
        elif prob >= 0.5:
            medium_high_risk.append(f"{info['translation']} ({info['name']}): {prob:.3f}")
        elif prob >= 0.3:
            medium_risk.append(f"{info['translation']} ({info['name']}): {prob:.3f}")

    # 高风险发现
    if high_risk:
        html += f"""
        <div style='background-color: #fef2f2; border-left: 4px solid #b91c1c; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #b91c1c;'>高风险发现 (≥0.7)</h3>
            <p style='margin-bottom: 0;'>{', '.join(high_risk)}</p>
            <p style='margin-top: 8px; font-size: 0.9em; color: #6b7280;'>强烈建议进一步专业检查</p>
        </div>
        """

    # 中高风险发现
    if medium_high_risk:
        html += f"""
        <div style='background-color: #fff5f5; border-left: 4px solid #ef4444; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #ef4444;'>中高风险发现 (0.5-0.7)</h3>
            <p style='margin-bottom: 0;'>{', '.join(medium_high_risk)}</p>
            <p style='margin-top: 8px; font-size: 0.9em; color: #6b7280;'>建议进一步专业检查</p>
        </div>
        """

    # 中度风险发现
    if medium_risk:
        html += f"""
        <div style='background-color: #fff7ed; border-left: 4px solid #f97316; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #f97316;'>中度风险发现 (0.3-0.5)</h3>
            <p style='margin-bottom: 0;'>{', '.join(medium_risk)}</p>
            <p style='margin-top: 8px; font-size: 0.9em; color: #6b7280;'>可能需要关注，建议咨询医生</p>
        </div>
        """

    # 无风险发现
    if not high_risk and not medium_high_risk and not medium_risk:
        html += f"""
        <div style='background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #047857;'>未检测到显著风险</h3>
            <p style='margin-bottom: 0;'>未检测到超过筛查阈值(0.3)的显著发现。</p>
        </div>
        """

    html += """
    <h3>所有发现详情</h3>
    <table style='width: 100%; border-collapse: collapse;'>
        <thead>
            <tr style='background-color: #f3f4f6;'>
                <th style='text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb;'>发现</th>
                <th style='text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb;'>描述</th>
                <th style='text-align: center; padding: 8px; border-bottom: 2px solid #e5e7eb;'>概率</th>
                <th style='text-align: center; padding: 8px; border-bottom: 2px solid #e5e7eb;'>风险级别</th>
            </tr>
        </thead>
        <tbody>
    """

    for disease, prob in sorted_preds:
        info = get_disease_info(disease)

        # 确定风险级别和颜色
        if prob >= 0.7:
            risk_level = "高风险"
            status_color = "#b91c1c"
            bg_color = f"background-color: rgba(239, 68, 68, 0.15);"
        elif prob >= 0.5:
            risk_level = "中高风险"
            status_color = "#ef4444"
            bg_color = f"background-color: rgba(239, 68, 68, 0.1);"
        elif prob >= 0.3:
            risk_level = "中度风险"
            status_color = "#f97316"
            bg_color = f"background-color: rgba(249, 115, 22, 0.1);"
        else:
            risk_level = "低风险"
            status_color = "#10b981"
            bg_color = ""

        html += f"""
        <tr style='{bg_color} border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 8px;'><strong>{info['translation']}</strong><br><span style='color: #6b7280; font-size: 0.9em;'>{info['name']}</span></td>
            <td style='padding: 8px;'>{info['description']}</td>
            <td style='text-align: center; padding: 8px;'><span style='font-weight: {"bold" if prob >= 0.3 else "normal"};'>{prob:.3f}</span></td>
            <td style='text-align: center; padding: 8px;'>
                <span style='display: inline-block; padding: 4px 8px; border-radius: 9999px; background-color: {status_color}; color: white; font-size: 0.85em;'>
                    {risk_level}
                </span>
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>

    <div style='margin-top: 20px; padding: 12px; background-color: #f3f4f6; border-radius: 4px; font-size: 0.9em;'>
        <p style='margin: 0;'><strong>筛查模式说明：</strong>该系统使用较低阈值(0.3)来提高敏感性，减少漏诊风险。
        这可能导致一些假阳性结果。此AI分析仅供参考筛查，不构成医疗诊断。请务必咨询医疗专业人员以获取适当的诊断和治疗。</p>
    </div>
    </div>
    """

    return html


def gradio_predict(image):
    global MODEL, DEVICE, DISEASE_LABELS, TRANSFORM

    try:
        temp_path = "temp_upload.png"
        image.save(temp_path)

        predictions, img = predict_image(MODEL, temp_path, DISEASE_LABELS, DEVICE, TRANSFORM)

        fig = create_bar_chart(predictions)

        html_results = format_html_results(predictions)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return img, fig, html_results

    except Exception as e:
        error_msg = f"预测过程中出错: {str(e)}"
        logger.error(error_msg)
        return image, None, f"<div style='color: red; padding: 20px; border: 1px solid red; border-radius: 4px;'><h3>错误</h3><p>{error_msg}</p></div>"


def main():
    parser = argparse.ArgumentParser(description='胸部X光疾病筛查应用')
    parser.add_argument('--model', type=str, default='./checkpoints/best_classifier_model.pt',
                        help='训练好的模型检查点路径')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (auto, mps, cuda, 或 cpu)')
    parser.add_argument('--port', type=int, default=7860, help='Gradio服务器端口')
    args = parser.parse_args()

    if args.device is None or args.device == 'auto':
        device = get_available_device()
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")

    dataset_path = download_chestxray14_dataset()
    disease_labels_path = Path(dataset_path) / "disease_labels.txt"

    if disease_labels_path.exists():
        with open(disease_labels_path, "r") as f:
            disease_labels = f.read().strip().split("\n")
    else:
        disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
                          'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
                          'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']

    transform = get_eval_transforms()

    model_path = Path(args.model)
    if not model_path.exists():
        logger.warning(f"在 {model_path} 未找到模型。创建一个带有预训练骨干网络的新模型。")
        model = ChestXrayClassifier(
            backbone_name='resnet50',
            pretrained=True,
            num_classes=len(disease_labels)
        ).to(device)
    else:
        model, device = load_model(model_path, num_classes=len(disease_labels), device=device)

    global MODEL, DEVICE, DISEASE_LABELS, TRANSFORM
    MODEL = model
    DEVICE = device
    DISEASE_LABELS = disease_labels
    TRANSFORM = transform

    disease_info = []
    for disease in disease_labels:
        info = get_disease_info(disease)
        disease_info.append(f"- **{info['translation']} ({info['name']})**: {info['description']}")

    disease_md = "\n".join(disease_info)

    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="red",
        neutral_hue="gray"
    ).set(
        body_text_size="16px",
        button_primary_background_fill="*primary_500",
        button_primary_text_color="white",
        block_title_text_weight="600"
    )

    with gr.Blocks(theme=theme, title="胸部X光疾病筛查器") as iface:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 1rem">
                <h1>胸部X光疾病筛查器</h1>
                <p style="font-size: 1.2rem;">使用AI辅助筛查胸部X光片，检测潜在异常</p>
                <div style="background-color: #eff6ff; padding: 10px; border-radius: 8px; display: inline-block; margin-top: 10px;">
                    <p style="margin: 0; color: #1d4ed8;"><strong>筛查模式:</strong> 使用较低阈值(0.3)提高敏感性，减少漏诊风险</p>
                </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.TabItem("分析X光片"):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="上传X光图像")
                        examples = gr.Examples(
                            examples=[
                                "./examples/example1.png",
                                "./examples/example2.png",
                                "./examples/example3.png",
                                "./examples/example12.png",
                                "./examples/example4.png",
                                "./examples/example5.png",
                                "./examples/example6.png",
                                "./examples/example11.png",
                                "./examples/example7.png",
                                "./examples/example8.png",
                            ],
                            inputs=input_image,
                            label="示例X光片"
                        )
                        analyze_btn = gr.Button("分析X光片", variant="primary")

                    with gr.Column(scale=2):
                        with gr.Row():
                            output_image = gr.Image(type="pil", label="原始X光片")

                        with gr.Row():
                            output_plot = gr.Plot(label="疾病概率")

                output_html = gr.HTML(label="分析结果")

                analyze_btn.click(
                    fn=gradio_predict,
                    inputs=input_image,
                    outputs=[output_image, output_plot, output_html]
                )

            with gr.TabItem("关于"):
                gr.HTML(
                    f"""
                    <div style='max-width: 900px; margin: 0 auto; padding: 20px;'>
                        <h1>胸部X光疾病筛查</h1>

                        <div style='background-color: #eff6ff; padding: 16px; border-radius: 8px; margin-bottom: 20px;'>
                            <h2 style='margin-top: 0; color: #1d4ed8;'>筛查模式说明</h2>
                            <p>本系统采用<strong>筛查模式</strong>，使用较低的概率阈值(0.3)来提高敏感性，减少漏诊风险。与标准诊断模式(阈值0.5)相比，
                            筛查模式可能会产生更多的假阳性结果，但能更好地捕捉潜在异常。</p>
                            <p>风险级别分类:</p>
                            <ul>
                                <li><strong style='color: #b91c1c;'>高风险 (≥0.7)</strong>: 强烈建议进一步专业检查</li>
                                <li><strong style='color: #ef4444;'>中高风险 (0.5-0.7)</strong>: 建议进一步专业检查</li>
                                <li><strong style='color: #f97316;'>中度风险 (0.3-0.5)</strong>: 可能需要关注，建议咨询医生</li>
                                <li><strong style='color: #10b981;'>低风险 (<0.3)</strong>: 风险较低</li>
                            </ul>
                        </div>

                        <p>上传一张胸部X光图像来检测潜在的疾病。该模型在NIH ChestX-ray14数据集上训练，可以检测14种不同的胸部疾病。</p>

                        <h2>工作原理</h2>
                        <ol>
                            <li>上传您的胸部X光图像</li>
                            <li>AI模型将分析图像</li>
                            <li>查看显示每种潜在发现概率分数的结果</li>
                            <li>根据风险级别获取建议</li>
                        </ol>

                        <h2>可检测的疾病</h2>
                        <div style='display: grid; grid-template-columns: repeat(auto-fill, minmax(400px, 1fr)); gap: 16px;'>
                    """
                )

                for disease in disease_labels:
                    info = get_disease_info(disease)
                    gr.HTML(
                        f"""
                        <div style='border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin-bottom: 16px;'>
                            <h3 style='margin-top: 0; color: #1f2937;'>{info['translation']} <span style='font-size: 0.9em; color: #6b7280;'>({info['name']})</span></h3>
                            <p>{info['description']}</p>
                        </div>
                        """
                    )

                gr.HTML(
                    """
                        </div>

                        <h2>关于结果</h2>
                        <p>模型预测每种疾病存在的概率。结果显示为：</p>
                        <ul>
                            <li>显示所有概率的图表</li>
                            <li>按风险级别分类的详细报告</li>
                            <li>针对不同风险级别的建议</li>
                        </ul>

                        <div style='background-color: #f3f4f6; padding: 16px; border-radius: 8px; margin-top: 24px;'>
                            <p style='margin: 0;'><strong>免责声明</strong>：此工具仅用于教育和研究目的，以及初步筛查。它不能替代专业医疗诊断。请务必咨询医疗专业人员进行正式评估。</p>
                        </div>
                    </div>
                    """
                )

            with gr.TabItem("帮助与常见问题"):
                gr.HTML(
                    """
                    <div style='max-width: 800px; margin: 0 auto; padding: 20px;'>
                        <h2>常见问题解答</h2>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>什么是筛查模式？</h3>
                            <p>筛查模式使用较低的阈值(0.3)来提高检测的敏感性，减少漏诊风险。这种模式适合初步筛查，
                            而不是最终诊断。任何检测到的风险都应由医疗专业人员进一步评估。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>我应该上传什么类型的X光图像？</h3>
                            <p>该模型最适合标准PA（后前位）胸部X光片，格式为JPEG或PNG。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>这个模型有多准确？</h3>
                            <p>该模型在NIH ChestX-ray14数据集上训练，但像所有AI系统一样，它并不完美。
                            在筛查模式下，系统倾向于过度检测（更多假阳性）以减少漏诊风险。不应将其用于临床诊断。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>风险级别意味着什么？</h3>
                            <p>
                              <ul>
                                <li><strong style='color: #b91c1c;'>高风险 (≥0.7)</strong>: 模型高度确信存在异常，强烈建议专业检查</li>
                                <li><strong style='color: #ef4444;'>中高风险 (0.5-0.7)</strong>: 模型相当确信存在异常，建议专业检查</li>
                                <li><strong style='color: #f97316;'>中度风险 (0.3-0.5)</strong>: 可能存在异常，建议咨询医生</li>
                                <li><strong style='color: #10b981;'>低风险 (<0.3)</strong>: 模型认为异常可能性较低</li>
                              </ul>
                            </p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>我可以将此模型用于医疗诊断吗？</h3>
                            <p>不可以。此工具仅用于研究、教育目的和初步筛查。请务必咨询合格的医疗专业人员获取医疗建议和诊断。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>如果模型检测到潜在问题，我应该怎么做？</h3>
                            <p>如果系统检测到任何中度及以上风险，建议咨询医疗专业人员。此工具不能替代专业医疗建议。</p>
                        </div>

                        <div style='background-color: #f3f4f6; padding: 16px; border-radius: 8px; margin-top: 24px;'>
                            <h3>使用提示</h3>
                            <ul>
                                <li>上传清晰、高质量的X光图像</li>
                                <li>确保图像未被裁剪或过度压缩</li>
                                <li>记住筛查模式会产生更多潜在发现，需要专业人士进一步确认</li>
                                <li>将结果视为初步参考，而非诊断</li>
                            </ul>
                        </div>
                    </div>
                    """
                )

    examples_dir = Path("./examples")
    examples_dir.mkdir(exist_ok=True)

    iface.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
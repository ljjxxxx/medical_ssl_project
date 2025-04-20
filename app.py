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

    colors = plt.cm.RdYlGn_r(sorted_probs)

    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(sorted_diseases, sorted_probs, color=colors, alpha=0.8)

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{sorted_probs[i]:.3f}',
            va='center',
            fontweight='bold' if sorted_probs[i] >= 0.5 else 'normal'
        )

    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值 (0.5)')

    ax.axvspan(0.5, 1.0, facecolor='red', alpha=0.1)

    ax.set_xlabel('概率', fontsize=12, fontweight='bold')
    ax.set_ylabel('发现', fontsize=12, fontweight='bold')
    ax.set_title('胸部X光疾病预测结果', fontsize=14, fontweight='bold')
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

    positive_findings = []
    for disease, prob in sorted_preds:
        if prob >= 0.5:
            info = get_disease_info(disease)
            positive_findings.append(f"{info['translation']} ({info['name']})")

    if positive_findings:
        html += f"""
        <div style='background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #b91c1c;'>检测到阳性发现</h3>
            <p style='margin-bottom: 0;'>{', '.join(positive_findings)}</p>
        </div>
        """
    else:
        html += f"""
        <div style='background-color: #f0fdf4; border-left: 4px solid #10b981; padding: 16px; margin-bottom: 20px; border-radius: 4px;'>
            <h3 style='margin-top: 0; color: #047857;'>无阳性发现</h3>
            <p style='margin-bottom: 0;'>未检测到超过阈值(0.5)的显著发现。</p>
        </div>
        """

    html += """
    <h3>所有发现</h3>
    <table style='width: 100%; border-collapse: collapse;'>
        <thead>
            <tr style='background-color: #f3f4f6;'>
                <th style='text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb;'>发现</th>
                <th style='text-align: left; padding: 8px; border-bottom: 2px solid #e5e7eb;'>描述</th>
                <th style='text-align: center; padding: 8px; border-bottom: 2px solid #e5e7eb;'>概率</th>
                <th style='text-align: center; padding: 8px; border-bottom: 2px solid #e5e7eb;'>状态</th>
            </tr>
        </thead>
        <tbody>
    """

    for disease, prob in sorted_preds:
        info = get_disease_info(disease)
        status = "阳性" if prob >= 0.5 else "阴性"
        status_color = "#ef4444" if prob >= 0.5 else "#10b981"

        bg_color = ""
        if prob >= 0.5:
            intensity = int(255 - (prob - 0.5) * 40)
            bg_color = f"background-color: rgba(239, 68, 68, {prob - 0.4});"

        html += f"""
        <tr style='{bg_color} border-bottom: 1px solid #e5e7eb;'>
            <td style='padding: 8px;'><strong>{info['translation']}</strong><br><span style='color: #6b7280; font-size: 0.9em;'>{info['name']}</span></td>
            <td style='padding: 8px;'>{info['description']}</td>
            <td style='text-align: center; padding: 8px;'><span style='font-weight: {"bold" if prob >= 0.5 else "normal"};'>{prob:.3f}</span></td>
            <td style='text-align: center; padding: 8px;'>
                <span style='display: inline-block; padding: 4px 8px; border-radius: 9999px; background-color: {status_color}; color: white; font-size: 0.85em;'>
                    {status}
                </span>
            </td>
        </tr>
        """

    html += """
        </tbody>
    </table>

    <div style='margin-top: 20px; padding: 12px; background-color: #f3f4f6; border-radius: 4px; font-size: 0.9em;'>
        <p style='margin: 0;'><strong>注意：</strong>此AI分析仅供参考，不构成医疗建议。
        请咨询医疗专业人员以获取适当的诊断和治疗。</p>
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
    parser = argparse.ArgumentParser(description='胸部X光疾病分类应用')
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

    about_description = f"""
# 胸部X光疾病分类

上传一张胸部X光图像来检测潜在的疾病。该模型在NIH ChestX-ray14数据集上训练，可以检测14种不同的胸部疾病。

## 工作原理
1. 上传您的胸部X光图像
2. AI模型将分析图像
3. 查看显示每种潜在发现概率分数的结果
4. 0.5或更高（50%）的分数表示阳性发现

## 可检测的疾病
{disease_md}

## 关于结果
模型预测每种疾病存在的概率。结果显示为：
- 显示所有概率的图表
- 突出显示阳性发现的详细报告

**免责声明**：此工具仅用于教育和研究目的。它不能替代专业医疗诊断。
"""

    help_faq = """
## 常见问题解答

### 我应该上传什么类型的X光图像？
该模型最适合标准PA（后前位）胸部X光片，格式为JPEG或PNG。

### 这个模型有多准确？
该模型在NIH ChestX-ray14数据集上训练，但像所有AI系统一样，它并不完美。
不应将其用于临床诊断。

### 概率分数意味着什么？
0.5（50%）或更高的概率被认为是该疾病的阳性发现。
较高的概率表示模型对检测结果更有信心。

### 我可以将此模型用于医疗诊断吗？
不可以。此工具仅用于研究和教育目的。
请务必咨询合格的医疗专业人员获取医疗建议和诊断。

### 如果模型检测到潜在问题，我应该怎么做？
如果您担心自己的健康状况，请咨询医疗专业人员。
此工具不能替代专业医疗建议。
"""

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

    with gr.Blocks(theme=theme, title="胸部X光疾病分类器") as iface:
        gr.HTML(
            """
            <div style="text-align: center; margin-bottom: 1rem">
                <h1>胸部X光疾病分类器</h1>
                <p style="font-size: 1.2rem;">使用AI分析胸部X光片，检测14种不同疾病</p>
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
                                "./examples/example2.png"
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
                        <h1>胸部X光疾病分类</h1>

                        <p>上传一张胸部X光图像来检测潜在的疾病。该模型在NIH ChestX-ray14数据集上训练，可以检测14种不同的胸部疾病。</p>

                        <h2>工作原理</h2>
                        <ol>
                            <li>上传您的胸部X光图像</li>
                            <li>AI模型将分析图像</li>
                            <li>查看显示每种潜在发现概率分数的结果</li>
                            <li>0.5或更高（50%）的分数表示阳性发现</li>
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
                            <li>突出显示阳性发现的详细报告</li>
                        </ul>

                        <div style='background-color: #f3f4f6; padding: 16px; border-radius: 8px; margin-top: 24px;'>
                            <p style='margin: 0;'><strong>免责声明</strong>：此工具仅用于教育和研究目的。它不能替代专业医疗诊断。</p>
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
                            <h3>我应该上传什么类型的X光图像？</h3>
                            <p>该模型最适合标准PA（后前位）胸部X光片，格式为JPEG或PNG。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>这个模型有多准确？</h3>
                            <p>该模型在NIH ChestX-ray14数据集上训练，但像所有AI系统一样，它并不完美。不应将其用于临床诊断。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>概率分数意味着什么？</h3>
                            <p>0.5（50%）或更高的概率被认为是该疾病的阳性发现。较高的概率表示模型对检测结果更有信心。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>我可以将此模型用于医疗诊断吗？</h3>
                            <p>不可以。此工具仅用于研究和教育目的。请务必咨询合格的医疗专业人员获取医疗建议和诊断。</p>
                        </div>

                        <div style='border-left: 4px solid #3b82f6; padding-left: 16px; margin-bottom: 24px;'>
                            <h3>如果模型检测到潜在问题，我应该怎么做？</h3>
                            <p>如果您担心自己的健康状况，请咨询医疗专业人员。此工具不能替代专业医疗建议。</p>
                        </div>

                        <div style='background-color: #f3f4f6; padding: 16px; border-radius: 8px; margin-top: 24px;'>
                            <h3>使用提示</h3>
                            <ul>
                                <li>上传清晰、高质量的X光图像</li>
                                <li>确保图像未被裁剪或过度压缩</li>
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
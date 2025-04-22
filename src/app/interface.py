import os
import torch
import gradio as gr
import numpy as np
import logging
from pathlib import Path
from PIL import Image

from src.models import ChestXrayClassifier
from src.data import get_eval_transforms, download_chestxray14_dataset
from src.utils import get_available_device, create_bar_chart
from .disease_info import get_disease_info, format_html_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL = None
DEVICE = None
DISEASE_LABELS = None
TRANSFORM = None


def predict_image(model, image_path, disease_labels, device, transform=None):
    """
    预测图像中的疾病

    Args:
        model: 分类器模型
        image_path: 图像路径
        disease_labels: 疾病标签列表
        device: 设备
        transform: 图像变换

    Returns:
        predictions: 预测结果
        img: 图像对象
    """
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


def gradio_predict(image):
    """
    Gradio预测函数

    Args:
        image: 输入图像

    Returns:
        img: 原始图像
        fig: 条形图
        html_results: HTML格式的预测结果
    """
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


def load_model(model_path, num_classes=14, device=None):
    """
    加载模型

    Args:
        model_path: 模型路径
        num_classes: 类别数量
        device: 设备

    Returns:
        model: 加载的模型
        device: 设备
    """
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


def create_app(args):
    """
    创建Gradio应用

    Args:
        args: 命令行参数

    Returns:
        iface: Gradio界面
    """
    global MODEL, DEVICE, DISEASE_LABELS, TRANSFORM

    if args.device is None or args.device == 'auto':
        device = get_available_device()
    else:
        device = torch.device(args.device)

    logger.info(f"使用设备: {device}")

    # 获取疾病标签和数据集
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

    # 加载模型
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

    MODEL = model
    DEVICE = device
    DISEASE_LABELS = disease_labels
    TRANSFORM = transform

    # 创建疾病信息列表
    disease_info = []
    for disease in disease_labels:
        info = get_disease_info(disease)
        disease_info.append(f"- **{info['translation']} ({info['name']})**: {info['description']}")

    disease_md = "\n".join(disease_info)

    # 设置Gradio主题
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

    # 创建Gradio界面
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

    # 创建examples目录
    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    return iface
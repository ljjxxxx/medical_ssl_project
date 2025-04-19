import os
import torch
import argparse
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

from models import ChestXrayClassifier
from data_utils import get_eval_transforms, download_chestxray14_dataset


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
        backbone_name='resnet18',
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
    sorted_indices = np.argsort(probs)[::-1]

    sorted_diseases = [diseases[i] for i in sorted_indices]
    sorted_probs = [probs[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(sorted_diseases, sorted_probs, color='skyblue')

    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f'{sorted_probs[i]:.3f}',
            va='center'
        )

    ax.set_xlabel('Probability')
    ax.set_ylabel('Disease')
    ax.set_title('Chest X-ray disease prediction')

    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='阈值 (0.5)')
    ax.legend()

    plt.tight_layout()

    return fig


def get_chinese_disease_name(english_name):
    disease_dict = {
        'Atelectasis': '肺不张',
        'Consolidation': '肺实变',
        'Infiltration': '肺浸润',
        'Pneumothorax': '气胸',
        'Edema': '肺水肿',
        'Emphysema': '肺气肿',
        'Fibrosis': '肺纤维化',
        'Effusion': '胸腔积液',
        'Pneumonia': '肺炎',
        'Pleural_Thickening': '胸膜增厚',
        'Cardiomegaly': '心脏肥大',
        'Nodule': '肺结节',
        'Mass': '肺肿块',
        'Hernia': '疝气'
    }

    return disease_dict.get(english_name, english_name)


def gradio_predict(image):
    global MODEL, DEVICE, DISEASE_LABELS, TRANSFORM

    try:
        temp_path = "temp_upload.png"
        image.save(temp_path)

        predictions, img = predict_image(MODEL, temp_path, DISEASE_LABELS, DEVICE, TRANSFORM)

        fig = create_bar_chart(predictions)

        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        results_text = "疾病预测结果:\n\n"
        positive_findings = []

        for disease, prob in sorted_preds:
            status = "阳性" if prob >= 0.5 else "阴性"
            chinese_name = get_chinese_disease_name(disease)
            if prob >= 0.5:
                positive_findings.append(chinese_name)
            results_text += f"{chinese_name} ({disease}): {prob:.3f} ({status})\n"

        if positive_findings:
            results_text += f"\n阳性发现: {', '.join(positive_findings)}"
        else:
            results_text += "\n未检测到阳性发现。"

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return img, fig, results_text

    except Exception as e:
        error_msg = f"预测过程中出错: {str(e)}"
        logger.error(error_msg)
        return image, None, error_msg


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
            backbone_name='resnet18',
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

    description = """
    # 胸部X光疾病分类

    上传一张胸部X光图像来检测潜在的疾病。
    该模型在NIH ChestX-ray14数据集上训练，可以检测14种不同的疾病：

    - 肺不张 (Atelectasis) - 肺部塌陷或闭合
    - 肺实变 (Consolidation) - 肺组织充满液体
    - 肺浸润 (Infiltration) - 不应该存在的物质
    - 气胸 (Pneumothorax) - 胸腔内有空气导致肺部塌陷
    - 肺水肿 (Edema) - 肺组织中过多的液体
    - 肺气肿 (Emphysema) - 肺泡损伤
    - 肺纤维化 (Fibrosis) - 肺组织疤痕形成
    - 胸腔积液 (Effusion) - 肺部和胸腔之间的液体
    - 肺炎 (Pneumonia) - 肺部感染
    - 胸膜增厚 (Pleural Thickening) - 肺衬里增厚
    - 心脏肥大 (Cardiomegaly) - 心脏扩大
    - 肺结节 (Nodule) - 小圆形生长物
    - 肺肿块 (Mass) - 较大的生长物或肿瘤
    - 疝气 (Hernia) - 器官穿过腔壁

    模型预测每种疾病的概率。概率 >= 0.5 被视为阳性发现。
    """

    iface = gr.Interface(
        fn=gradio_predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(type="pil", label="原始图像"),
            gr.Plot(label="疾病概率"),
            gr.Textbox(label="结果", lines=10)
        ],
        title="胸部X光疾病分类器",
        description=description,
        examples=[
            "./examples/example1.png",
            "./examples/example2.png"
        ],
        allow_flagging="never"
    )

    examples_dir = Path("./examples")
    examples_dir.mkdir(exist_ok=True)

    iface.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
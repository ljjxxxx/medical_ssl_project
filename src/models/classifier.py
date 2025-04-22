import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .backbones import get_backbone


class MultiLabelClassificationHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_classes: int,
    ):
        """
        多标签分类头

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            num_classes: 类别数
        """
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class ChestXrayClassifier(nn.Module):
    def __init__(
            self,
            ssl_model: nn.Module = None,
            backbone_name: str = 'resnet50',
            pretrained: bool = True,
            num_classes: int = 14,
            freeze_backbone: bool = False
    ):
        """
        胸部X光疾病分类器

        Args:
            ssl_model: 预训练的SSL模型 (可选)
            backbone_name: 骨干网络名称
            pretrained: 是否使用预训练权重
            num_classes: 分类类别数量
            freeze_backbone: 是否冻结骨干网络
        """
        super().__init__()

        if ssl_model is not None:
            self.backbone = ssl_model.backbone

            try:
                feature_dim = ssl_model.projection_head.projection[0].in_features
            except (AttributeError, IndexError):
                with torch.no_grad():
                    dummy_input = torch.zeros(1, 3, 224, 224)
                    features = self.backbone(dummy_input)
                    feature_dim = features.shape[1]
        else:
            self.backbone, feature_dim = get_backbone(backbone_name, pretrained)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 创建分类头
        self.classifier = MultiLabelClassificationHead(
            input_dim=feature_dim,
            hidden_dim=512,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        logits = self.classifier(features)

        return {
            'features': features,
            'logits': logits
        }

    def predict_diseases(self, logits, disease_labels):
        """
        预测疾病概率

        Args:
            logits: 模型输出的logits
            disease_labels: 疾病标签列表

        Returns:
            predictions: 疾病预测概率字典
        """
        probs = torch.sigmoid(logits)

        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        predictions = {}
        for i, disease in enumerate(disease_labels):
            predictions[disease] = float(probs[i])

        return predictions
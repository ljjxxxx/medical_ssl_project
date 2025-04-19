import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class ProjectionHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


def get_backbone(name: str = 'resnet18', pretrained: bool = False):
    if name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
    elif name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
    else:
        raise ValueError(f"不支持的骨干网络: {name}")

    # 移除最终的分类层
    backbone.fc = nn.Identity()

    return backbone, feature_dim


class SimCLREncoder(nn.Module):
    def __init__(
            self,
            backbone_name: str = 'resnet18',
            pretrained: bool = False,
            projection_dim: int = 128,
            temperature: float = 0.1
    ):
        super().__init__()

        self.backbone, feature_dim = get_backbone(backbone_name, pretrained)

        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=2048,
            output_dim=projection_dim
        )

        self.temperature = temperature

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)

        projections = self.projection_head(features)

        return features, projections

    def contrastive_loss(self, projections_1, projections_2):
        n = projections_1.size(0)
        device = projections_1.device

        z1 = F.normalize(projections_1, dim=1)
        z2 = F.normalize(projections_2, dim=1)

        z_all = torch.cat([z1, z2], dim=0)

        similarity = torch.mm(z_all, z_all.t()) / self.temperature

        indices = torch.arange(0, n, device=device)
        mask = torch.zeros((2 * n, 2 * n), dtype=torch.bool, device=device)

        mask[indices, indices + n] = True
        mask[indices + n, indices] = True

        diag_mask = torch.eye(2 * n, dtype=torch.bool, device=device)

        exp_sim = torch.exp(similarity)

        exp_sim_no_diag = exp_sim.clone()
        exp_sim_no_diag[diag_mask] = 0

        pos_sim = similarity[mask]
        denominator = exp_sim_no_diag.sum(dim=1)

        loss = -torch.mean(pos_sim - torch.log(denominator))

        return loss


class MultiLabelClassificationHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_classes: int,
    ):
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
            backbone_name: str = 'resnet18',
            pretrained: bool = True,
            num_classes: int = 14,
            freeze_backbone: bool = False
    ):
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
        probs = torch.sigmoid(logits)

        if isinstance(probs, torch.Tensor):
            probs = probs.cpu().numpy()

        predictions = {}
        for i, disease in enumerate(disease_labels):
            predictions[disease] = float(probs[i])

        return predictions

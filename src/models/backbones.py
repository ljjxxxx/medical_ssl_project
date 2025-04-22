import torch
import torch.nn as nn
import torchvision.models as models


def get_backbone(name: str = 'resnet50', pretrained: bool = False):
    """
    获取骨干网络模型

    Args:
        name: 骨干网络名称 ('resnet18' 或 'resnet50')
        pretrained: 是否使用预训练权重

    Returns:
        backbone: 骨干网络模型
        feature_dim: 特征维度
    """
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
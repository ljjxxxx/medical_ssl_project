import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import get_backbone


class ProjectionHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int
    ):
        """
        投影头网络，将特征投影到对比学习空间

        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class SimCLREncoder(nn.Module):
    def __init__(
            self,
            backbone_name: str = 'resnet50',
            pretrained: bool = False,
            projection_dim: int = 128,
            temperature: float = 0.1
    ):
        """
        SimCLR 自监督学习编码器

        Args:
            backbone_name: 骨干网络名称
            pretrained: 是否使用预训练权重
            projection_dim: 投影维度
            temperature: 温度参数
        """
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
        """
        计算对比损失

        Args:
            projections_1: 第一组投影
            projections_2: 第二组投影

        Returns:
            loss: 对比损失
        """
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
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        hidden = channels // 2
        self.conv1 = ConvBNAct(channels, hidden, 1)
        self.conv2 = ConvBNAct(hidden, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class CSPBlock(nn.Module):
    """Small CSP/C2f-style fusion block used in the neck."""

    def __init__(self, channels: int, num_blocks: int = 2):
        super().__init__()
        hidden = channels // 2
        self.cv1 = ConvBNAct(channels, hidden * 2, 1)
        self.blocks = nn.ModuleList(ResidualBlock(hidden) for _ in range(num_blocks))
        self.cv2 = ConvBNAct(hidden * (2 + num_blocks), channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.cv1(x).chunk(2, dim=1)
        features = [left, right]
        current = right
        for block in self.blocks:
            current = block(current)
            features.append(current)
        return self.cv2(torch.cat(features, dim=1))


class PartialSpatialAttention(nn.Module):
    """C2PSA/A2-inspired lightweight partial spatial attention.

    It keeps most channels convolutional and applies a cheap spatial gate only to
    a channel slice, avoiding the memory cost of full self-attention on 640px
    feature maps.
    """

    def __init__(self, channels: int, ratio: float = 0.5):
        super().__init__()
        attn_channels = max(16, int(channels * ratio))
        attn_channels = min(attn_channels, channels)
        self.attn_channels = attn_channels
        self.pre = ConvBNAct(attn_channels, attn_channels, 1)
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(attn_channels, attn_channels, 7, padding=3, groups=attn_channels, bias=False),
            nn.BatchNorm2d(attn_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(attn_channels, attn_channels, 1),
            nn.Sigmoid(),
        )
        self.post = ConvBNAct(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, bypass = x[:, : self.attn_channels], x[:, self.attn_channels :]
        attended = self.pre(attended)
        attended = attended * self.spatial_gate(attended)
        return self.post(torch.cat([attended, bypass], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast, implemented from basic PyTorch layers."""

    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 5):
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = ConvBNAct(in_channels, hidden, 1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.cv2 = ConvBNAct(hidden * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        try:
            from torchvision.models import ResNet50_Weights, resnet50
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for the ResNet-50 ImageNet backbone. "
                "Install it with `pip install torchvision` or use Kaggle's PyTorch image."
            ) from exc

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class FPNPAN(nn.Module):
    def __init__(self, channels: int = 256, neck_variant: str = "baseline", use_attention: bool = False):
        super().__init__()
        if neck_variant not in {"baseline", "csp"}:
            raise ValueError("neck_variant must be 'baseline' or 'csp'.")
        self.neck_variant = neck_variant
        self.use_attention = use_attention
        self.lat3 = ConvBNAct(512, channels, 1)
        self.lat4 = ConvBNAct(1024, channels, 1)
        self.sppf = SPPF(2048, channels)

        self.fpn4 = self._fusion_block(channels * 2, channels)
        self.fpn3 = self._fusion_block(channels * 2, channels)

        self.down3 = ConvBNAct(channels, channels, 3, stride=2)
        self.pan4 = self._fusion_block(channels * 2, channels)
        self.down4 = ConvBNAct(channels, channels, 3, stride=2)
        self.pan5 = self._fusion_block(channels * 2, channels)
        self.attn3 = PartialSpatialAttention(channels) if use_attention else nn.Identity()
        self.attn4 = PartialSpatialAttention(channels) if use_attention else nn.Identity()
        self.attn5 = PartialSpatialAttention(channels) if use_attention else nn.Identity()

    def _fusion_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.neck_variant == "baseline":
            return nn.Sequential(ConvBNAct(in_channels, out_channels, 3), ResidualBlock(out_channels))
        return nn.Sequential(ConvBNAct(in_channels, out_channels, 3), CSPBlock(out_channels, num_blocks=2))

    def forward(self, features: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[torch.Tensor]:
        c3, c4, c5 = features
        p5 = self.sppf(c5)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3)

        p4 = self.fpn4(torch.cat([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")], dim=1))
        p3 = self.fpn3(torch.cat([p3, F.interpolate(p4, size=p3.shape[-2:], mode="nearest")], dim=1))

        n4 = self.pan4(torch.cat([self.down3(p3), p4], dim=1))
        n5 = self.pan5(torch.cat([self.down4(n4), p5], dim=1))
        return [self.attn3(p3), self.attn4(n4), self.attn5(n5)]


class DetectionHead(nn.Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        out_channels = 5 + num_classes
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(channels, channels, 3),
                    ConvBNAct(channels, channels, 3),
                    nn.Conv2d(channels, out_channels, 1),
                )
                for _ in range(3)
            ]
        )
        self._init_bias()

    def _init_bias(self) -> None:
        for head in self.heads:
            conv = head[-1]
            if isinstance(conv, nn.Conv2d) and conv.bias is not None:
                nn.init.constant_(conv.bias[0], -4.0)
                nn.init.constant_(conv.bias[1 : 5], 1.0)
                nn.init.constant_(conv.bias[5:], -2.0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [head(feature) for head, feature in zip(self.heads, features)]


class DecoupledDetectionHead(nn.Module):
    """YOLO-style decoupled regression/objectness and classification head."""

    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(channels, channels, 3),
                    ConvBNAct(channels, channels, 3),
                    nn.Conv2d(channels, 5, 1),
                )
                for _ in range(3)
            ]
        )
        cls_channels = max(96, channels // 2)
        self.cls_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(channels, cls_channels, 3),
                    ConvBNAct(cls_channels, cls_channels, 3),
                    nn.Conv2d(cls_channels, num_classes, 1),
                )
                for _ in range(3)
            ]
        )
        self._init_bias()

    def _init_bias(self) -> None:
        for reg_head, cls_head in zip(self.reg_heads, self.cls_heads):
            reg_conv = reg_head[-1]
            cls_conv = cls_head[-1]
            if isinstance(reg_conv, nn.Conv2d) and reg_conv.bias is not None:
                nn.init.constant_(reg_conv.bias[0], -4.0)
                nn.init.constant_(reg_conv.bias[1:5], 1.0)
            if isinstance(cls_conv, nn.Conv2d) and cls_conv.bias is not None:
                nn.init.constant_(cls_conv.bias, -2.0)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for feature, reg_head, cls_head in zip(features, self.reg_heads, self.cls_heads):
            reg_obj = reg_head(feature)
            cls_logits = cls_head(feature)
            outputs.append(torch.cat([reg_obj[:, :1], reg_obj[:, 1:5], cls_logits], dim=1))
        return outputs


class YoloResNet50(nn.Module):
    strides = (8, 16, 32)

    def __init__(
        self,
        num_classes: int = 5,
        pretrained_backbone: bool = True,
        neck_channels: int = 256,
        neck_variant: str = "baseline",
        head_variant: str = "coupled",
        use_attention: bool = False,
    ):
        super().__init__()
        if head_variant not in {"coupled", "decoupled"}:
            raise ValueError("head_variant must be 'coupled' or 'decoupled'.")
        self.num_classes = num_classes
        self.backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.neck = FPNPAN(neck_channels, neck_variant=neck_variant, use_attention=use_attention)
        if head_variant == "decoupled":
            self.head = DecoupledDetectionHead(neck_channels, num_classes)
        else:
            self.head = DetectionHead(neck_channels, num_classes)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.head(self.neck(self.backbone(x)))

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _logit(value: torch.Tensor | float) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).clamp(1e-4, 1.0 - 1e-4)
    return torch.log(tensor / (1.0 - tensor))


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
    channels_p2 = (256, 512, 1024, 2048)
    channels = (512, 1024, 2048)

    def __init__(self, pretrained: bool = True, return_p2: bool = False):
        super().__init__()
        self.return_p2 = return_p2
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        if self.return_p2:
            return c2, c3, c4, c5
        return c3, c4, c5


class TimmFeatureBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        return_p2: bool = False,
    ):
        super().__init__()
        self.return_p2 = return_p2
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for ConvNeXtV2-Tiny. Install it with `pip install timm` "
                "or add `timm` to requirements.txt before running on Kaggle."
            ) from exc

        out_indices = (0, 1, 2, 3) if return_p2 else (1, 2, 3)
        candidate_names = [model_name]
        if model_name == "convnextv2_tiny":
            candidate_names.extend(
                [
                    "convnextv2_tiny.fcmae_ft_in22k_in1k",
                    "convnextv2_tiny.fcmae_ft_in1k",
                ]
            )

        last_error = None
        for candidate in dict.fromkeys(candidate_names):
            try:
                self.model = timm.create_model(
                    candidate,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=out_indices,
                )
                self.model_name = candidate
                break
            except Exception as exc:  # pragma: no cover - depends on installed timm model registry.
                last_error = exc
        else:
            raise RuntimeError(f"Could not create timm backbone '{model_name}'. Last error: {last_error}") from last_error

        self.out_channels = tuple(int(ch) for ch in self.model.feature_info.channels())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.model(x))


class FPNPAN(nn.Module):
    def __init__(
        self,
        channels: int = 256,
        neck_variant: str = "baseline",
        use_attention: bool = False,
        use_p2: bool = False,
        in_channels: tuple[int, ...] = (512, 1024, 2048),
    ):
        super().__init__()
        if neck_variant not in {"baseline", "csp"}:
            raise ValueError("neck_variant must be 'baseline' or 'csp'.")
        self.neck_variant = neck_variant
        self.use_attention = use_attention
        self.use_p2 = use_p2
        expected_features = 4 if use_p2 else 3
        if len(in_channels) != expected_features:
            raise ValueError(f"FPNPAN expected {expected_features} input feature channels, got {len(in_channels)}.")
        if use_p2:
            c2_channels, c3_channels, c4_channels, c5_channels = in_channels
            self.lat2 = ConvBNAct(c2_channels, channels, 1)
        else:
            c3_channels, c4_channels, c5_channels = in_channels
        self.lat3 = ConvBNAct(c3_channels, channels, 1)
        self.lat4 = ConvBNAct(c4_channels, channels, 1)
        self.sppf = SPPF(c5_channels, channels)

        self.fpn4 = self._fusion_block(channels * 2, channels)
        self.fpn3 = self._fusion_block(channels * 2, channels)
        if use_p2:
            self.fpn2 = self._fusion_block(channels * 2, channels)

        if use_p2:
            self.down2 = ConvBNAct(channels, channels, 3, stride=2)
            self.pan3 = self._fusion_block(channels * 2, channels)
        self.down3 = ConvBNAct(channels, channels, 3, stride=2)
        self.pan4 = self._fusion_block(channels * 2, channels)
        self.down4 = ConvBNAct(channels, channels, 3, stride=2)
        self.pan5 = self._fusion_block(channels * 2, channels)
        self.attn2 = PartialSpatialAttention(channels) if use_attention and use_p2 else nn.Identity()
        self.attn3 = PartialSpatialAttention(channels) if use_attention else nn.Identity()
        self.attn4 = PartialSpatialAttention(channels) if use_attention else nn.Identity()
        self.attn5 = PartialSpatialAttention(channels) if use_attention else nn.Identity()

    def _fusion_block(self, in_channels: int, out_channels: int) -> nn.Module:
        if self.neck_variant == "baseline":
            return nn.Sequential(ConvBNAct(in_channels, out_channels, 3), ResidualBlock(out_channels))
        return nn.Sequential(ConvBNAct(in_channels, out_channels, 3), CSPBlock(out_channels, num_blocks=2))

    def forward(self, features: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        if self.use_p2:
            c2, c3, c4, c5 = features
        else:
            c3, c4, c5 = features
        p5 = self.sppf(c5)
        p4 = self.lat4(c4)
        p3 = self.lat3(c3)

        p4 = self.fpn4(torch.cat([p4, F.interpolate(p5, size=p4.shape[-2:], mode="nearest")], dim=1))
        p3 = self.fpn3(torch.cat([p3, F.interpolate(p4, size=p3.shape[-2:], mode="nearest")], dim=1))

        if self.use_p2:
            p2 = self.lat2(c2)
            p2 = self.fpn2(torch.cat([p2, F.interpolate(p3, size=p2.shape[-2:], mode="nearest")], dim=1))
            n3 = self.pan3(torch.cat([self.down2(p2), p3], dim=1))
            n4 = self.pan4(torch.cat([self.down3(n3), p4], dim=1))
            n5 = self.pan5(torch.cat([self.down4(n4), p5], dim=1))
            return [self.attn2(p2), self.attn3(n3), self.attn4(n4), self.attn5(n5)]

        n4 = self.pan4(torch.cat([self.down3(p3), p4], dim=1))
        n5 = self.pan5(torch.cat([self.down4(n4), p5], dim=1))
        return [self.attn3(p3), self.attn4(n4), self.attn5(n5)]


class DetectionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        class_priors: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        objectness_priors: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        num_scales: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_priors = None if class_priors is None else torch.as_tensor(class_priors, dtype=torch.float32)
        self.objectness_priors = None if objectness_priors is None else torch.as_tensor(objectness_priors, dtype=torch.float32)
        out_channels = 5 + num_classes
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(channels, channels, 3),
                    ConvBNAct(channels, channels, 3),
                    nn.Conv2d(channels, out_channels, 1),
                )
                for _ in range(num_scales)
            ]
        )
        self._init_bias()

    def _init_bias(self) -> None:
        class_bias = _logit(self.class_priors) if self.class_priors is not None else torch.full((self.num_classes,), -2.0)
        for scale_idx, head in enumerate(self.heads):
            conv = head[-1]
            if isinstance(conv, nn.Conv2d) and conv.bias is not None:
                obj_bias = -4.0
                if self.objectness_priors is not None:
                    obj_bias = float(_logit(self.objectness_priors[scale_idx]).item())
                nn.init.constant_(conv.bias[0], obj_bias)
                nn.init.constant_(conv.bias[1 : 5], 1.0)
                conv.bias.data[5:].copy_(class_bias.to(conv.bias.device))

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        return [head(feature) for head, feature in zip(self.heads, features)]


class DecoupledDetectionHead(nn.Module):
    """YOLO-style decoupled regression/objectness and classification head."""

    def __init__(
        self,
        channels: int,
        num_classes: int,
        class_priors: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        objectness_priors: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        num_scales: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_priors = None if class_priors is None else torch.as_tensor(class_priors, dtype=torch.float32)
        self.objectness_priors = None if objectness_priors is None else torch.as_tensor(objectness_priors, dtype=torch.float32)
        self.reg_heads = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(channels, channels, 3),
                    ConvBNAct(channels, channels, 3),
                    nn.Conv2d(channels, 5, 1),
                )
                for _ in range(num_scales)
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
                for _ in range(num_scales)
            ]
        )
        self._init_bias()

    def _init_bias(self) -> None:
        class_bias = _logit(self.class_priors) if self.class_priors is not None else torch.full((self.num_classes,), -2.0)
        for scale_idx, (reg_head, cls_head) in enumerate(zip(self.reg_heads, self.cls_heads)):
            reg_conv = reg_head[-1]
            cls_conv = cls_head[-1]
            if isinstance(reg_conv, nn.Conv2d) and reg_conv.bias is not None:
                obj_bias = -4.0
                if self.objectness_priors is not None:
                    obj_bias = float(_logit(self.objectness_priors[scale_idx]).item())
                nn.init.constant_(reg_conv.bias[0], obj_bias)
                nn.init.constant_(reg_conv.bias[1:5], 1.0)
            if isinstance(cls_conv, nn.Conv2d) and cls_conv.bias is not None:
                cls_conv.bias.data.copy_(class_bias.to(cls_conv.bias.device))

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
        backbone_name: str = "resnet50",
        neck_channels: int = 256,
        neck_variant: str = "baseline",
        head_variant: str = "coupled",
        use_attention: bool = False,
        use_p2: bool = False,
        class_priors: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        objectness_priors: list[float] | tuple[float, ...] | torch.Tensor | None = None,
    ):
        super().__init__()
        if head_variant not in {"coupled", "decoupled"}:
            raise ValueError("head_variant must be 'coupled' or 'decoupled'.")
        if backbone_name not in {"resnet50", "convnextv2_tiny"}:
            raise ValueError("backbone_name must be 'resnet50' or 'convnextv2_tiny'.")
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.use_p2 = use_p2
        self.strides = (4, 8, 16, 32) if use_p2 else (8, 16, 32)
        num_scales = len(self.strides)
        if backbone_name == "resnet50":
            self.backbone = ResNet50Backbone(pretrained=pretrained_backbone, return_p2=use_p2)
            backbone_channels = ResNet50Backbone.channels_p2 if use_p2 else ResNet50Backbone.channels
        else:
            self.backbone = TimmFeatureBackbone(backbone_name, pretrained=pretrained_backbone, return_p2=use_p2)
            backbone_channels = self.backbone.out_channels
        self.neck = FPNPAN(
            neck_channels,
            neck_variant=neck_variant,
            use_attention=use_attention,
            use_p2=use_p2,
            in_channels=backbone_channels,
        )
        if head_variant == "decoupled":
            self.head = DecoupledDetectionHead(
                neck_channels,
                num_classes,
                class_priors=class_priors,
                objectness_priors=objectness_priors,
                num_scales=num_scales,
            )
        else:
            self.head = DetectionHead(
                neck_channels,
                num_classes,
                class_priors=class_priors,
                objectness_priors=objectness_priors,
                num_scales=num_scales,
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.head(self.neck(self.backbone(x)))

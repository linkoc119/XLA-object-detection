from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .bbox import generalized_iou_loss


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 512,
        strides: tuple[int, int, int] = (8, 16, 32),
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 5.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight

    @staticmethod
    def _scale_index(box: torch.Tensor) -> int:
        width = box[2] - box[0]
        height = box[3] - box[1]
        size = torch.sqrt((width * height).clamp(min=1.0))
        if size < 64:
            return 0
        if size < 160:
            return 1
        return 2

    @staticmethod
    def _decode_boxes(raw_box: torch.Tensor, stride: int) -> torch.Tensor:
        b, _, h, w = raw_box.shape
        device = raw_box.device
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        centers_x = (xx + 0.5) * stride
        centers_y = (yy + 0.5) * stride
        distances = F.softplus(raw_box) * stride
        left, top, right, bottom = distances[:, 0], distances[:, 1], distances[:, 2], distances[:, 3]
        boxes = torch.stack(
            [
                centers_x.unsqueeze(0) - left,
                centers_y.unsqueeze(0) - top,
                centers_x.unsqueeze(0) + right,
                centers_y.unsqueeze(0) + bottom,
            ],
            dim=1,
        )
        return boxes.clamp(min=0)

    def forward(
        self,
        outputs: list[torch.Tensor],
        targets: list[torch.Tensor],
        labels: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        obj_losses = []
        cls_losses = []
        box_losses = []
        total_pos = 0

        for scale_idx, (output, stride) in enumerate(zip(outputs, self.strides)):
            bsz, _, feat_h, feat_w = output.shape
            device = output.device
            obj_target = torch.zeros((bsz, feat_h, feat_w), device=device)
            cls_target = torch.zeros((bsz, feat_h, feat_w, self.num_classes), device=device)
            box_target = torch.zeros((bsz, feat_h, feat_w, 4), device=device)
            pos_mask = torch.zeros((bsz, feat_h, feat_w), dtype=torch.bool, device=device)
            area_target = torch.full((bsz, feat_h, feat_w), float("inf"), device=device)

            for batch_idx, (boxes, classes) in enumerate(zip(targets, labels)):
                boxes = boxes.to(device)
                classes = classes.to(device)
                for box, class_idx in zip(boxes, classes):
                    if self._scale_index(box) != scale_idx:
                        continue
                    cx = ((box[0] + box[2]) * 0.5 / stride).floor().long().clamp(0, feat_w - 1)
                    cy = ((box[1] + box[3]) * 0.5 / stride).floor().long().clamp(0, feat_h - 1)
                    area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
                    if area >= area_target[batch_idx, cy, cx]:
                        continue
                    area_target[batch_idx, cy, cx] = area
                    obj_target[batch_idx, cy, cx] = 1.0
                    cls_target[batch_idx, cy, cx, class_idx] = 1.0
                    box_target[batch_idx, cy, cx] = box
                    pos_mask[batch_idx, cy, cx] = True

            obj_logits = output[:, 0]
            raw_box = output[:, 1:5]
            cls_logits = output[:, 5 : 5 + self.num_classes].permute(0, 2, 3, 1)

            obj_loss_map = F.binary_cross_entropy_with_logits(obj_logits, obj_target, reduction="none")
            weights = torch.where(obj_target > 0, torch.ones_like(obj_target), torch.full_like(obj_target, 0.25))
            obj_losses.append((obj_loss_map * weights).mean())

            if pos_mask.any():
                cls_losses.append(F.binary_cross_entropy_with_logits(cls_logits[pos_mask], cls_target[pos_mask]))
                pred_boxes = self._decode_boxes(raw_box, stride).permute(0, 2, 3, 1)[pos_mask]
                tgt_boxes = box_target[pos_mask]
                l1 = F.smooth_l1_loss(pred_boxes, tgt_boxes)
                giou = generalized_iou_loss(pred_boxes, tgt_boxes)
                box_losses.append(l1 / self.image_size + giou)
                total_pos += int(pos_mask.sum().item())

        obj_loss = torch.stack(obj_losses).sum() if obj_losses else outputs[0].sum() * 0
        cls_loss = torch.stack(cls_losses).mean() if cls_losses else obj_loss * 0
        box_loss = torch.stack(box_losses).mean() if box_losses else obj_loss * 0
        total = self.obj_weight * obj_loss + self.cls_weight * cls_loss + self.box_weight * box_loss
        metrics = {
            "loss": float(total.detach().cpu()),
            "obj_loss": float(obj_loss.detach().cpu()),
            "cls_loss": float(cls_loss.detach().cpu()),
            "box_loss": float(box_loss.detach().cpu()),
            "num_pos": float(total_pos),
        }
        return total, metrics

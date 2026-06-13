from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .bbox import complete_iou_loss, distance_iou_loss, generalized_iou_loss


class DetectionLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        image_size: int = 512,
        strides: tuple[int, ...] = (8, 16, 32),
        obj_weight: float = 1.0,
        cls_weight: float = 1.0,
        box_weight: float = 5.0,
        assign_radius: int = 1,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        class_weights: list[float] | tuple[float, ...] | None = None,
        box_loss: str = "giou",
    ):
        super().__init__()
        if box_loss not in {"giou", "diou", "ciou"}:
            raise ValueError("box_loss must be 'giou', 'diou', or 'ciou'.")
        self.num_classes = num_classes
        self.image_size = image_size
        self.strides = strides
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.assign_radius = assign_radius
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.box_loss = box_loss
        weights = torch.ones(num_classes, dtype=torch.float32) if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        if weights.numel() != num_classes:
            raise ValueError(f"class_weights must contain {num_classes} values.")
        self.register_buffer("class_weights", weights)

    def _scale_index(self, box: torch.Tensor) -> int:
        width = box[2] - box[0]
        height = box[3] - box[1]
        size = torch.sqrt((width * height).clamp(min=1.0))
        scale = self.image_size / 512.0
        if len(self.strides) == 4:
            if size < 32 * scale:
                return 0
            if size < 80 * scale:
                return 1
            if size < 160 * scale:
                return 2
            return 3
        if size < 64 * scale:
            return 0
        if size < 160 * scale:
            return 1
        return 2

    def _focal_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_t = self.focal_alpha * targets + (1.0 - self.focal_alpha) * (1.0 - targets)
        return alpha_t * (1.0 - pt).pow(self.focal_gamma) * bce

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
                    cx_float = (box[0] + box[2]) * 0.5 / stride
                    cy_float = (box[1] + box[3]) * 0.5 / stride
                    cx = cx_float.floor().long().clamp(0, feat_w - 1)
                    cy = cy_float.floor().long().clamp(0, feat_h - 1)
                    cx_value = float(cx_float.item())
                    cy_value = float(cy_float.item())
                    x1, y1, x2, y2 = [float(value.item()) for value in box]
                    area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
                    for offset_y in range(-self.assign_radius, self.assign_radius + 1):
                        for offset_x in range(-self.assign_radius, self.assign_radius + 1):
                            gx = int(cx.item()) + offset_x
                            gy = int(cy.item()) + offset_y
                            if gx < 0 or gx >= feat_w or gy < 0 or gy >= feat_h:
                                continue
                            cell_x = (gx + 0.5) * stride
                            cell_y = (gy + 0.5) * stride
                            in_box = x1 <= cell_x <= x2 and y1 <= cell_y <= y2
                            near_center = abs(gx + 0.5 - cx_value) <= self.assign_radius + 0.5 and abs(gy + 0.5 - cy_value) <= self.assign_radius + 0.5
                            if not in_box and not near_center:
                                continue
                            if area >= area_target[batch_idx, gy, gx]:
                                continue
                            area_target[batch_idx, gy, gx] = area
                            obj_target[batch_idx, gy, gx] = 1.0
                            cls_target[batch_idx, gy, gx].zero_()
                            cls_target[batch_idx, gy, gx, class_idx] = 1.0
                            box_target[batch_idx, gy, gx] = box
                            pos_mask[batch_idx, gy, gx] = True

            obj_logits = output[:, 0]
            raw_box = output[:, 1:5]
            cls_logits = output[:, 5 : 5 + self.num_classes].permute(0, 2, 3, 1)

            obj_loss_map = self._focal_bce(obj_logits, obj_target)
            obj_losses.append(obj_loss_map.mean())

            if pos_mask.any():
                cls_loss_map = self._focal_bce(cls_logits[pos_mask], cls_target[pos_mask]).sum(dim=1)
                target_classes = cls_target[pos_mask].argmax(dim=1)
                cls_losses.append((cls_loss_map * self.class_weights[target_classes]).mean())
                pred_boxes = self._decode_boxes(raw_box, stride).permute(0, 2, 3, 1)[pos_mask]
                tgt_boxes = box_target[pos_mask]
                l1 = F.smooth_l1_loss(pred_boxes, tgt_boxes)
                if self.box_loss == "ciou":
                    iou_loss = complete_iou_loss(pred_boxes, tgt_boxes)
                elif self.box_loss == "diou":
                    iou_loss = distance_iou_loss(pred_boxes, tgt_boxes)
                else:
                    iou_loss = generalized_iou_loss(pred_boxes, tgt_boxes)
                box_losses.append(l1 / self.image_size + iou_loss)
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

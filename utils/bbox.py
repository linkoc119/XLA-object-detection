from __future__ import annotations

import torch


def clip_boxes(boxes: torch.Tensor, width: int | float, height: int | float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes
    boxes = boxes.clone()
    boxes[:, 0::2].clamp_(0, float(width))
    boxes[:, 1::2].clamp_(0, float(height))
    return boxes


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def generalized_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.sum()

    inter_lt = torch.maximum(pred[:, :2], target[:, :2])
    inter_rb = torch.minimum(pred[:, 2:], target[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    pred_area = (pred[:, 2] - pred[:, 0]).clamp(min=0) * (pred[:, 3] - pred[:, 1]).clamp(min=0)
    target_area = (target[:, 2] - target[:, 0]).clamp(min=0) * (target[:, 3] - target[:, 1]).clamp(min=0)
    union = pred_area + target_area - inter
    iou = inter / union.clamp(min=1e-6)

    enc_lt = torch.minimum(pred[:, :2], target[:, :2])
    enc_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    enc_wh = (enc_rb - enc_lt).clamp(min=0)
    enc_area = enc_wh[:, 0] * enc_wh[:, 1]
    giou = iou - (enc_area - union) / enc_area.clamp(min=1e-6)
    return (1.0 - giou).mean()


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[current].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)

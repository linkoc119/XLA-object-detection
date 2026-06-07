from __future__ import annotations

import torch
import torch.nn.functional as F

from .bbox import clip_boxes, nms
from .dataset import IDX_TO_CLASS


@torch.no_grad()
def decode_outputs(
    outputs: list[torch.Tensor],
    image_size: int,
    orig_sizes: list[tuple[int, int]],
    scales: list[float],
    pad_xs: list[float],
    pad_ys: list[float],
    conf_threshold: float = 0.25,
    nms_iou: float = 0.5,
    max_detections: int = 100,
    strides: tuple[int, int, int] = (8, 16, 32),
) -> list[list[dict[str, object]]]:
    batch_size = outputs[0].shape[0]
    all_results: list[list[dict[str, object]]] = [[] for _ in range(batch_size)]

    for batch_idx in range(batch_size):
        image_boxes = []
        image_scores = []
        image_classes = []
        for output, stride in zip(outputs, strides):
            pred = output[batch_idx]
            _, feat_h, feat_w = pred.shape
            obj = torch.sigmoid(pred[0])
            raw_box = pred[1:5]
            cls_prob = torch.sigmoid(pred[5:])
            scores, classes = (obj.unsqueeze(0) * cls_prob).max(dim=0)
            keep = scores > conf_threshold
            if not keep.any():
                continue

            yy, xx = torch.meshgrid(
                torch.arange(feat_h, device=pred.device, dtype=torch.float32),
                torch.arange(feat_w, device=pred.device, dtype=torch.float32),
                indexing="ij",
            )
            centers_x = (xx + 0.5) * stride
            centers_y = (yy + 0.5) * stride
            distances = F.softplus(raw_box) * stride
            boxes = torch.stack(
                [
                    centers_x - distances[0],
                    centers_y - distances[1],
                    centers_x + distances[2],
                    centers_y + distances[3],
                ],
                dim=-1,
            )
            image_boxes.append(boxes[keep])
            image_scores.append(scores[keep])
            image_classes.append(classes[keep])

        if not image_boxes:
            continue

        boxes = torch.cat(image_boxes, dim=0)
        scores = torch.cat(image_scores, dim=0)
        classes = torch.cat(image_classes, dim=0)

        final_indices = []
        for class_idx in range(len(IDX_TO_CLASS)):
            class_mask = classes == class_idx
            if not class_mask.any():
                continue
            class_indices = class_mask.nonzero(as_tuple=False).flatten()
            kept = nms(boxes[class_indices], scores[class_indices], nms_iou)
            final_indices.append(class_indices[kept])

        if not final_indices:
            continue
        final_indices = torch.cat(final_indices)
        final_indices = final_indices[scores[final_indices].argsort(descending=True)[:max_detections]]

        orig_w, orig_h = orig_sizes[batch_idx]
        scale = scales[batch_idx]
        pad_x = pad_xs[batch_idx]
        pad_y = pad_ys[batch_idx]
        final_boxes = boxes[final_indices].clone()
        final_boxes[:, [0, 2]] = (final_boxes[:, [0, 2]] - pad_x) / scale
        final_boxes[:, [1, 3]] = (final_boxes[:, [1, 3]] - pad_y) / scale
        final_boxes = clip_boxes(final_boxes, orig_w, orig_h)
        final_scores = scores[final_indices]
        final_classes = classes[final_indices]

        for box, score, class_idx in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = box.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            all_results[batch_idx].append(
                {
                    "class": IDX_TO_CLASS[int(class_idx.item())],
                    "confidence": round(float(score.item()), 6),
                    "bbox": [round(float(x1), 3), round(float(y1), 3), round(float(x2), 3), round(float(y2), 3)],
                }
            )

    return all_results

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def evaluate_map(ground_truth_path: str | Path, predictions: list[dict[str, Any]], iou_threshold: float = 0.5) -> dict[str, Any]:
    gt = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    classes = gt["classes"]
    gt_by_class: dict[str, dict[str, list[dict[str, Any]]]] = {name: defaultdict(list) for name in classes}
    for ann in gt["annotations"]:
        gt_by_class[ann["class"]][ann["image_id"]].append({"bbox": [float(v) for v in ann["bbox"]], "matched": False})

    pred_by_class: dict[str, list[dict[str, Any]]] = {name: [] for name in classes}
    for entry in predictions:
        for box in entry["boxes"]:
            pred_by_class[box["class"]].append(
                {"image_id": entry["image_id"], "confidence": float(box["confidence"]), "bbox": [float(v) for v in box["bbox"]]}
            )

    per_class = {}
    aps = []
    total_tp = 0
    total_fp = 0
    total_gt = 0
    for class_name in classes:
        class_gt = gt_by_class[class_name]
        num_gt = sum(len(items) for items in class_gt.values())
        preds = sorted(pred_by_class[class_name], key=lambda item: item["confidence"], reverse=True)
        tp = []
        fp = []
        for pred in preds:
            candidates = class_gt.get(pred["image_id"], [])
            best_iou = 0.0
            best_idx = -1
            for idx, target in enumerate(candidates):
                if target["matched"]:
                    continue
                iou = bbox_iou(pred["bbox"], target["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou >= iou_threshold:
                candidates[best_idx]["matched"] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        cum_tp = []
        cum_fp = []
        tps = 0
        fps = 0
        for t, f in zip(tp, fp):
            tps += t
            fps += f
            cum_tp.append(tps)
            cum_fp.append(fps)
        recalls = [v / num_gt if num_gt else 0.0 for v in cum_tp]
        precisions = [t / max(t + f, 1) for t, f in zip(cum_tp, cum_fp)]
        ap = compute_ap(recalls, precisions) if num_gt else 0.0
        if num_gt:
            aps.append(ap)
        total_tp += tps
        total_fp += fps
        total_gt += num_gt
        per_class[class_name] = {
            "ap": round(ap, 6),
            "num_ground_truth": num_gt,
            "num_predictions": len(preds),
            "true_positives": tps,
            "false_positives": fps,
            "recall": round(tps / num_gt, 6) if num_gt else 0.0,
            "precision": round(tps / max(tps + fps, 1), 6),
        }

    map_50 = sum(aps) / len(aps) if aps else 0.0
    return {
        "mAP@0.5": round(map_50, 6),
        "num_ground_truth_boxes": total_gt,
        "num_predictions": sum(len(entry["boxes"]) for entry in predictions),
        "micro_precision": round(total_tp / max(total_tp + total_fp, 1), 6),
        "micro_recall": round(total_tp / total_gt, 6) if total_gt else 0.0,
        "per_class": per_class,
    }

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import YoloResNet50
from utils.dataset import CLASS_TO_IDX, InferenceImageDataset, collate_fn
from utils.inference import decode_outputs
from utils.bbox import nms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference and export predictions.json.")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="./models/best.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--conf_threshold", type=float, default=None)
    parser.add_argument("--nms_iou", type=float, default=None)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--checkpoint_key", choices=["auto", "model", "ema_model"], default="auto")
    parser.add_argument("--tta_flip", action="store_true")
    return parser.parse_args()


def merge_predictions(
    predictions_a: list[list[dict[str, object]]],
    predictions_b: list[list[dict[str, object]]],
    nms_iou: float,
    max_detections: int,
) -> list[list[dict[str, object]]]:
    class_to_idx = CLASS_TO_IDX
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    merged: list[list[dict[str, object]]] = []
    for dets_a, dets_b in zip(predictions_a, predictions_b):
        dets = dets_a + dets_b
        if not dets:
            merged.append([])
            continue

        boxes = torch.tensor([det["bbox"] for det in dets], dtype=torch.float32)
        scores = torch.tensor([float(det["confidence"]) for det in dets], dtype=torch.float32)
        classes = torch.tensor([class_to_idx[str(det["class"])] for det in dets], dtype=torch.long)
        keep_all = []
        for class_idx in range(len(class_to_idx)):
            class_mask = classes == class_idx
            if not class_mask.any():
                continue
            class_indices = class_mask.nonzero(as_tuple=False).flatten()
            keep_all.append(class_indices[nms(boxes[class_indices], scores[class_indices], nms_iou)])
        if not keep_all:
            merged.append([])
            continue

        keep = torch.cat(keep_all)
        keep = keep[scores[keep].argsort(descending=True)[:max_detections]]
        image_dets = []
        for idx in keep:
            box = boxes[idx].tolist()
            image_dets.append(
                {
                    "class": idx_to_class[int(classes[idx].item())],
                    "confidence": round(float(scores[idx].item()), 6),
                    "bbox": [round(float(value), 3) for value in box],
                }
            )
        merged.append(image_dets)
    return merged


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    image_size = args.image_size or int(checkpoint.get("image_size", 512))
    conf_threshold = args.conf_threshold if args.conf_threshold is not None else float(checkpoint.get("conf_threshold", 0.25))
    nms_iou = args.nms_iou if args.nms_iou is not None else float(checkpoint.get("nms_iou", 0.5))

    model = YoloResNet50(
        num_classes=len(CLASS_TO_IDX),
        pretrained_backbone=False,
        backbone_name=checkpoint.get("backbone_name", "resnet50"),
        neck_variant=checkpoint.get("neck_variant", "baseline"),
        head_variant=checkpoint.get("head_variant", "coupled"),
        use_attention=bool(checkpoint.get("use_attention", False)),
        use_p2=bool(checkpoint.get("use_p2", False)),
    ).to(device)
    checkpoint_key = args.checkpoint_key
    if checkpoint_key == "auto":
        checkpoint_key = "ema_model" if "ema_model" in checkpoint else "model"
    if checkpoint_key not in checkpoint:
        raise KeyError(f"Checkpoint key '{checkpoint_key}' not found in {checkpoint_path}.")
    model.load_state_dict(checkpoint[checkpoint_key])
    model.eval()

    dataset = InferenceImageDataset(args.image_dir, image_size=image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predict"):
            images = batch["images"].to(device, non_blocking=True)
            outputs = model(images)
            decoded = decode_outputs(
                outputs,
                image_size=image_size,
                orig_sizes=batch["orig_sizes"],
                scales=batch["scales"],
                pad_xs=batch["pad_xs"],
                pad_ys=batch["pad_ys"],
                conf_threshold=conf_threshold,
                nms_iou=nms_iou,
                max_detections=args.max_detections,
                strides=model.strides,
            )
            if args.tta_flip:
                flip_outputs = model(torch.flip(images, dims=[3]))
                decoded_flip = decode_outputs(
                    flip_outputs,
                    image_size=image_size,
                    orig_sizes=batch["orig_sizes"],
                    scales=batch["scales"],
                    pad_xs=batch["pad_xs"],
                    pad_ys=batch["pad_ys"],
                    conf_threshold=conf_threshold,
                    nms_iou=nms_iou,
                    max_detections=args.max_detections,
                    strides=model.strides,
                    flip_horizontal=True,
                )
                decoded = merge_predictions(decoded, decoded_flip, nms_iou=nms_iou, max_detections=args.max_detections)
            for image_id, boxes in zip(batch["image_ids"], decoded):
                predictions.append({"image_id": image_id, "boxes": boxes})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path("") else None
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} image predictions to {output_path}")


if __name__ == "__main__":
    main()

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
    return parser.parse_args()


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

    model = YoloResNet50(num_classes=len(CLASS_TO_IDX), pretrained_backbone=False).to(device)
    model.load_state_dict(checkpoint["model"])
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
            )
            for image_id, boxes in zip(batch["image_ids"], decoded):
                predictions.append({"image_id": image_id, "boxes": boxes})

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True) if output_path.parent != Path("") else None
    output_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} image predictions to {output_path}")


if __name__ == "__main__":
    main()

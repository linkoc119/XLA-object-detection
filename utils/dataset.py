from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset


CLASS_TO_IDX = {"person": 0, "car": 1, "dog": 2, "cat": 3, "chair": 4}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _resolve_image_path(image_dir: Path, image: dict[str, Any], annotation_path: Path) -> Path:
    direct = image_dir / image["id"]
    if direct.exists():
        return direct
    by_file_name = annotation_path.parent.parent / image["file_name"]
    if by_file_name.exists():
        return by_file_name
    return direct


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _letterbox(image: Image.Image, boxes: torch.Tensor, image_size: int) -> tuple[Image.Image, torch.Tensor, dict[str, float]]:
    orig_w, orig_h = image.size
    scale = min(image_size / orig_w, image_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_x = (image_size - new_w) // 2
    pad_y = (image_size - new_h) // 2

    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
    canvas.paste(resized, (pad_x, pad_y))

    if boxes.numel() > 0:
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_y

    meta = {"scale": scale, "pad_x": float(pad_x), "pad_y": float(pad_y)}
    return canvas, boxes, meta


def _horizontal_flip(image: Image.Image, boxes: torch.Tensor) -> tuple[Image.Image, torch.Tensor]:
    width = image.size[0]
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if boxes.numel() > 0:
        boxes = boxes.clone()
        old_x1 = boxes[:, 0].clone()
        old_x2 = boxes[:, 2].clone()
        boxes[:, 0] = width - old_x2
        boxes[:, 2] = width - old_x1
    return image, boxes


def _color_jitter(image: Image.Image) -> Image.Image:
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        image = enhancer(image).enhance(random.uniform(0.85, 1.15))
    return image


def _random_scale_crop(
    image: Image.Image, boxes: torch.Tensor, image_size: int
) -> tuple[Image.Image, torch.Tensor, torch.Tensor | None]:
    scale = random.uniform(0.85, 1.15)
    scaled_size = max(1, int(round(image_size * scale)))
    scaled = image.resize((scaled_size, scaled_size), Image.BILINEAR)
    if boxes.numel() > 0:
        boxes = boxes.clone() * scale

    if scaled_size >= image_size:
        max_offset = scaled_size - image_size
        off_x = random.randint(0, max_offset)
        off_y = random.randint(0, max_offset)
        image = scaled.crop((off_x, off_y, off_x + image_size, off_y + image_size))
        if boxes.numel() > 0:
            boxes[:, [0, 2]] -= off_x
            boxes[:, [1, 3]] -= off_y
    else:
        off_x = random.randint(0, image_size - scaled_size)
        off_y = random.randint(0, image_size - scaled_size)
        canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
        canvas.paste(scaled, (off_x, off_y))
        image = canvas
        if boxes.numel() > 0:
            boxes[:, [0, 2]] += off_x
            boxes[:, [1, 3]] += off_y

    if boxes.numel() > 0:
        boxes[:, 0::2].clamp_(0, image_size)
        boxes[:, 1::2].clamp_(0, image_size)
        keep = ((boxes[:, 2] - boxes[:, 0]) > 2) & ((boxes[:, 3] - boxes[:, 1]) > 2)
        boxes = boxes[keep]
        return image, boxes, keep
    return image, boxes, None


class DetectionDataset(Dataset):
    def __init__(self, annotation_path: str | Path, image_dir: str | Path, image_size: int = 512, train: bool = False):
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.train = train

        data = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.images = data["images"]
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ann in data["annotations"]:
            grouped[ann["image_id"]].append(ann)
        self.annotations = grouped

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_info = self.images[index]
        image_path = _resolve_image_path(self.image_dir, image_info, self.annotation_path)
        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

        anns = self.annotations.get(image_info["id"], [])
        boxes = torch.tensor([ann["bbox"] for ann in anns], dtype=torch.float32)
        labels = torch.tensor([CLASS_TO_IDX[ann["class"]] for ann in anns], dtype=torch.long)
        if boxes.numel() == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        image, boxes, meta = _letterbox(image, boxes, self.image_size)
        if self.train:
            if random.random() < 0.5:
                image, boxes = _horizontal_flip(image, boxes)
            if random.random() < 0.8:
                image = _color_jitter(image)
            if random.random() < 0.3:
                image, boxes, keep = _random_scale_crop(image, boxes, self.image_size)
                if keep is not None and labels.numel() > 0:
                    labels = labels[keep.cpu()]

        return {
            "image": _pil_to_tensor(image),
            "boxes": boxes,
            "labels": labels,
            "image_id": image_info["id"],
            "orig_size": (orig_w, orig_h),
            "scale": meta["scale"],
            "pad_x": meta["pad_x"],
            "pad_y": meta["pad_y"],
        }


class InferenceImageDataset(Dataset):
    def __init__(self, image_dir: str | Path, image_size: int = 512):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.paths = sorted(
            [p for p in self.image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        orig_w, orig_h = image.size
        image, _, meta = _letterbox(image, torch.zeros((0, 4), dtype=torch.float32), self.image_size)
        return {
            "image": _pil_to_tensor(image),
            "image_id": path.name,
            "orig_size": (orig_w, orig_h),
            "scale": meta["scale"],
            "pad_x": meta["pad_x"],
            "pad_y": meta["pad_y"],
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    return {
        "images": images,
        "boxes": [item.get("boxes", torch.zeros((0, 4), dtype=torch.float32)) for item in batch],
        "labels": [item.get("labels", torch.zeros((0,), dtype=torch.long)) for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "orig_sizes": [item["orig_size"] for item in batch],
        "scales": [item["scale"] for item in batch],
        "pad_xs": [item["pad_x"] for item in batch],
        "pad_ys": [item["pad_y"] for item in batch],
    }

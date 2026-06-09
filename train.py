from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models import YoloResNet50
from utils.dataset import CLASS_TO_IDX, DetectionDataset, collate_fn
from utils.inference import decode_outputs
from utils.loss import DetectionLoss
from utils.metrics import evaluate_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a from-scratch YOLO-style detector with ResNet-50 backbone.")
    parser.add_argument("--train_data", default="./annotations/train.json")
    parser.add_argument("--val_data", default="./annotations/val.json")
    parser.add_argument("--image_dir", default="./train/images")
    parser.add_argument("--val_image_dir", default="./val/images")
    parser.add_argument("--checkpoint_dir", default="./models")
    parser.add_argument("--experiment_log", default="./experiments.md")
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--conf_threshold", type=float, default=0.005)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--max_detections", type=int, default=100)
    parser.add_argument("--neck_variant", choices=["baseline", "csp"], default="csp")
    parser.add_argument("--head_variant", choices=["coupled", "decoupled"], default="decoupled")
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9998)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--assign_radius", type=int, default=1, help="Positive center assignment radius in feature cells. 1 means a 3x3 region.")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument(
        "--class_weights",
        default="1.0,1.3,1.4,1.2,1.8",
        help="Comma-separated weights for person,car,dog,cat,chair.",
    )
    parser.add_argument("--limit_train", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--limit_val", type=int, default=0, help="Optional smoke-test limit.")
    parser.add_argument("--no_pretrained_backbone", action="store_true")
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def parse_class_weights(value: str) -> list[float]:
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != len(CLASS_TO_IDX):
        raise ValueError(f"--class_weights must have {len(CLASS_TO_IDX)} comma-separated values.")
    return weights


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    batch = dict(batch)
    batch["images"] = batch["images"].to(device, non_blocking=True)
    batch["boxes"] = [boxes.to(device) for boxes in batch["boxes"]]
    batch["labels"] = [labels.to(device) for labels in batch["labels"]]
    return batch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        model_state = model.state_dict()
        for name, ema_value in self.ema.state_dict().items():
            model_value = model_state[name].detach()
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict[str, object]:
        return {"model": self.ema.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "model" in state:
            self.ema.load_state_dict(state["model"])
            self.updates = int(state.get("updates", 0))
            self.decay = float(state.get("decay", self.decay))
        else:
            self.ema.load_state_dict(state)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    val_data: str,
    image_size: int,
    conf_threshold: float,
    nms_iou: float,
    max_detections: int,
) -> tuple[list[dict], dict]:
    model.eval()
    predictions: list[dict] = []
    for batch in tqdm(dataloader, desc="Validate", leave=False):
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
            max_detections=max_detections,
        )
        for image_id, boxes in zip(batch["image_ids"], decoded):
            predictions.append({"image_id": image_id, "boxes": boxes})
    score = evaluate_map(val_data, predictions)
    return predictions, score


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_map: float,
    args: argparse.Namespace,
    class_weights: list[float],
    ema_model: torch.nn.Module | None = None,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_map": best_map,
        "classes": list(CLASS_TO_IDX.keys()),
        "image_size": args.image_size,
        "conf_threshold": args.conf_threshold,
        "nms_iou": args.nms_iou,
        "architecture": "YoloResNet50",
        "neck_variant": args.neck_variant,
        "head_variant": args.head_variant,
        "use_attention": args.use_attention,
        "use_ema": not args.no_ema,
        "ema_decay": args.ema_decay,
        "assign_radius": args.assign_radius,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "class_weights": class_weights,
    }
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.ema.state_dict() if isinstance(ema_model, ModelEMA) else ema_model.state_dict()
        if isinstance(ema_model, ModelEMA):
            checkpoint["ema"] = ema_model.state_dict()
    torch.save(
        checkpoint,
        path,
    )


def append_experiment_log(path: Path, args: argparse.Namespace, best_score: dict, best_epoch: int, checkpoint_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_name = args.experiment_name or f"resnet50-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    per_class = best_score.get("per_class", {})
    lines = [
        f"\n## {run_name}",
        f"- Time: {datetime.now().isoformat(timespec='seconds')}",
        f"- Device: {device_name}",
        "- Backbone: ResNet-50 ImageNet pretrained"
        + (" disabled" if args.no_pretrained_backbone else " enabled"),
        "- Architecture: ResNet-50 C3/C4/C5 + custom FPN/PAN + SPPF + anchor-free head",
        f"- Neck/head: neck_variant={args.neck_variant}, head_variant={args.head_variant}, use_attention={args.use_attention}",
        f"- EMA: enabled={not args.no_ema}, decay={args.ema_decay}",
        f"- Image size: {args.image_size}",
        f"- Batch size: {args.batch_size}",
        f"- Epochs: {args.epochs}",
        f"- Optimizer: AdamW lr={args.lr}, weight_decay={args.weight_decay}",
        f"- Assignment: center radius={args.assign_radius}",
        f"- Loss: focal_gamma={args.focal_gamma}, focal_alpha={args.focal_alpha}, class_weights={args.class_weights}",
        f"- Inference: conf_threshold={args.conf_threshold}, nms_iou={args.nms_iou}, max_detections={args.max_detections}",
        f"- Best epoch: {best_epoch}",
        f"- Checkpoint: {checkpoint_path.as_posix()}",
        f"- Val mAP@0.5: {best_score.get('mAP@0.5', 0.0)}",
        f"- Micro precision/recall: {best_score.get('micro_precision', 0.0)} / {best_score.get('micro_recall', 0.0)}",
        "- Per-class AP: "
        + ", ".join(f"{name}={stats.get('ap', 0.0)}" for name, stats in per_class.items()),
        "- Notes: fill in observed errors and next change after reviewing predictions.",
        "",
    ]
    with path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    class_weights = parse_class_weights(args.class_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location=device)
        args.image_size = int(resume_checkpoint.get("image_size", args.image_size))
        args.neck_variant = resume_checkpoint.get("neck_variant", "baseline")
        args.head_variant = resume_checkpoint.get("head_variant", "coupled")
        args.use_attention = bool(resume_checkpoint.get("use_attention", False))

    train_dataset = DetectionDataset(args.train_data, args.image_dir, image_size=args.image_size, train=True)
    val_dataset = DetectionDataset(args.val_data, args.val_image_dir, image_size=args.image_size, train=False)
    if args.limit_train > 0:
        train_dataset = Subset(train_dataset, range(min(args.limit_train, len(train_dataset))))
    if args.limit_val > 0:
        val_dataset = Subset(val_dataset, range(min(args.limit_val, len(val_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    model = YoloResNet50(
        num_classes=len(CLASS_TO_IDX),
        pretrained_backbone=not args.no_pretrained_backbone,
        neck_variant=args.neck_variant,
        head_variant=args.head_variant,
        use_attention=args.use_attention,
    ).to(device)
    criterion = DetectionLoss(
        num_classes=len(CLASS_TO_IDX),
        image_size=args.image_size,
        assign_radius=args.assign_radius,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        class_weights=class_weights,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())
    ema = None if args.no_ema else ModelEMA(model, decay=args.ema_decay)

    start_epoch = 1
    best_map = -1.0
    best_epoch = 0
    best_score: dict = {"mAP@0.5": 0.0}
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"

    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint.get("optimizer", optimizer.state_dict()))
        if ema is not None and "ema_model" in checkpoint:
            ema.load_state_dict(checkpoint.get("ema", checkpoint["ema_model"]))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_map = float(checkpoint.get("best_map", -1.0))

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "obj_loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "num_pos": 0.0}
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(batch["images"])
                loss, metrics = criterion(outputs, batch["boxes"], batch["labels"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            for key in running:
                running[key] += metrics[key]
            progress.set_postfix({key: f"{running[key] / step:.4f}" for key in ("loss", "obj_loss", "cls_loss", "box_loss")})
        scheduler.step()
        eval_model = ema.ema if ema is not None else model
        save_checkpoint(last_path, model, optimizer, epoch, best_map, args, class_weights, ema_model=ema)
        print(f"Saved latest checkpoint to {last_path}")

        if epoch % args.val_every == 0 or epoch == args.epochs:
            val_predictions, score = validate(
                eval_model,
                val_loader,
                device,
                args.val_data,
                image_size=args.image_size,
                conf_threshold=args.conf_threshold,
                nms_iou=args.nms_iou,
                max_detections=args.max_detections,
            )
            save_json(checkpoint_dir / "val_predictions.json", val_predictions)
            save_json(checkpoint_dir / "val_score.json", score)
            current_map = float(score["mAP@0.5"])
            print(f"Epoch {epoch}: val mAP@0.5={current_map:.6f}")
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch
                best_score = score
                save_checkpoint(best_path, model, optimizer, epoch, best_map, args, class_weights, ema_model=ema)
                print(f"Saved best checkpoint to {best_path}")

    append_experiment_log(Path(args.experiment_log), args, best_score, best_epoch, best_path)
    print(f"Best mAP@0.5={best_map:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()

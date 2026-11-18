#!/usr/bin/env python
"""Lightweight Sea-only fine-grained classifier.

Pipeline:
  1. Freeze SigLIP2 and extract full-image + bbox-crop features.
  2. Train a small projection/classification head with CE + SupCon.
  3. Select the checkpoint by validation macro-F1.

This deliberately does not train Qwen or generate CoT text.
"""

import argparse
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import AutoImageProcessor, AutoModel


CLASSES = [
    "Sea.ArleighBurke", "Sea.Buyan", "Sea.GeraldFord", "Sea.Gorshkov",
    "Sea.Independence", "Sea.Kilo", "Sea.Kirov", "Sea.Molniya",
    "Sea.Nanuchka", "Sea.OsaII", "Sea.Ticonderoga", "Sea.Type002.Shandong",
    "Sea.Type022.Houbei", "Sea.Type051B.Luhai", "Sea.Type053H3.JiangweiII",
    "Sea.Type054A.JiangkaiII", "Sea.Type055.Renhai", "Sea.TypeAsahi",
    "Sea.TypeChengKung", "Sea.TypeKangDing", "Sea.Wasp",
]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def materialize_data(data_dir):
    """Return train/validation parquet paths from a local dir or HF dataset ID."""
    root = Path(data_dir)
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(
            repo_id=data_dir,
            repo_type="dataset",
            local_dir=os.environ.get("SEA_DATA_LOCAL_DIR", "/kaggle/working/sea_data"),
        ))
    files = list(root.rglob("*.parquet"))
    train = next((p for p in files if "train" in p.name.lower()), None)
    val = next((p for p in files if "validation" in p.name.lower() or "val" in p.name.lower()), None)
    if train is None or val is None:
        raise FileNotFoundError(f"Could not find train/validation Parquet under {root}")
    return train, val


def decode_image(value):
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, str) and os.path.isfile(value):
        return Image.open(value).convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value)}")


def crop_bbox(image, bbox, padding=0.10):
    if bbox is None:
        return image
    if isinstance(bbox, dict):
        bbox = [bbox.get(k) for k in ("x", "y", "width", "height")]
    if len(bbox) != 4 or any(v is None for v in bbox):
        return image
    x, y, w, h = [float(v) for v in bbox]
    pw, ph = w * padding, h * padding
    left = max(0, int(round(x - pw)))
    top = max(0, int(round(y - ph)))
    right = min(image.width, int(round(x + w + pw)))
    bottom = min(image.height, int(round(y + h + ph)))
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def make_views(image, bbox=None, padding=0.10, image_is_pre_cropped=True,
               full_view_mode="same"):
    """Create the two views used by the classifier.

    The published Sea train/validation Parquet stores ``image`` *after* the
    original 640x640 image was cropped by bbox. Its bbox column still contains
    coordinates in the original image, so applying that bbox again is invalid.
    For that format, the safe representation is image + image (or image plus
    a resized center crop). Real-test images are full photographs and have no
    bbox, so they use the same image + image representation.
    """
    if image_is_pre_cropped or bbox is None:
        if full_view_mode == "center_crop":
            w, h = image.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            second = image.crop((left, top, left + side, top + side))
            return image, second
        return image, image
    return image, crop_bbox(image, bbox, padding)


class SeaParquetDataset(Dataset):
    def __init__(self, parquet_path, crop_padding=0.10, full_view_mode="same"):
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path, columns=["image", "bbox", "gt_class"])
        self.images = table["image"].to_pylist()
        self.bboxes = table["bbox"].to_pylist()
        labels = [str(x) for x in table["gt_class"].to_pylist()]
        unknown = sorted(set(labels) - set(CLASSES))
        if unknown:
            raise ValueError(f"Unknown Sea labels: {unknown}")
        self.labels = torch.tensor([LABEL_TO_ID[x] for x in labels], dtype=torch.long)
        self.crop_padding = crop_padding
        # The repository's Parquet is made by convert_and_push_hf_parquet.py
        # with crop_bbox=True: image bytes are already bbox crops.
        self.image_is_pre_cropped = True
        self.full_view_mode = full_view_mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = decode_image(self.images[index])
        full, second = make_views(
            image, self.bboxes[index], self.crop_padding,
            image_is_pre_cropped=self.image_is_pre_cropped,
            full_view_mode=self.full_view_mode,
        )
        return full, second, self.labels[index]


def collate_images(batch):
    full, crop, labels = zip(*batch)
    return list(full), list(crop), torch.stack(labels)


def make_backbone(model_id, device):
    processor = AutoImageProcessor.from_pretrained(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["dtype"] = dtype
    try:
        model = AutoModel.from_pretrained(model_id, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        model = AutoModel.from_pretrained(model_id, **kwargs)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def vision_features(model, pixel_values):
    vision = getattr(model, "vision_model", model)
    output = vision(pixel_values=pixel_values)
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = output.last_hidden_state
    return hidden.mean(dim=1)


@torch.inference_mode()
def extract_features(dataset, processor, model, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=min(2, os.cpu_count() or 1),
                        pin_memory=torch.cuda.is_available(), collate_fn=collate_images)
    chunks, labels = [], []
    for full, crop, y in loader:
        images = full + crop
        encoded = processor(images=images, return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        pixels = encoded["pixel_values"].to(device=device, dtype=model_dtype, non_blocking=True)
        feat = vision_features(model, pixels).float()
        b = len(y)
        feat = torch.cat([feat[:b], feat[b:]], dim=1)
        chunks.append(feat.cpu())
        labels.append(y)
    return torch.cat(chunks), torch.cat(labels)


class Head(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes):
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        z = self.project(x)
        return self.classifier(z), z


def supcon_loss(features, labels, temperature=0.07):
    z = F.normalize(features, dim=1)
    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    mask = labels[:, None].eq(labels[None, :]).to(logits.dtype)
    eye = torch.eye(len(labels), device=labels.device, dtype=logits.dtype)
    mask = mask - eye
    exp_logits = torch.exp(logits) * (1.0 - eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positives = mask.sum(dim=1)
    valid = positives > 0
    if not valid.any():
        return logits.sum() * 0.0
    mean_log_prob = (mask * log_prob).sum(dim=1) / positives.clamp_min(1)
    return -mean_log_prob[valid].mean()


def evaluate(head, features, labels, device):
    head.eval()
    with torch.no_grad():
        logits, _ = head(features.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
    y = labels.numpy()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(CLASSES)))).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="KhanhChien/Data-Qwen3.5-SFT-Sea-Only")
    parser.add_argument("--model_id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--output_dir", default="/kaggle/working/sea_metric_siglip2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--backbone_batch_size", type=int, default=8)
    parser.add_argument("--head_batch_size", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--supcon_weight", type=float, default=0.3)
    parser.add_argument("--crop_padding", type=float, default=0.10)
    parser.add_argument("--full_view_mode", choices=["same", "center_crop"], default="same",
                        help="Second view for already-cropped images; bbox is never reapplied.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_path, val_path = materialize_data(args.data)
    print(f"Train: {train_path}\nValidation: {val_path}")
    train_ds = SeaParquetDataset(train_path, args.crop_padding, args.full_view_mode)
    val_ds = SeaParquetDataset(val_path, args.crop_padding, args.full_view_mode)
    processor, backbone = make_backbone(args.model_id, device)

    train_features, train_labels = extract_features(train_ds, processor, backbone, device, args.backbone_batch_size)
    val_features, val_labels = extract_features(val_ds, processor, backbone, device, args.backbone_batch_size)
    torch.save({"features": train_features, "labels": train_labels}, Path(args.output_dir) / "train_features.pt")
    torch.save({"features": val_features, "labels": val_labels}, Path(args.output_dir) / "val_features.pt")
    print(f"Features: train={tuple(train_features.shape)}, val={tuple(val_features.shape)}")

    head = Head(train_features.shape[1], args.embedding_dim, len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=args.head_batch_size,
                        shuffle=True, drop_last=False)
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, z = head(x)
            loss = ce(logits, y) + args.supcon_weight * supcon_loss(z, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(y)
        scheduler.step()
        metrics = evaluate(head, val_features, val_labels, device)
        metrics.update(epoch=epoch, train_loss=total / len(train_ds))
        history.append(metrics)
        print(json.dumps({k: v for k, v in metrics.items() if k != "confusion_matrix"}))
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({"state_dict": head.state_dict(), "input_dim": train_features.shape[1],
                        "embedding_dim": args.embedding_dim, "classes": CLASSES,
                        "model_id": args.model_id, "crop_padding": args.crop_padding,
                        "val_metrics": metrics}, Path(args.output_dir) / "best_head.pt")
    (Path(args.output_dir) / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (Path(args.output_dir) / "classes.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    print(f"Best validation macro-F1: {best:.4f}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

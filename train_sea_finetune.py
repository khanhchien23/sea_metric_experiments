#!/usr/bin/env python
"""Fine-tune an image backbone for the Sea classes.

Unlike the frozen-feature baseline, this trains the vision encoder itself.
Each sample contains an original full image and a bbox crop. The full-image
loss is the primary objective because real-test images have no bbox; the crop
loss is an auxiliary objective that teaches fine-grained ship structure.
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
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from train_sea_metric import CLASSES, LABEL_TO_ID, decode_image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def materialize_data(data_dir):
    root = Path(data_dir)
    if not root.is_dir():
        from huggingface_hub import snapshot_download
        root = Path(snapshot_download(
            repo_id=data_dir,
            repo_type="dataset",
            local_dir=os.environ.get("SEA_FULL_DATA_LOCAL_DIR", "/kaggle/working/sea_full_data"),
        ))
    files = list(root.rglob("*.parquet"))
    train = next((p for p in files if "train" in p.name.lower()), None)
    validation = next((p for p in files if "validation" in p.name.lower() or "val" in p.name.lower()), None)
    if train is None or validation is None:
        raise FileNotFoundError(
            f"Need train/validation full-image Parquet under {root}. "
            "Expected columns: full_image, crop_image, gt_class."
        )
    return train, validation


class PairDataset(Dataset):
    def __init__(self, parquet_path, train=False):
        import pyarrow.parquet as pq

        names = set(pq.read_schema(parquet_path).names)
        required = {"full_image", "crop_image", "gt_class"}
        missing = required - names
        if missing:
            raise ValueError(
                f"{parquet_path} is missing {sorted(missing)}. Run pack_sea_full_parquet.py first."
            )
        table = pq.read_table(parquet_path, columns=["full_image", "crop_image", "gt_class"])
        self.full = table["full_image"].to_pylist()
        self.crop = table["crop_image"].to_pylist()
        labels = [str(value) for value in table["gt_class"].to_pylist()]
        unknown = sorted(set(labels) - set(CLASSES))
        if unknown:
            raise ValueError(f"Unknown Sea labels: {unknown}")
        self.labels = torch.tensor([LABEL_TO_ID[value] for value in labels], dtype=torch.long)
        self.train = train
        self.full_aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.75, 1.35), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.03),
            transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.12),
        ])
        self.crop_aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.70, 1.0), ratio=(0.80, 1.25), antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.16, contrast=0.16, saturation=0.10, hue=0.025),
        ])
        self.eval_transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        full = decode_image(self.full[index])
        crop = decode_image(self.crop[index])
        if self.train:
            full = self.full_aug(full)
            crop = self.crop_aug(crop)
        else:
            full = self.eval_transform(full)
            crop = self.eval_transform(crop)
        return full, crop, self.labels[index]


def collate(batch):
    full, crop, labels = zip(*batch)
    return list(full), list(crop), torch.stack(labels)


def load_backbone(model_id, kind, device):
    processor = AutoImageProcessor.from_pretrained(model_id)
    # AutoModel is used deliberately for compatibility with both SigLIP2 and
    # DINOv2. For SigLIP2 we keep only its vision tower in the returned model.
    load_kwargs = {"trust_remote_code": True}
    if device.type == "cuda":
        # Keep parameters in fp32 for stable AdamW updates; autocast below
        # keeps activations small enough for a Kaggle T4/P100.
        load_kwargs["torch_dtype"] = torch.float32
    model = AutoModel.from_pretrained(model_id, **load_kwargs)
    if kind == "siglip2" and hasattr(model, "vision_model"):
        model = model.vision_model
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return processor, model


def get_embedding(model, pixel_values):
    output = model(pixel_values=pixel_values)
    pooled = getattr(output, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = output.last_hidden_state
    # DINOv2's CLS token; use mean pooling as a safe fallback.
    return hidden[:, 0] if hidden.shape[1] > 1 else hidden.mean(dim=1)


def embedding_dim(model):
    for name in ("hidden_size", "projection_dim"):
        value = getattr(model.config, name, None)
        if value:
            return int(value)
    raise ValueError("Could not determine backbone embedding size")


def layer_modules(model):
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return []
    layers = getattr(encoder, "layers", None)
    if layers is None:
        layers = getattr(encoder, "layer", None)
    return list(layers) if layers is not None else []


def configure_trainable(model, unfreeze_layers):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if unfreeze_layers < 0:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return
    layers = layer_modules(model)
    for layer in layers[-unfreeze_layers:] if unfreeze_layers else []:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if any(key in name.lower() for key in ("layernorm", "layer_norm", "post_layernorm", "pooler")):
            parameter.requires_grad_(True)


class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.20)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, embedding):
        return self.fc(self.dropout(self.norm(embedding)))


def encode(processor, images, device):
    result = processor(images=images, return_tensors="pt")
    return result["pixel_values"].to(device)


@torch.no_grad()
def evaluate(backbone, head, loader, processor, device, amp_dtype):
    backbone.eval()
    head.eval()
    true, pred = [], []
    for full, _crop, labels in loader:
        pixels = encode(processor, full, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            logits = head(get_embedding(backbone, pixels))
        true.extend(labels.tolist())
        pred.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Full-image Parquet directory or HF dataset ID")
    parser.add_argument("--model_id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--kind", choices=["siglip2", "dinov2"], default="siglip2")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--unfreeze_layers", type=int, default=-1,
                        help="-1=all vision layers; 0=head only; positive=last N layers")
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16
    print(f"Device: {device}")
    train_path, val_path = materialize_data(args.data)
    print(f"Train: {train_path}\nValidation: {val_path}")
    train_ds = PairDataset(train_path, train=True)
    val_ds = PairDataset(val_path, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=min(2, os.cpu_count() or 1), pin_memory=device.type == "cuda",
                              collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=min(2, os.cpu_count() or 1), pin_memory=device.type == "cuda",
                            collate_fn=collate)
    processor, backbone = load_backbone(args.model_id, args.kind, device)
    configure_trainable(backbone, args.unfreeze_layers)
    head = Classifier(embedding_dim(backbone), len(CLASSES)).to(device)
    trainable_backbone = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone, "lr": args.backbone_lr},
        {"params": head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * ((len(train_loader) + args.grad_accum - 1) // args.grad_accum))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    best = -1.0
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        backbone.train()
        head.train()
        running = 0.0
        for step, (full, crop, labels) in enumerate(train_loader, 1):
            labels = labels.to(device, non_blocking=True)
            pixels = encode(processor, full + crop, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                embeddings = get_embedding(backbone, pixels)
                size = len(labels)
                full_logits = head(embeddings[:size])
                crop_logits = head(embeddings[size:])
                # Full view is weighted more because real-test has no bbox.
                loss = (0.65 * criterion(full_logits, labels)
                        + 0.20 * criterion(crop_logits, labels)
                        + 0.15 * criterion((full_logits + crop_logits) / 2.0, labels))
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            running += loss.item() * args.grad_accum * len(labels)
        metrics = evaluate(backbone, head, val_loader, processor, device, amp_dtype)
        metrics.update(epoch=epoch, train_loss=running / len(train_ds))
        history.append(metrics)
        print(json.dumps(metrics))
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            torch.save({
                "backbone": backbone.state_dict(),
                "head": head.state_dict(),
                "input_dim": embedding_dim(backbone),
                "classes": CLASSES,
                "model_id": args.model_id,
                "kind": args.kind,
                "unfreeze_layers": args.unfreeze_layers,
                "val_metrics": metrics,
            }, output_dir / "best_finetuned.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best validation macro-F1: {best:.4f}")
    print(f"Saved to: {output_dir / 'best_finetuned.pt'}")


if __name__ == "__main__":
    main()

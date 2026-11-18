#!/usr/bin/env python
"""Ensemble two fine-tuned vision classifiers.

The ensemble weight is selected on synthetic validation only. Real labels are
never read until the final evaluation report.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModel

from infer_sea_finetune import RealDataset, collate as real_collate
from train_sea_finetune import Classifier, PairDataset, collate as pair_collate, get_embedding
from train_sea_metric import CLASSES


def load_bundle(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    processor = AutoImageProcessor.from_pretrained(checkpoint["model_id"])
    model = AutoModel.from_pretrained(checkpoint["model_id"], trust_remote_code=True)
    if checkpoint.get("kind", "siglip2") == "siglip2" and hasattr(model, "vision_model"):
        model = model.vision_model
    model.load_state_dict(checkpoint["backbone"])
    model.to(device).eval()
    head = Classifier(checkpoint["input_dim"], len(checkpoint.get("classes", CLASSES)))
    head.load_state_dict(checkpoint["head"])
    head.to(device).eval()
    return checkpoint, processor, model, head


@torch.no_grad()
def logits_for_images(bundle, images, device):
    _checkpoint, processor, model, head = bundle
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        return head(get_embedding(model, pixels)).float()


@torch.no_grad()
def collect_validation(bundle_a, bundle_b, loader, device):
    logits_a, logits_b, labels = [], [], []
    for full, _crop, y in loader:
        logits_a.append(logits_for_images(bundle_a, full, device).cpu())
        logits_b.append(logits_for_images(bundle_b, full, device).cpu())
        labels.append(y)
    return torch.cat(logits_a), torch.cat(logits_b), torch.cat(labels)


def score(logits, labels):
    pred = logits.argmax(dim=1).numpy()
    y = labels.numpy()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


@torch.no_grad()
def evaluate_real(bundle_a, bundle_b, loader, weight, device):
    true, pred = [], []
    for _names, images, labels in loader:
        mixed = (weight * logits_for_images(bundle_a, images, device)
                 + (1.0 - weight) * logits_for_images(bundle_b, images, device))
        true.extend(labels.tolist())
        pred.extend(mixed.argmax(dim=1).cpu().tolist())
    y, p = np.array(true), np.array(pred)
    return {
        "count": int(len(y)),
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True, help="Synthetic full-image validation Parquet")
    parser.add_argument("--real", required=True, help="Real Parquet; evaluation only")
    parser.add_argument("--model_a", required=True, help="First fine-tuned checkpoint")
    parser.add_argument("--model_b", required=True, help="Second fine-tuned checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle_a = load_bundle(args.model_a, device)
    bundle_b = load_bundle(args.model_b, device)

    validation = PairDataset(args.validation, train=False)
    validation_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False,
                                    num_workers=2, collate_fn=pair_collate)
    val_a, val_b, val_y = collect_validation(bundle_a, bundle_b, validation_loader, device)
    best_weight, best_metrics = 0.5, None
    for weight in np.linspace(0.0, 1.0, 21):
        metrics = score(weight * val_a + (1.0 - weight) * val_b, val_y)
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_weight, best_metrics = float(weight), metrics
    print(json.dumps({"synthetic_validation": best_metrics, "weight_model_a": best_weight}, indent=2))

    real = RealDataset(args.real)
    real_loader = DataLoader(real, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=real_collate)
    real_metrics = evaluate_real(bundle_a, bundle_b, real_loader, best_weight, device)
    result = {
        "protocol": "ensemble weight selected on synthetic validation; real used only for final evaluation",
        "model_a": args.model_a,
        "model_b": args.model_b,
        "weight_model_a": best_weight,
        "weight_model_b": 1.0 - best_weight,
        "synthetic_validation": best_metrics,
        "real_test": real_metrics,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"real_test": real_metrics}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Inference/evaluation for the frozen-SigLIP2 Sea metric classifier."""

import argparse
import csv
import io
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModel

from train_sea_metric import CLASSES, Head, crop_bbox, decode_image, vision_features


class RealParquetDataset(Dataset):
    def __init__(self, parquet_path, crop_padding=0.10):
        import pyarrow.parquet as pq

        schema = set(pq.read_schema(parquet_path).names)
        columns = ["file_name", "gt_class"]
        if "image" in schema:
            columns.append("image")
        elif "image_bytes" in schema:
            columns.append("image_bytes")
        else:
            raise ValueError("Real Parquet needs image or image_bytes")
        if "bbox" in schema:
            columns.append("bbox")
        table = pq.read_table(parquet_path, columns=columns)
        self.names = table["file_name"].to_pylist() if "file_name" in columns else [str(i) for i in range(table.num_rows)]
        self.labels_text = [str(x) for x in table["gt_class"].to_pylist()]
        unknown = sorted(set(self.labels_text) - set(CLASSES))
        if unknown:
            raise ValueError(f"Unknown labels in real data: {unknown}")
        self.labels = torch.tensor([CLASSES.index(x) for x in self.labels_text], dtype=torch.long)
        image_column = "image" if "image" in columns else "image_bytes"
        self.images = table[image_column].to_pylist()
        self.bboxes = table["bbox"].to_pylist() if "bbox" in columns else [None] * table.num_rows
        self.crop_padding = crop_padding

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = decode_image(self.images[index])
        # selected_sea_only_real.parquet has no bbox; its image_bytes are
        # already the selected object image, so use the same image for both
        # branches. If bbox exists, use the same crop rule as training.
        crop = crop_bbox(image, self.bboxes[index], self.crop_padding)
        return self.names[index], image, crop, self.labels[index]


def collate(batch):
    names, full, crop, labels = zip(*batch)
    return list(names), list(full), list(crop), torch.stack(labels)


def load_backbone(model_id, device):
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
    return processor, model


@torch.inference_mode()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.head, map_location="cpu")
    classes = checkpoint.get("classes", CLASSES)
    model_id = checkpoint.get("model_id", args.model_id)
    processor, backbone = load_backbone(model_id, device)
    head = Head(
        checkpoint["input_dim"], checkpoint["embedding_dim"], len(classes)
    ).to(device)
    head.load_state_dict(checkpoint["state_dict"])
    head.eval()

    dataset = RealParquetDataset(args.data, checkpoint.get("crop_padding", args.crop_padding))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(2, os.cpu_count() or 1),
                        pin_memory=torch.cuda.is_available(), collate_fn=collate)
    rows = []
    all_pred, all_true = [], []
    for names, full, crop, labels in loader:
        encoded = processor(images=full + crop, return_tensors="pt")
        model_dtype = next(backbone.parameters()).dtype
        pixels = encoded["pixel_values"].to(device=device, dtype=model_dtype, non_blocking=True)
        features = vision_features(backbone, pixels).float()
        size = len(labels)
        features = torch.cat([features[:size], features[size:]], dim=1)
        logits, _ = head(features)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
        for name, true_id, pred_id, conf, probs in zip(
            names, labels.tolist(), prediction.tolist(), confidence.tolist(), probabilities.tolist()
        ):
            rows.append({
                "file_name": name,
                "ground_truth": classes[true_id],
                "prediction": classes[pred_id],
                "correct": int(true_id == pred_id),
                "confidence": float(conf),
                "top5": "|".join(classes[i] for i in np.argsort(probs)[::-1][:5]),
            })
        all_true.extend(labels.tolist())
        all_pred.extend(prediction.tolist())

    y_true, y_pred = np.array(all_true), np.array(all_pred)
    metrics = {
        "count": len(rows),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
        "classes": classes,
    }
    per_class = {}
    for i, cls in enumerate(classes):
        mask = y_true == i
        per_class[cls] = {
            "count": int(mask.sum()),
            "correct": int(((y_pred == i) & mask).sum()),
            "accuracy": float(((y_pred == i) & mask).sum() / mask.sum()) if mask.sum() else None,
        }
    metrics["per_class"] = per_class
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("count", "accuracy", "macro_f1", "balanced_accuracy")}, indent=2))
    print(f"Predictions: {output}")
    print(f"Metrics: {metrics_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model_id", default="google/siglip2-base-patch16-224")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--crop_padding", type=float, default=0.10)
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()

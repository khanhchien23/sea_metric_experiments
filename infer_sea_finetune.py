#!/usr/bin/env python
"""Evaluate a fine-tuned Sea image classifier on the real Parquet."""

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

from train_sea_finetune import Classifier, get_embedding
from train_sea_metric import CLASSES, decode_image


class RealDataset(Dataset):
    def __init__(self, path):
        import pyarrow.parquet as pq
        schema = set(pq.read_schema(path).names)
        image_column = "image_bytes" if "image_bytes" in schema else "image"
        table = pq.read_table(path, columns=[image_column, "gt_class", "file_name"])
        self.images = table[image_column].to_pylist()
        self.labels_text = [str(x) for x in table["gt_class"].to_pylist()]
        self.names = table["file_name"].to_pylist()
        self.labels = torch.tensor([CLASSES.index(x) for x in self.labels_text], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.names[index], decode_image(self.images[index]), self.labels[index]


def collate(batch):
    names, images, labels = zip(*batch)
    return list(names), list(images), torch.stack(labels)


def load_model(checkpoint, device):
    model_id = checkpoint["model_id"]
    kind = checkpoint.get("kind", "siglip2")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    if kind == "siglip2" and hasattr(model, "vision_model"):
        model = model.vision_model
    model.load_state_dict(checkpoint["backbone"])
    model.to(device).eval()
    head = Classifier(checkpoint["input_dim"], len(checkpoint.get("classes", CLASSES)))
    head.load_state_dict(checkpoint["head"])
    head.to(device).eval()
    return processor, model, head


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint.get("classes", CLASSES)
    processor, backbone, head = load_model(checkpoint, device)
    dataset = RealDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=min(2, os.cpu_count() or 1), collate_fn=collate,
                        pin_memory=device.type == "cuda")
    all_true, all_pred, rows = [], [], []
    for names, images, labels in loader:
        pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = head(get_embedding(backbone, pixels))
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)
        for name, y, p, conf, scores in zip(names, labels.tolist(), pred.tolist(), confidence.tolist(), probs.tolist()):
            rows.append({"file_name": name, "ground_truth": classes[y], "prediction": classes[p],
                         "correct": int(y == p), "confidence": float(conf),
                         "top5": "|".join(classes[i] for i in np.argsort(scores)[::-1][:5])})
        all_true.extend(labels.tolist())
        all_pred.extend(pred.tolist())
    y_true, y_pred = np.array(all_true), np.array(all_pred)
    metrics = {
        "count": len(rows), "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist(),
        "classes": classes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("count", "accuracy", "macro_f1", "balanced_accuracy")}, indent=2))


if __name__ == "__main__":
    main()

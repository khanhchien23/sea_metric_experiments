#!/usr/bin/env python
"""Frozen DINOv2 retrieval: synthetic TRAIN gallery, held-out image queries.

Each gallery image has one L2-normalized CLS embedding. Exact cosine search
is tiny for 2,781 images and needs neither FAISS nor a trainable classifier.
Query labels are optional and are used only after prediction for reporting.
"""

import argparse
import csv
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def resolve_parquet(data, filename):
    path = Path(data)
    if path.is_file():
        return path
    if path.is_dir():
        target = path / filename
        if target.is_file():
            return target
        raise FileNotFoundError(f"Expected {target}; pass the Parquet file directly if needed")
    if data.startswith(("/", "\\", ".")) or ":" in data or data.endswith(".parquet"):
        raise FileNotFoundError(f"Local data path does not exist: {data}")
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=data, repo_type="dataset", filename=filename))


def read_images(path, column=None, require_labels=False):
    import pyarrow.parquet as pq
    names = set(pq.read_schema(path).names)
    if column is None:
        column = next((c for c in ("image_bytes", "image", "crop_image", "full_image") if c in names), None)
    if column not in names:
        raise ValueError(f"Image column {column!r} not found in {path}; columns={sorted(names)}")
    if require_labels and "gt_class" not in names:
        raise ValueError("Gallery requires gt_class labels")
    columns = list(dict.fromkeys([column] + [c for c in ("id", "file_name", "gt_class") if c in names]))
    table = pq.read_table(path, columns=columns)
    if table.num_rows == 0:
        raise ValueError(f"Empty Parquet: {path}")
    records = table.to_pylist()
    images = [r[column] for r in records]
    identifiers = [str(r.get("id") or r.get("file_name") or i) for i, r in enumerate(records)]
    filenames = [str(r.get("file_name") or identifiers[i]) for i, r in enumerate(records)]
    labels = [r.get("gt_class") for r in records]
    if require_labels and any(not isinstance(y, str) or not y.startswith("Sea.") for y in labels):
        raise ValueError("Gallery must contain only labeled Sea images")
    return images, identifiers, filenames, labels, column


def decode(value):
    if isinstance(value, dict):
        value = value.get("bytes") or value.get("path")
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as source:
            return source.convert("RGB")
    if isinstance(value, str):
        with Image.open(value) as source:
            return source.convert("RGB")
    raise ValueError(f"Unsupported image type: {type(value)}")


def load_encoder(model_id, revision=None):
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision)
    if model.config.model_type != "dinov2":
        raise ValueError("This implementation expects a DINOv2 encoder, e.g. facebook/dinov2-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval().requires_grad_(False)
    print(f"Encoder: {model_id}; device: {device}; pretrained frozen; pooling: CLS")
    return processor, model, device


@torch.inference_mode()
def extract(images, processor, model, device, batch_size):
    from tqdm.auto import tqdm
    chunks = []
    for start in tqdm(range(0, len(images), batch_size), desc="Embedding images"):
        batch = [decode(value) for value in images[start:start + batch_size]]
        # Use precisely the same native processor for gallery and queries.
        pixels = processor(images=batch, return_tensors="pt")["pixel_values"].to(device)
        # fp32 for a simple, reproducible baseline; no model updates.
        output = model(pixel_values=pixels)
        features = output.last_hidden_state[:, 0].float()
        if not torch.isfinite(features).all() or (features.norm(dim=1) <= 0).any():
            raise ValueError("Invalid encoder features")
        chunks.append(F.normalize(features, dim=1).cpu())
    return torch.cat(chunks)


def search_embeddings(query, gallery, labels, classes, rule="nearest", class_k=3):
    """Return predicted class IDs and all cosine similarities; never reads query labels."""
    query = F.normalize(query.float(), dim=1)
    gallery = F.normalize(gallery.float(), dim=1)
    if query.shape[1] != gallery.shape[1]:
        raise ValueError("Query/gallery embedding dimensions differ")
    similarities = query @ gallery.T
    label_ids = torch.tensor([classes.index(y) for y in labels])
    if rule == "nearest":
        prediction = label_ids[similarities.argmax(dim=1)]
    elif rule == "class_topk":
        scores = []
        for i in range(len(classes)):
            values = similarities[:, label_ids == i]
            scores.append(values.topk(min(class_k, values.shape[1]), dim=1).values.mean(dim=1))
        prediction = torch.stack(scores, dim=1).argmax(dim=1)
    else:
        raise ValueError(f"Unknown rule: {rule}")
    return prediction, similarities


def build(args):
    # Fail closed: do not silently build a gallery out of validation/real files.
    if "train" not in Path(args.filename).name.lower():
        raise ValueError("Gallery filename must identify the synthetic TRAIN split")
    path = resolve_parquet(args.data, args.filename)
    if "train" not in path.name.lower():
        raise ValueError("Gallery must be a synthetic TRAIN Parquet, not validation or real-test")
    images, ids, names, labels, column = read_images(path, args.column, require_labels=True)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gallery IDs; inspect the source split")
    processor, model, device = load_encoder(args.model_id)
    features = extract(images, processor, model, device, args.batch_size)
    gallery = {
        "format_version": 1,
        "model_id": args.model_id,
        "revision": getattr(model.config, "_commit_hash", None),
        "pooling": "cls",
        "features": features,
        "labels": labels,
        "ids": ids,
        "file_names": names,
        "classes": sorted(set(labels)),
        "source": args.data,
        "source_filename": args.filename,
        "image_column": column,
        "rule": args.rule,
        "class_k": args.class_k,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gallery, output)
    print(f"Saved gallery: {output}; embeddings={tuple(features.shape)}; classes={len(gallery['classes'])}")
    print(f"Fixed prediction rule: {args.rule}; class_k={args.class_k} (only used for class_topk)")


def predict(args):
    gallery = torch.load(args.gallery, map_location="cpu", weights_only=True)
    if gallery.get("format_version") != 1 or gallery.get("pooling") != "cls":
        raise ValueError("Unsupported gallery format")
    path = resolve_parquet(args.data, args.filename)
    images, ids, names, true_labels, column = read_images(path, args.column)
    overlap = set(ids) & set(gallery["ids"])
    name_overlap = set(names) & set(gallery["file_names"])
    if overlap or name_overlap:
        raise ValueError("Query/gallery IDs or filenames overlap; refusing self-retrieval evaluation")
    processor, model, device = load_encoder(gallery["model_id"], gallery["revision"])
    query = extract(images, processor, model, device, args.batch_size)
    classes = gallery["classes"]
    prediction, similarities = search_embeddings(
        query, gallery["features"], gallery["labels"], classes,
        gallery["rule"], gallery["class_k"],
    )
    values, indices = similarities.topk(min(args.neighbors, len(gallery["labels"])), dim=1)
    predicted_labels = [classes[i] for i in prediction.tolist()]
    rows = []
    for i, label in enumerate(predicted_labels):
        neighbors = [
            {"gallery_id": gallery["ids"][j], "file_name": gallery["file_names"][j],
             "label": gallery["labels"][j], "cosine_similarity": float(score)}
            for j, score in zip(indices[i].tolist(), values[i].tolist())
        ]
        rows.append({
            "file_name": names[i], "prediction": label,
            "ground_truth": true_labels[i] if true_labels[i] is not None else "",
            "correct": int(label == true_labels[i]) if true_labels[i] is not None else "",
            "nearest_cosine_similarity": float(values[i, 0]),
            "neighbors": json.dumps(neighbors, ensure_ascii=False),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "count": len(rows), "gallery_count": len(gallery["labels"]),
        "model_id": gallery["model_id"], "rule": gallery["rule"], "class_k": gallery["class_k"],
        "query_image_column": column, "classes": classes,
        "note": "Cosine similarity is not a probability; query labels are used only for reporting.",
    }
    # Prediction above is entirely independent of the optional ground truth.
    labeled = [i for i, y in enumerate(true_labels) if y is not None]
    if labeled:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
        unknown = set(true_labels[i] for i in labeled) - set(classes)
        if unknown:
            raise ValueError(f"Query labels not present in gallery: {sorted(unknown)}")
        y = [true_labels[i] for i in labeled]
        p = [predicted_labels[i] for i in labeled]
        report.update({
            "labeled_count": len(labeled),
            "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=classes, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p)),
            "confusion_matrix": confusion_matrix(y, p, labels=classes).tolist(),
            "per_class": classification_report(y, p, labels=classes, output_dict=True, zero_division=0),
        })
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("count", "gallery_count", "rule", "accuracy", "macro_f1", "balanced_accuracy") if k in report}, indent=2))
    print(f"Predictions: {output}\nMetrics: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="Embed synthetic TRAIN images; no training")
    build_parser.add_argument("--data", default="KhanhChien/Data-Sea-Full-Synthetic")
    build_parser.add_argument("--filename", default="data/train-full-sea.parquet")
    build_parser.add_argument("--column", default="crop_image")
    build_parser.add_argument("--model_id", default="facebook/dinov2-base")
    build_parser.add_argument("--rule", choices=["nearest", "class_topk"], default="nearest")
    build_parser.add_argument("--class_k", type=int, default=3)
    build_parser.add_argument("--output", required=True)
    build_parser.add_argument("--batch_size", type=int, default=16)
    build_parser.set_defaults(func=build)
    query_parser = sub.add_parser("predict", help="Search gallery; optionally report held-out metrics")
    query_parser.add_argument("--gallery", required=True)
    query_parser.add_argument("--data", required=True)
    query_parser.add_argument("--filename", default="data/selected_sea_only_real.parquet")
    query_parser.add_argument("--column", default=None)
    query_parser.add_argument("--output", required=True)
    query_parser.add_argument("--batch_size", type=int, default=16)
    query_parser.add_argument("--neighbors", type=int, default=5)
    query_parser.set_defaults(func=predict)
    args = parser.parse_args()
    for name in ("batch_size", "class_k", "neighbors"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    args.func(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Pack the original 640x640 Sea images plus bbox crops into Parquet.

The public Sea Parquet currently contains only the bbox-cropped image. This
script creates the data needed for domain-robust fine-tuning: both the
original full image and the object crop are stored in every row.
"""

import argparse
import io
import json
import math
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from PIL import Image
from tqdm.auto import tqdm


def encode_image(image, quality):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    return output.getvalue()


def crop_from_bbox(image, bbox, padding):
    if not isinstance(bbox, list) or len(bbox) != 4:
        return image
    x, y, w, h = [float(value) for value in bbox]
    px, py = w * padding, h * padding
    box = (
        max(0, math.floor(x - px)),
        max(0, math.floor(y - py)),
        min(image.width, math.ceil(x + w + px)),
        min(image.height, math.ceil(y + h + py)),
    )
    if box[2] <= box[0] + 4 or box[3] <= box[1] + 4:
        return image
    return image.crop(box)


def records(jsonl_path, images_dir, crop_padding, quality):
    lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line_no, line in enumerate(tqdm(lines, desc=f"Packing {jsonl_path.name}"), 1):
        item = json.loads(line)
        label = str(item.get("gt_class", ""))
        if not label.startswith("Sea."):
            continue
        image_name = item.get("file_name") or Path(item.get("image", "")).name
        image_path = images_dir / image_name
        if not image_path.is_file():
            image_path = images_dir.parent.parent / item.get("image", "")
        if not image_path.is_file():
            raise FileNotFoundError(f"{jsonl_path}:{line_no}: {image_path}")

        with Image.open(image_path) as source:
            full = source.convert("RGB")
        crop = crop_from_bbox(full, item.get("bbox"), crop_padding)
        item_id = str(item.get("id", f"{label}_{line_no}"))
        yield {
            "id": item_id,
            "file_name": str(image_name),
            "gt_class": label,
            "bbox": [float(value) for value in item.get("bbox", [0, 0, 0, 0])],
            "full_image": {"bytes": encode_image(full, quality), "path": f"{item_id}_full.jpg"},
            "crop_image": {"bytes": encode_image(crop, quality), "path": f"{item_id}_crop.jpg"},
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--validation-jsonl", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-padding", type=float, default=0.08)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    args = parser.parse_args()

    features = Features({
        "id": Value("string"),
        "file_name": Value("string"),
        "gt_class": Value("string"),
        "bbox": [Value("float32")],
        "full_image": HFImage(),
        "crop_image": HFImage(),
    })
    kwargs = {
        "images_dir": args.images_dir,
        "crop_padding": args.crop_padding,
        "quality": args.jpeg_quality,
    }
    train = Dataset.from_generator(
        records, gen_kwargs={"jsonl_path": args.train_jsonl, **kwargs}, features=features
    )
    validation = Dataset.from_generator(
        records, gen_kwargs={"jsonl_path": args.validation_jsonl, **kwargs}, features=features
    )
    data_dir = args.output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(str(data_dir / "train-full-sea.parquet"))
    validation.to_parquet(str(data_dir / "validation-full-sea.parquet"))
    DatasetDict({"train": train, "validation": validation}).save_to_disk(
        str(args.output_dir / "dataset")
    )
    print(f"Wrote {len(train):,} train and {len(validation):,} validation rows")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Qwen evidence reranker for sea_rag.py outputs."""
import argparse, csv, io, json, re, sys
from pathlib import Path
from PIL import Image
import torch

def decode(value):
    if isinstance(value, dict): value = value.get("bytes") or value.get("path")
    if isinstance(value, Image.Image): return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as image: return image.convert("RGB")
    if isinstance(value, str):
        with Image.open(value) as image: return image.convert("RGB")
    raise ValueError(type(value))

def find_query_rows(path, column):
    import pyarrow.parquet as pq
    schema = set(pq.read_schema(path).names)
    if column not in schema: raise ValueError(f"Missing {column}; columns={sorted(schema)}")
    cols = [column] + [x for x in ("file_name", "gt_class") if x in schema]
    result = {}
    for i, row in enumerate(pq.read_table(path, columns=cols).to_pylist()):
        result[str(row.get("file_name") or i)] = {"image": decode(row[column]), "ground_truth": str(row.get("gt_class") or "")}
    return result

def read_gallery_images(evidence_dir, ids):
    images = []
    for ident in ids:
        path = Path(evidence_dir) / f"{ident}.jpg"
        if not path.is_file(): raise FileNotFoundError(path)
        with Image.open(path).convert("RGB") as image: images.append(image.copy())
    return images

def parse_answer(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    return match.group(1).strip() if match else ""

def load_qwen(repo, adapter, checkpoint):
    repo = Path(repo).resolve(); sys.path.insert(0, str(repo))
    from inference_selected_sea import load_model, resolve_adapter
    adapter_path = resolve_adapter(adapter, checkpoint) if checkpoint else resolve_adapter(adapter, "")
    return load_model(adapter_path)

def rerank(query, candidates, evidence_dir, processor, model):
    labels = [x["class"] for x in candidates]
    content = [{"type": "image", "image": query}]
    text = ("Compare the query vessel with retrieved synthetic evidence. Ignore water, sky, color, watermark, rendering style and image quality. Focus on hull, superstructure, mast, radar, tower, chimney, deck and weapon arrangement. Choose exactly one candidate. Return only <answer>Exact_Class_Name</answer>.\nCandidates:\n")
    all_images = [query]
    for candidate in candidates:
        evidence = read_gallery_images(evidence_dir, candidate["evidence"])
        content.append({"type": "text", "text": f"Evidence for {candidate['class']}:"})
        for image in evidence:
            content.append({"type": "image", "image": image}); all_images.append(image)
        text += f"- {candidate['class']}\n"
    content.append({"type": "text", "text": text})
    messages = [{"role": "system", "content": [{"type": "text", "text": "You are a careful fine-grained naval vessel reranker."}]}, {"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[prompt], images=all_images, padding=True, truncation=False, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=32, do_sample=False, pad_token_id=processor.tokenizer.pad_token_id)
    raw = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=False)[0]
    answer = parse_answer(raw)
    if answer not in labels: answer = labels[0]
    for image in all_images[1:]: image.close()
    return answer, raw

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True); parser.add_argument("--query_data", required=True)
    parser.add_argument("--query_filename", default="data/selected_sea_only_real.parquet"); parser.add_argument("--query_column", default="image_bytes")
    parser.add_argument("--gallery", required=True); parser.add_argument("--qwen_repo", required=True); parser.add_argument("--qwen_adapter", required=True); parser.add_argument("--qwen_checkpoint", default=""); parser.add_argument("--output", required=True); parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("Qwen reranking requires CUDA")
    from sea_rag import resolve
    rows = list(csv.DictReader(Path(args.predictions).open(newline="", encoding="utf-8")))
    if args.limit: rows = rows[:args.limit]
    query_rows = find_query_rows(resolve(args.query_data, args.query_filename), args.query_column)
    evidence_dir = Path(args.gallery).with_suffix("") / "evidence"
    processor, model = load_qwen(args.qwen_repo, args.qwen_adapter, args.qwen_checkpoint)
    for index, row in enumerate(rows, 1):
        item = query_rows[row["file_name"]]; prediction, raw = rerank(item["image"], json.loads(row["evidence"]), evidence_dir, processor, model)
        row["qwen_prediction"] = prediction; row["prediction"] = prediction; row["qwen_raw"] = raw; row["ground_truth"] = item["ground_truth"]; row["correct"] = str(int(prediction == item["ground_truth"])) if item["ground_truth"] else ""
        print(f"[{index}/{len(rows)}] {row['file_name']} -> {prediction} | gt={row['ground_truth']} | correct={row['correct']}", flush=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    labeled = [x for x in rows if x["ground_truth"]]; report = {"count":len(rows), "labeled_count":len(labeled), "note":"Synthetic gallery evidence only; labels are metrics-only."}
    if labeled:
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        truth=[x["ground_truth"] for x in labeled]; pred=[x["prediction"] for x in labeled]
        report.update({"accuracy":float(accuracy_score(truth,pred)), "macro_f1":float(f1_score(truth,pred,average="macro",zero_division=0)), "balanced_accuracy":float(balanced_accuracy_score(truth,pred))})
    metrics=output.with_suffix(".metrics.json"); metrics.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__ == "__main__": main()

#!/usr/bin/env python
"""Hybrid synthetic-only RAG retrieval for Sea classification."""
import argparse, csv, io, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

CLASSES = [
    "Sea.ArleighBurke", "Sea.Buyan", "Sea.GeraldFord", "Sea.Gorshkov",
    "Sea.Independence", "Sea.Kilo", "Sea.Kirov", "Sea.Molniya",
    "Sea.Nanuchka", "Sea.OsaII", "Sea.Ticonderoga", "Sea.Type002.Shandong",
    "Sea.Type022.Houbei", "Sea.Type051B.Luhai", "Sea.Type053H3.JiangweiII",
    "Sea.Type054A.JiangkaiII", "Sea.Type055.Renhai", "Sea.TypeAsahi",
    "Sea.TypeChengKung", "Sea.TypeKangDing", "Sea.Wasp",
]

def resolve(data, filename):
    path = Path(data).expanduser()
    if path.is_file(): return path
    if path.is_dir():
        target = path / filename
        if target.is_file(): return target
        raise FileNotFoundError(target)
    text = str(data)
    if text.startswith(("/", "\\", ".")) or ":" in text or text.endswith(".parquet"):
        raise FileNotFoundError(data)
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=text, repo_type="dataset", filename=filename))

def decode(value):
    if isinstance(value, dict): value = value.get("bytes") or value.get("path")
    if isinstance(value, Image.Image): return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as im: return im.convert("RGB")
    if isinstance(value, str):
        with Image.open(value) as im: return im.convert("RGB")
    raise ValueError(f"Unsupported image value: {type(value)}")

def read_records(path, column=None, require_labels=False):
    import pyarrow.parquet as pq
    schema = set(pq.read_schema(path).names)
    if column is None: column = next((x for x in ("image_bytes", "image", "crop_image", "full_image") if x in schema), None)
    if column not in schema: raise ValueError(f"Missing image column {column}; columns={sorted(schema)}")
    wanted = [column] + [x for x in ("id", "file_name", "gt_class", "crop_image", "full_image") if x in schema]
    out = []
    for i, row in enumerate(pq.read_table(path, columns=list(dict.fromkeys(wanted))).to_pylist()):
        ident = str(row.get("id") or row.get("file_name") or i)
        out.append({"id": ident, "file_name": str(row.get("file_name") or ident), "image": row.get(column), "crop_image": row.get("crop_image"), "full_image": row.get("full_image"), "label": row.get("gt_class")})
    if not out: raise ValueError(f"Empty Parquet: {path}")
    if require_labels:
        bad = sorted({str(x["label"]) for x in out if x["label"] not in CLASSES})
        if bad: raise ValueError(f"Invalid labels: {bad}")
    return out, column

def pad_square(image, fill=(114,114,114)):
    image = image.convert("RGB"); side = max(image.size)
    out = Image.new("RGB", (side, side), fill)
    out.paste(image, ((side-image.width)//2, (side-image.height)//2))
    return out

def query_views(image):
    image = image.convert("RGB"); width, height = image.size; out = [pad_square(image)]
    if width >= 1.2 * height:
        window = max(height, int(width*.72))
        for left in (0, (width-window)//2, width-window): out.append(pad_square(image.crop((left,0,left+window,height))))
    elif height >= 1.2 * width:
        window = max(width, int(height*.72))
        for top in (0, (height-window)//2, height-window): out.append(pad_square(image.crop((0,top,width,top+window))))
    return out

def load_encoder(model_id, device):
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained(model_id, trust_remote_code=True)
    kwargs = {"trust_remote_code": True}
    if device.type == "cuda": kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try: model = AutoModel.from_pretrained(model_id, **kwargs)
    except TypeError:
        dtype = kwargs.pop("dtype", None)
        if dtype is not None: kwargs["torch_dtype"] = dtype
        model = AutoModel.from_pretrained(model_id, **kwargs)
    return processor, model.to(device).eval().requires_grad_(False)

def pooled(model, model_id, pixels):
    output = model(pixel_values=pixels)
    if "siglip" in model_id.lower() and hasattr(output, "pooler_output"): value = output.pooler_output
    elif hasattr(output, "last_hidden_state"): value = output.last_hidden_state[:,0]
    elif hasattr(output, "pooler_output"): value = output.pooler_output
    else: raise ValueError(f"No image feature in {type(output).__name__}")
    return F.normalize(value.float(), dim=1)

@torch.inference_mode()
def embed(images, processor, model, model_id, device, batch_size, title):
    from tqdm.auto import tqdm
    chunks = []
    for start in tqdm(range(0, len(images), batch_size), desc=title):
        pixels = processor(images=images[start:start+batch_size], return_tensors="pt")["pixel_values"].to(device)
        chunks.append(pooled(model, model_id, pixels).cpu())
    return torch.cat(chunks)

def gallery_image(record): return decode(record.get("crop_image") or record.get("image") or record.get("full_image"))
def query_image(record): return decode(record.get("image") or record.get("full_image") or record.get("crop_image"))

def build(args):
    if "train" not in Path(args.filename).name.lower(): raise ValueError("--filename must be synthetic TRAIN")
    path = resolve(args.data, args.filename)
    if "train" not in path.name.lower(): raise ValueError("Refusing non-train gallery")
    records, column = read_records(path, args.column, True)
    if len({x["id"] for x in records}) != len(records): raise ValueError("Duplicate gallery IDs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp, sm = load_encoder(args.siglip_model, device); dp, dm = load_encoder(args.dino_model, device)
    images = [gallery_image(x) for x in records]
    sig = embed(images, sp, sm, args.siglip_model, device, args.batch_size, "SigLIP gallery")
    dino = embed(images, dp, dm, args.dino_model, device, args.batch_size, "DINO gallery")
    gallery = {"format_version":2, "siglip_model":args.siglip_model, "dino_model":args.dino_model, "siglip_features":sig, "dino_features":dino, "ids":[x["id"] for x in records], "file_names":[x["file_name"] for x in records], "labels":[x["label"] for x in records], "classes":sorted({x["label"] for x in records}), "source":str(args.data), "source_filename":args.filename, "image_column":column}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True); torch.save(gallery, output)
    evidence = output.with_suffix("") / "evidence"; evidence.mkdir(parents=True, exist_ok=True)
    for record, image in zip(records, images):
        image.thumbnail((448,448), Image.Resampling.LANCZOS); image.save(evidence/f"{record['id']}.jpg", quality=92)
    print(f"Saved {output}; SigLIP={tuple(sig.shape)} DINO={tuple(dino.shape)} classes={len(gallery['classes'])}")
    print(f"Evidence directory: {evidence}")

def fused_scores(query_sig, query_dino, gallery):
    sig = (query_sig @ gallery["siglip_features"].T).max(dim=0).values
    dino = (query_dino @ gallery["dino_features"].T).max(dim=0).values
    sr = torch.argsort(torch.argsort(sig, descending=True)); dr = torch.argsort(torch.argsort(dino, descending=True)); n=len(sig)
    return .5*(1-sr.float()/max(1,n-1)) + .5*(1-dr.float()/max(1,n-1))

def candidates(scores, gallery, count, evidence_per_class):
    result=[]
    for cls in gallery["classes"]:
        indices=[i for i,label in enumerate(gallery["labels"]) if label==cls]
        indices.sort(key=lambda i:float(scores[i]), reverse=True)
        best=[float(scores[i]) for i in indices[:min(3,len(indices))]]
        result.append({"class":cls,"score":float(np.mean(best)),"evidence":[gallery["ids"][i] for i in indices[:evidence_per_class]]})
    return sorted(result,key=lambda x:x["score"],reverse=True)[:count]

@torch.inference_mode()
def predict(args):
    gallery=torch.load(args.gallery,map_location="cpu",weights_only=True)
    if gallery.get("format_version")!=2: raise ValueError("Rebuild with sea_rag.py build")
    path=resolve(args.data,args.filename); records,column=read_records(path,args.column)
    if set(gallery["ids"]).intersection(x["id"] for x in records) or set(gallery["file_names"]).intersection(x["file_name"] for x in records): raise ValueError("Query/gallery overlap")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp,sm=load_encoder(gallery["siglip_model"],device); dp,dm=load_encoder(gallery["dino_model"],device); rows=[]
    for start in range(0,len(records),args.batch_size):
        batch=records[start:start+args.batch_size]; originals=[query_image(x) for x in batch]; groups=[query_views(x) for x in originals]; flat=[v for group in groups for v in group]
        sig=embed(flat,sp,sm,gallery["siglip_model"],device,args.embed_batch,"SigLIP query"); dino=embed(flat,dp,dm,gallery["dino_model"],device,args.embed_batch,"DINO query"); cursor=0
        for record,original,group in zip(batch,originals,groups):
            n=len(group); score=fused_scores(sig[cursor:cursor+n],dino[cursor:cursor+n],gallery); cursor+=n; cand=candidates(score,gallery,args.candidate_classes,args.evidence_per_class); truth=record.get("label") or ""; labels=[x["class"] for x in cand]
            rows.append({"file_name":record["file_name"],"prediction":cand[0]["class"],"ground_truth":truth,"correct":int(cand[0]["class"]==truth) if truth else "","candidate_hit":int(truth in labels) if truth else "","candidate_classes":"|".join(labels),"evidence":json.dumps(cand,ensure_ascii=False)}); original.close()
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report={"count":len(rows),"gallery_count":len(gallery["labels"]),"candidate_classes":args.candidate_classes,"evidence_per_class":args.evidence_per_class,"query_image_column":column,"note":"Synthetic TRAIN gallery only; labels are metrics-only."}; labeled=[x for x in rows if x["ground_truth"]]
    if labeled:
        from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
        truth=[x["ground_truth"] for x in labeled]; pred=[x["prediction"] for x in labeled]
        report.update({"labeled_count":len(labeled),"accuracy":float(accuracy_score(truth,pred)),"macro_f1":float(f1_score(truth,pred,labels=gallery["classes"],average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(truth,pred)),"candidate_recall":float(np.mean([int(x["candidate_hit"]) for x in labeled]))})
    metrics=output.with_suffix(".metrics.json"); metrics.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); print(json.dumps(report,indent=2,ensure_ascii=False)); print(f"Predictions: {output}\nMetrics: {metrics}")

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("build"); p.add_argument("--data",default="KhanhChien/Data-Sea-Full-Synthetic"); p.add_argument("--filename",default="data/train-full-sea.parquet"); p.add_argument("--column",default="crop_image"); p.add_argument("--siglip_model",default="google/siglip2-base-patch16-224"); p.add_argument("--dino_model",default="facebook/dinov2-base"); p.add_argument("--output",required=True); p.add_argument("--batch_size",type=int,default=16); p.set_defaults(func=build)
    p=sub.add_parser("predict"); p.add_argument("--gallery",required=True); p.add_argument("--data",required=True); p.add_argument("--filename",default="data/selected_sea_only_real.parquet"); p.add_argument("--column",default=None); p.add_argument("--output",required=True); p.add_argument("--batch_size",type=int,default=1); p.add_argument("--embed_batch",type=int,default=16); p.add_argument("--candidate_classes",type=int,default=5); p.add_argument("--evidence_per_class",type=int,default=3); p.set_defaults(func=predict)
    args=parser.parse_args(); args.func(args)

if __name__=="__main__": main()

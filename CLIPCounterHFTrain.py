
import json, re, os
from pathlib import Path
from collections import defaultdict
import math
import requests
import pandas as pd
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import argparse
#hibrid za so zero-shot i so supervised head i so fusion

OBJECTS_JSON = Path(r"C:\Users\ilios\PycharmProjects\PythonProject\objects.json/objects.json")
IMAGES_JSON  = Path(r"C:\Users\ilios\PycharmProjects\PythonProject\image_data.json/image_data.json")
IMG_OUT_DIR  = Path(r"C:\Users\ilios\Desktop\images")
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGES = 500
MAX_CARS_FOR_PROMPT = 10

VEHICLE_KEYWORDS = {"car","automobile","auto","sedan","suv","van","taxi"}
VEH_RE = re.compile(r"\b(" + "|".join(sorted(VEHICLE_KEYWORDS)) + r")s?\b", re.I)


device = "cuda" if torch.cuda.is_available() else "cpu"
HF_MODEL_ID = "openai/clip-vit-large-patch14-336"

model = CLIPModel.from_pretrained(HF_MODEL_ID).to(device)
processor = CLIPProcessor.from_pretrained(HF_MODEL_ID)
model.eval()
torch.set_grad_enabled(False)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def is_vehicle_label(label: str) -> bool:
    return bool(label and VEH_RE.search(label))

def filter_vehicle_images(objects_data) -> dict:
    counts = defaultdict(int)
    for entry in objects_data:
        img_id = entry["image_id"]
        for obj in entry.get("objects", []):
            for n in obj.get("names", []):
                if is_vehicle_label(n):
                    counts[img_id] += 1
    return counts

def download_image(url: str, out_path: Path) -> bool:
    if out_path.exists():
        return True
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        out_path.write_bytes(r.content)
        return True
    except:
        return False


def _build_prompt_variants(k: int):
    if k == 1:
        return [
            f"a traffic camera photo showing exactly {k} passenger car",
            f"a street scene with exactly {k} car visible",
            f"a dashcam image with {k} car in view",
            f"a parking lot photo containing {k} car (no trucks, no buses)",
            f"an urban roadway photo with exactly {k} car and no buses or trucks",
        ]
    else:
        return [
            f"a traffic camera photo showing exactly {k} passenger cars",
            f"a street scene with exactly {k} cars visible",
            f"a dashcam image with {k} cars in view",
            f"a parking lot photo containing {k} cars (no trucks, no buses)",
            f"an urban roadway photo with exactly {k} cars and no buses or trucks",
        ]

@torch.no_grad()
def _encode_text_ensemble_hf(variants):

    text_inputs = processor(text=variants, padding=True, return_tensors="pt").to(device)
    t_feats = model.get_text_features(**text_inputs)
    t_feats = F.normalize(t_feats, dim=-1)
    t_mean  = t_feats.mean(dim=0, keepdim=True)
    return F.normalize(t_mean, dim=-1)

@torch.no_grad()
def _build_txt_feats_hf(max_k: int, include_more_than_k: bool = True) -> torch.Tensor:

    txt_feats = []#se gradi matrica za klasite
    for k in range(max_k + 1):
        variants = _build_prompt_variants(k)
        txt_feats.append(_encode_text_ensemble_hf(variants))
    if include_more_than_k:
        more_variants = [f"a street scene with more than {max_k} passenger cars visible"]
        txt_feats.append(_encode_text_ensemble_hf(more_variants))
    return torch.cat(txt_feats, dim=0)

_TXT_FEATS_CACHE = {}
def _get_txt_feats_cached(max_k: int, include_more_than_k: bool = True) -> torch.Tensor:
    key = (HF_MODEL_ID, max_k, include_more_than_k)
    if key not in _TXT_FEATS_CACHE:
        _TXT_FEATS_CACHE[key] = _build_txt_feats_hf(max_k, include_more_than_k).to(device)
    return _TXT_FEATS_CACHE[key]


@torch.no_grad()
def extract_img_feat(pil_img: Image.Image) -> np.ndarray:#se vrakja eden vektor za so head

    ims = [pil_img, ImageOps.mirror(pil_img)]
    inputs = processor(images=ims, return_tensors="pt").to(device)
    feats = model.get_image_features(**inputs)
    feats = F.normalize(feats, dim=-1)
    feat = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
    return feat.squeeze(0).cpu().numpy()


def image_augmentations(pil_img: Image.Image):
    return [pil_img, ImageOps.mirror(pil_img)]

@torch.no_grad()
def clip_estimate_cars(pil_img: Image.Image,max_k: int = 10,include_more_than_k: bool = True,head=None,fuse_alpha: float = 0.5,):

    aug_imgs   = image_augmentations(pil_img)
    img_inputs = processor(images=aug_imgs, return_tensors="pt").to(device)
    i_feats    = model.get_image_features(**img_inputs)##vektor
    i_feats    = F.normalize(i_feats, dim=-1)
    i_feat     = F.normalize(i_feats.mean(dim=0, keepdim=True), dim=-1)


    t_feat = _get_txt_feats_cached(max_k, include_more_than_k)#vektor od prompts
    logits = i_feat @ t_feat.T#gi mnozime da vidime dali se slicni
    scale  = torch.clamp(model.logit_scale.exp(), 1.0, 100.0)
    logits = logits * scale

    probs_zeroshot = logits.softmax(dim=1).squeeze(0)
    N = probs_zeroshot.shape[0]


    if head is not None:

        feat_np = i_feat.squeeze(0).cpu().numpy().reshape(1, -1)

        if hasattr(head, "predict_proba"):
            probs_head = head.predict_proba(feat_np).ravel()

            pz = probs_zeroshot[:max_k+1].cpu().numpy()
            p_fused = (1.0 - fuse_alpha) * pz + fuse_alpha * probs_head#fused together, ama imam napraveno sekogas da i veruvame 100% na naucenata head

            s = p_fused.sum()
            if s > 0:
                p_fused = p_fused / s #za da e validno(do 1)

            if N == (max_k + 2):
                probs_all = np.concatenate([p_fused, probs_zeroshot[-1:].cpu().numpy()])#za ako iame plus uste edna klasa za >k ama koa imame head nemame extra klasa
            else:
                probs_all = p_fused
            probs = torch.from_numpy(probs_all)
        else:

            probs = probs_zeroshot
    else:
        probs = probs_zeroshot


    k_pred = int(probs.argmax().item())
    ent  = -(probs * (probs + 1e-12).log()).sum().item()
    conf = 1.0 - ent / math.log(len(probs))

    return k_pred, probs.tolist(), conf


def train_head(train_csv: str, out_path: str):

    df = pd.read_csv(train_csv)
    X, y = [], []

    for _, row in df.iterrows():
        path = Path(row["image_path"])
        if not path.exists():
            continue
        try:
            pil = Image.open(path).convert("RGB")
        except:
            continue
        X.append(extract_img_feat(pil))#vektor
        y.append(int(row["count"]))

    if len(X) == 0:
        raise RuntimeError("No training samples loaded. Check your CSV paths.")

    X = np.stack(X)
    y = np.array(y)


    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)#podelba na train i na test
    except ValueError:

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)


    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("lr", LogisticRegression(max_iter=2000,class_weight="balanced",multi_class="auto"))
    ])

    clf.fit(Xtr, ytr)#trenirame
    preds = clf.predict(Xte)
    acc = accuracy_score(yte, preds)
    mae = mean_absolute_error(yte, preds)
    print(f"[Head] Test Accuracy: {acc:.3f}")
    print(f"[Head] Test MAE:      {mae:.3f}")

    joblib.dump(clf, out_path)#go zacuvuvame vo file
    print(f"[Head] Saved to {out_path}")


def run_infer(max_images=MAX_IMAGES, max_k=MAX_CARS_FOR_PROMPT, head_path=None, include_more_than_k=True, fuse_alpha=0.5):

    head = None
    if head_path and Path(head_path).exists():
        head = joblib.load(head_path)
        print(f"[Head] Loaded: {head_path}")

    objects_data = load_json(OBJECTS_JSON)
    image_meta   = {int(d["image_id"]): d["url"] for d in load_json(IMAGES_JSON) if d.get("url")}
    veh_counts   = filter_vehicle_images(objects_data)

    rows = []
    picked = list(veh_counts.items())[:max_images]
    for img_id, vg_count in picked:
        url = image_meta.get(img_id)
        if not url:
            continue

        out_path = IMG_OUT_DIR / f"{img_id}.jpg"
        if not download_image(url, out_path):
            continue

        try:
            pil = Image.open(out_path).convert("RGB")
        except:
            continue

        k_pred, probs, conf = clip_estimate_cars(
            pil,
            max_k=max_k,
            include_more_than_k=include_more_than_k,
            head=head,
            fuse_alpha=fuse_alpha
        )

        rows.append({
            "image_id": img_id,
            "vg_vehicle_objs": vg_count,
            "clip_count_pred": k_pred,
            "confidence": conf,
            "max_softmax": float(max(probs)),
            "image_path": str(out_path)
        })

    df = pd.DataFrame(rows)
    print(df.head())
    df.to_csv("vg_clip_vehicle_counts.csv", index=False)
    print("[Infer] Wrote vg_clip_vehicle_counts.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["infer","train"], default="infer", help="infer=run VG demo; train=train head")
    parser.add_argument("--train_csv", type=str, help="CSV with image_path,count (for --mode train)")
    parser.add_argument("--head_out", type=str, default="clip_count_head.joblib", help="Output path for trained head")
    parser.add_argument("--head_in", type=str, help="Path to trained head (for fusion in infer)")
    parser.add_argument("--max_k", type=int, default=MAX_CARS_FOR_PROMPT, help="Max count class K for zero-shot and head")
    parser.add_argument("--max_images", type=int, default=MAX_IMAGES, help="How many VG images to sample in infer")
    parser.add_argument("--include_more_than_k", action="store_true", help="(ignored in fusion-only)")
    parser.add_argument("--fuse_alpha", type=float, default=1.0, help="(fusion-only) use head=1.0")#nashi atributi dodavame

    args = parser.parse_args()

    if args.mode == "train":
        if not args.train_csv:
            raise SystemExit("Please provide --train_csv for training.")
        train_head(args.train_csv, args.head_out)
    else:

        args.include_more_than_k = False
        args.fuse_alpha = 1.0

        run_infer(
            max_images=args.max_images,
            max_k=args.max_k,
            head_path=args.head_in,
            include_more_than_k=args.include_more_than_k,
            fuse_alpha=args.fuse_alpha
        )

if __name__ == "__main__":
    main()

# vehicle_count_clip_hf.py
import json, re, os
from pathlib import Path
from collections import defaultdict
import math
import requests
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


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
    with open(path, "r", encoding="utf-8") as f:
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
def _build_txt_feats_hf(max_k: int) -> torch.Tensor:

    txt_feats = []
    for k in range(max_k + 1):
        variants = _build_prompt_variants(k)
        txt_feats.append(_encode_text_ensemble_hf(variants))
    # OPTIONAL: Add a ">K" class
    more_variants = [f"a street scene with more than {max_k} passenger cars visible"]
    txt_feats.append(_encode_text_ensemble_hf(more_variants))
    return torch.cat(txt_feats, dim=0)  # [N, D]


_TXT_FEATS_CACHE = {}
def _get_txt_feats_cached(max_k: int) -> torch.Tensor:
    key = (HF_MODEL_ID, max_k)
    if key not in _TXT_FEATS_CACHE:
        _TXT_FEATS_CACHE[key] = _build_txt_feats_hf(max_k).to(device)
    return _TXT_FEATS_CACHE[key]


def image_augmentations(pil_img: Image.Image):

    return [pil_img, ImageOps.mirror(pil_img)]

@torch.no_grad()
def clip_estimate_cars(pil_img: Image.Image, max_k: int = 10):

    aug_imgs   = image_augmentations(pil_img)
    img_inputs = processor(images=aug_imgs, return_tensors="pt").to(device)
    i_feats    = model.get_image_features(**img_inputs)
    i_feats    = F.normalize(i_feats, dim=-1)
    i_feat     = F.normalize(i_feats.mean(dim=0, keepdim=True), dim=-1)


    t_feat = _get_txt_feats_cached(max_k)


    logits = i_feat @ t_feat.T
    scale  = torch.clamp(model.logit_scale.exp(), 1.0, 100.0)
    logits = logits * scale


    probs  = logits.softmax(dim=1).squeeze(0)
    k_pred = int(probs.argmax().item())


    ent  = -(probs * (probs + 1e-12).log()).sum().item()
    conf = 1.0 - ent / math.log(len(probs))

    return k_pred, probs.tolist(), conf


def main():

    objects_data = load_json(OBJECTS_JSON)
    image_meta   = {int(d["image_id"]): d["url"] for d in load_json(IMAGES_JSON) if d.get("url")}
    veh_counts   = filter_vehicle_images(objects_data)

    rows = []
    picked = list(veh_counts.items())[:MAX_IMAGES]
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

        k_pred, probs, conf = clip_estimate_cars(pil, MAX_CARS_FOR_PROMPT)
        rows.append({
            "image_id": img_id,
            "vg_vehicle_objs": vg_count,
            "clip_count_pred": k_pred,
            "confidence": conf,
            "max_softmax": max(probs),
            "image_path": str(out_path)
        })

    df = pd.DataFrame(rows)
    print(df.head())
    df.to_csv("vgh_clip_vehicle_counts.csv", index=False)

if __name__ == "__main__":
    main()

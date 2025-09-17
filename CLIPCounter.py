import json, re, os  #za da mozam da gi loadnam json formatite, regex
from pathlib import Path
from collections import defaultdict #za broenje
import requests #da gi danlodiram slikitr
from PIL import Image
import torch, clip
import pandas as pd # za csv
import torch
import torch.nn.functional as F
import math
from PIL import ImageOps
#zero-shot counting


OBJECTS_JSON = Path(r"C:\Users\ilios\PycharmProjects\PythonProject\objects.json/objects.json")
IMAGES_JSON = Path(r"C:\Users\ilios\PycharmProjects\PythonProject\image_data.json/image_data.json")
IMG_OUT_DIR = Path(r"C:\Users\ilios\Desktop\images")
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)#kreirame ako ne postoi

MAX_IMAGES = 500
MAX_CARS_FOR_PROMPT = 10 # od 0 do 10 koli


VEHICLE_KEYWORDS = {"car","automobile","auto","sedan","suv","van","taxi"}
VEH_RE = re.compile(r"\b(" + "|".join(sorted(VEHICLE_KEYWORDS)) + r")s?\b", re.I)#bus|truck|car


def _build_prompt_variants(k: int):#prompting
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
def _encode_text_ensemble(model, device, clip, variants):
    tokens = clip.tokenize(variants).to(device)#site varijanti vo tekni gi pravime
    feats = model.encode_text(tokens)#gi zemame od clip tekstovite
    feats = F.normalize(feats, dim=-1)#normalizirame
    feat_mean = feats.mean(dim=0, keepdim=True)#srednata vrednost
    return F.normalize(feat_mean, dim=-1) #normalizirame srednata vrednost

@torch.no_grad()
def _build_txt_feats(model, device, clip, max_k: int):
    txt_feats = []
    for k in range(max_k + 1):#za site od 0 do 10 oravi razlicni varijacii na ednoto isto
        variants = _build_prompt_variants(k)#pravime razlicni varijacii od istoto
        txt_feats.append(_encode_text_ensemble(model, device, clip, variants))
    return torch.cat(txt_feats, dim=0)

def is_vehicle_label(label):
    return bool(label and VEH_RE.search(label))#za sek slucaj label da ne e None/" "


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_vehicle_images(objects_data):
    counts = defaultdict(int)#dict
    for entry in objects_data:
        img_id = entry["image_id"]
        for obj in entry.get("objects", []):#niz site
            for n in obj.get("names", []):
                if is_vehicle_label(n):
                  counts[img_id] += 1#broeime od zborovite site kolki se od nasha kroist
    return counts#iamge_id:vehicle_count




def download_image(url, out_path):
    if out_path.exists(): return True
    try:
        r = requests.get(url, timeout=10)#HTTP GET
        r.raise_for_status()#za ako e error
        out_path.write_bytes(r.content)#raw bites
        return True
    except:
        return False


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)#vanilla clip


def image_augmentations(pil):
    return [pil, ImageOps.mirror(pil)]#flipped verzija

@torch.no_grad()
def clip_estimate_cars(pil_img, max_k=10):
    ims = [preprocess(im) for im in image_augmentations(pil_img)]
    img_tensor = torch.stack(ims).to(device)
    img_feat = F.normalize(model.encode_image(img_tensor), dim=-1).mean(0, keepdim=True)
    img_feat = F.normalize(img_feat, dim=-1)

    txt_feat = _build_txt_feats(model, device, clip, max_k).to(device)

    logits = img_feat @ txt_feat.T#prozivod na dva vektori
    scale = torch.clamp(model.logit_scale.exp(), 1.0, 100.0)
    logits = logits * scale

    probs = logits.softmax(dim=1).squeeze(0)
    k_pred = probs.argmax().item()

    ent = -(probs * (probs + 1e-12).log()).sum().item()#ako entroipija e mala znaci povekje sme sigurni
    conf = 1 - ent / math.log(len(probs))

    return k_pred, probs.tolist(), conf


def main():
    objects_data = load_json(OBJECTS_JSON)
    image_meta = {int(d["image_id"]): d["url"] for d in load_json(IMAGES_JSON) if d.get("url")}
    veh_counts = filter_vehicle_images(objects_data)

    rows = []
    for img_id, vg_count in list(veh_counts.items())[:MAX_IMAGES]:
        url = image_meta.get(img_id)
        if not url: continue
        out_path = IMG_OUT_DIR / f"{img_id}.jpg"
        if not download_image(url, out_path): continue
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
    df.to_csv("vgc_clip_vehicle_counts.csv", index=False)


if __name__ == "__main__":
    main()

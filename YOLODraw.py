from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd


IMG_IN_DIR  = Path(r"C:\Users\ilios\Desktop\images")
IMG_OUT_DIR = Path(r"C:\Users\ilios\Desktop\images_annotated")
IMG_OUT_DIR.mkdir(parents=True, exist_ok=True)


VEHICLE_CLASS_IDS = {2}#{1, 2, 3, 5, 7, 8}
CLASS_NAMES = {0:"person",1:"bicycle",2:"car",3:"motorcycle",4:"airplane",5:"bus",6:"train",
               7:"truck",8:"boat"}


model = YOLO("yolov8n.pt")

def draw_boxes_opencv(in_path: Path, out_path: Path, conf_thres=0.25):
    """
    Detect vehicles in an image and draw green boxes. Returns number of vehicle detections.
    """
    img_bgr = cv2.imread(str(in_path))
    if img_bgr is None:
        print(f"Warning: couldn't open {in_path}")
        return 0


    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb, conf=conf_thres, verbose=False)[0]

    vehicle_count = 0
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        clses = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cls_id, c in zip(boxes, clses, confs):
            if cls_id in VEHICLE_CLASS_IDS:
                vehicle_count += 1

                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 200, 0), 2)
                label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {c:.2f}"

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_bgr, (x1, y1 - h - 6), (x1 + w + 4, y1), (0, 200, 0), -1)
                cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


    cv2.imwrite(str(out_path), img_bgr)
    return vehicle_count

def main():
    rows = []
    for img_path in sorted(IMG_IN_DIR.glob("*.jpg")):
        out_path = IMG_OUT_DIR / img_path.name
        count = draw_boxes_opencv(img_path, out_path, conf_thres=0.25)
        rows.append({
            "image_path": str(img_path),
            "annotated_path": str(out_path),
            "vehicle_detections": count
        })
        print(f"{img_path.name}: {count} car(s)")


    df = pd.DataFrame(rows)
    csv_out = IMG_OUT_DIR / "vehicle_detections_summary.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

if __name__ == "__main__":
    main()

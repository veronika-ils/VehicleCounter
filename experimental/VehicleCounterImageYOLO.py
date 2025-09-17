from ultralytics import YOLO
import cv2

model = YOLO('../yolov8n.pt')

img_path = r'C:\Users\ilios\Desktop\street.jpg'

results = model.predict(img_path,conf=0.25,verbose=False)
res = results[0]

names = res.names
car_count = 0
for box in res.boxes:
    cls_id = int(box.cls[0].item())
    label = names[cls_id]
    if label == 'car':
        car_count += 1

print(f"{car_count} cars detected")

annotated = res.plot()
out_path = r'C:\Users\ilios\Desktop\street_annotated.jpg'
cv2.imwrite(out_path,annotated)
print(f"Annotated image saved to: {out_path}")
# 🚦 Vehicle Counter with CLIP  

A computer vision project for detecting and **counting vehicles in images** using **OpenAI CLIP** features and a lightweight trained head.  
Instead of traditional real-time detection, this project uses **CLIP embeddings** to classify and estimate the number of vehicles in a given image.  

---

## 🚗 Features  
- Vehicle counting on **static images** (no real-time video)  
- Uses **CLIP model** to extract semantic features  
- Small supervised head trained on top of CLIP for vehicle count classification (0..N cars)  
- Works on custom datasets of labeled vehicle images  
- Exports results to CSV for analysis  

---

## 🛠 Technologies Used  
- **Python 3.9+**  
- **PyTorch** – CLIP backbone  
- **OpenAI CLIP** – for image embeddings  
- **scikit-learn** – lightweight classification head  
- **NumPy / Pandas** – data handling  
- **Pillow** – image preprocessing  



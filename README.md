# Vehicle Counting with CLIP  

This project explores the use of **CLIP (Contrastive Language–Image Pretraining)** models for experimental vehicle counting in images.  
Three different code approaches were implemented and compared, each attempting to estimate the number of cars present in images.  

The project is **experimental** – CLIP is not designed for object counting, so results are limited, but the analysis demonstrates different methods and how accuracy changes across approaches.  

---

## Project Structure  

- **Code 1 (`CLIPCounter.py`)** – Baseline CLIP approach  
  - Uses OpenAI’s CLIP ViT model directly.  
  - Compares text prompts (`"0 cars"`, `"1 car"`, …) against image embeddings.  
  - Produces stable but low-confidence predictions.  

- **Code 2 (`CLIPCounterHuggingFace.py`)** – Prompt engineering with thresholds  
  - Adds calibration and probability thresholds to filter uncertain results.  
  - Appears more confident in graphs but actually produces weaker accuracy.  

- **Code 3 (`CLIPCounterHFTrain.py`)** – CLIP + Logistic Regression Head  
  - Trains a small classifier (scikit-learn) on top of CLIP embeddings.  
  - Achieves the best overall performance among the three approaches.  
  - Still limited in precision due to CLIP’s design.  

- **Data**  
  - `train.csv` – training data with image paths and counts.  
  - `vgc_clip_vehicle_counts.csv` – results of code 1.  
  - `vgh_clip_vehicle_counts.csv` – results of code 2.
  - `vg_clip_vehicle_counts.csv` – results of code 3.   

---

## Running the Code  

### Requirements  
- Python 3.9+  
- PyTorch  
- Transformers (Hugging Face)  
- scikit-learn  
- pandas  
- joblib  

Install dependencies:  
```bash
pip install torch torchvision transformers scikit-learn pandas joblib

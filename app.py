import uvicorn
import torch
import time
import csv
import os
import logging
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import DeformableDetrForObjectDetection, DeformableDetrImageProcessor
from PIL import Image

torch.set_num_threads(os.cpu_count() or 4)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Init FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Label gesture
id2label = {
    0: 'ambil', 1: 'anjing', 2: 'apa', 3: 'atas', 4: 'ayah', 5: 'bantu', 6: 'bau', 7: 'bawah', 8: 'berdoa', 9: 'berhenti',
    10: 'berjalan', 11: 'bermain', 12: 'berpikir', 13: 'bertemu', 14: 'betul', 15: 'bingung', 16: 'bisindo', 17: 'bodoh',
    18: 'buat', 19: 'cantik', 20: 'cinta', 21: 'egois', 22: 'halo', 23: 'hati-hati', 24: 'hubungan', 25: 'ibu', 26: 'ingat',
    27: 'jangan', 28: 'janji', 29: 'kacamata', 30: 'kakak', 31: 'kakek', 32: 'kamu', 33: 'kanan', 34: 'keren', 35: 'kerja',
    36: 'kiri', 37: 'kuat', 38: 'kucing', 39: 'lagi', 40: 'laki-laki', 41: 'lelah', 42: 'lucu', 43: 'maaf', 44: 'makan',
    45: 'mandi', 46: 'marah', 47: 'mau', 48: 'melihat', 49: 'membaca', 50: 'memukul', 51: 'menangis', 52: 'mendengar',
    53: 'mendorong', 54: 'menggambar', 55: 'menuangkan', 56: 'menulis', 57: 'minta', 58: 'minum', 59: 'mobil', 60: 'motor',
    61: 'mulai', 62: 'naik', 63: 'nama', 64: 'nenek', 65: 'ngapain', 66: 'om', 67: 'pagi', 68: 'paham', 69: 'perempuan',
    70: 'perkenalkan', 71: 'ramah', 72: 'rumah', 73: 'sabar', 74: 'sahabat', 75: 'sakit', 76: 'salah', 77: 'sama-sama',
    78: 'sapi', 79: 'saudara', 80: 'saya', 81: 'sedih', 82: 'sedikit', 83: 'semangat', 84: 'senang', 85: 'siapa', 86: 'suka',
    87: 'takut', 88: 'telepon', 89: 'tempat', 90: 'terima kasih', 91: 'terlambat', 92: 'tidur', 93: 'tinggal', 94: 'tolong',
    95: 'tunggu', 96: 'uang', 97: 'untuk', 98: 'waktu', 99: 'ya'
}

# Preprocess image
def preprocess_cv2(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gagal membaca gambar, format tidak didukung!")
    img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    return img_pil

# Load model
MODEL_PATH = "./model_30epoch"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Loading model from {MODEL_PATH} to {DEVICE}")

model = DeformableDetrForObjectDetection.from_pretrained(
    MODEL_PATH, local_files_only=True
).to(DEVICE).eval()

image_processor = DeformableDetrImageProcessor.from_pretrained(
    MODEL_PATH, local_files_only=True
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        image_bytes = await file.read()
        image = preprocess_cv2(image_bytes)

        # Inference
        inputs = image_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            target_size = torch.tensor([image.size[::-1]]).to(DEVICE)
            results = image_processor.post_process_object_detection(
                outputs, threshold=0.2, target_sizes=target_size
            )[0]

        all_detections = [
            {
                "bbox": [float(x) for x in box],
                "label": id2label.get(int(class_id), str(class_id)),
                "confidence": float(score)
            }
            for box, score, class_id in zip(
                results["boxes"], results["scores"], results["labels"]
            )
        ]

        # Ambil hanya deteksi dengan skor tertinggi
        top_detection = max(all_detections, key=lambda d: d["confidence"], default=None)
        response = {
            "detections": [top_detection] if top_detection else []
        }

        latency = time.time() - start_time
        logger.info(f"Detections: {response['detections']} | Latency: {latency:.3f}s")

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"detections": []}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
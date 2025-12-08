import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os

# --- KONFIGURASI ---
NAMA_DATASET = 'X_Plate_2.pickle' # Ganti sesuai file yang kamu punya
PATH_DATASET = 'pickle/' + NAMA_DATASET

def load_data(filepath):
    print(f"📂 Loading dataset {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Dataset Kaggle ini isinya Dictionary: {'images': ..., 'labels': ...}
    # Atau kadang langsung array. Kita coba handle dua-duanya.
    
    images = []
    labels = []

    if isinstance(data, dict):
        # Struktur umum dataset ini di Kaggle
        # Biasanya ada key 'images' dan 'labels' atau dipisah per folder
        # TAPI, dataset Ishihara ini agak unik strukturnya.
        # Jika kamu punya file X_Plate_2.pickle, biasanya dia cuma gambar (X).
        # Kamu butuh file LABEL-nya juga (biasanya y_Plate_2.pickle atau labels).
        pass 
        
    # STOP! Karena dataset Kaggle "Ishihara Like MNIST" ini kadang file labelnya terpisah
    # (y_plate_x.pickle), pastikan kamu punya file labelnya.
    # KALAU TIDAK ADA LABEL, KITA TIDAK BISA LATIH CLASSIFIER.
    
    # --- ALTERNATIF JIKA GAK ADA LABEL ---
    # Kita pakai model MNIST standar (hitam putih) lalu kita latih dia
    # untuk menebak Ishihara. Tapi ini susah.
    
    # SAYA ASUMSIKAN KAMU PUNYA FILE LABELNYA (y_Plate_X.pickle)
    # Kalau di Kaggle biasanya satu folder ada X (gambar) dan y (label).
    return None

# --- EITS, TUNGGU SEBENTAR ---
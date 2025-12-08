import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# ================= KONFIGURASI =================
# Ganti nama file sesuai model yang mau dites
MODEL_PATH = 'generator_X_Plate_2.h5' 

# Arahkan ke file pickle TEST (Bukan Train!)
# Sesuai screenshot pertama kamu, biasanya ada di folder Test_images
PATH_TEST_PICKLE = 'X_PlateTest_2.pickle' 

# Mau menampilkan berapa gambar?
JUMLAH_SAMPEL = 8 
# ===============================================

def load_real_test_sample(filepath, n_samples):
    """Fungsi bongkar pickle yang lebih sakti (mirip script training)"""
    if not os.path.exists(filepath):
        print(f"⚠️ Warning: File {filepath} tidak ditemukan. Hanya akan menampilkan gambar Generator.")
        return None

    print(f"   📂 Sedang membongkar pickle: {filepath}...")
    with open(filepath, 'rb') as f:
        data_raw = pickle.load(f)

    collected_images = []

    # --- LOGIKA BONGKAR (Diambil dari script trainingmu) ---
    # 1. Jika Dictionary
    if isinstance(data_raw, dict):
        if 'files_dic' in data_raw:
            data_raw = data_raw['files_dic']
        
        # Ambil semua value dari dict
        for key, val in data_raw.items():
            if isinstance(val, list): val = np.array(val)
            
            # Masukkan ke list penampung
            if len(val.shape) == 4: # Batch images
                for img in val: collected_images.append(img)
            elif len(val.shape) == 3: # Single image
                collected_images.append(val)

    # 2. Jika List/Array langsung
    elif isinstance(data_raw, (list, np.ndarray)):
        data_raw = np.array(data_raw)
        for img in data_raw:
            collected_images.append(img)

    # Convert list ke numpy array besar
    data_clean = np.array(collected_images)
    print(f"   ✅ Berhasil menemukan {len(data_clean)} gambar asli.")

    # Safety check kalau data kosong
    if len(data_clean) == 0: return None

    # Ambil sampel acak
    if len(data_clean) < n_samples: n_samples = len(data_clean)
    indices = np.random.choice(len(data_clean), n_samples, replace=False)
    sample_data = data_clean[indices]

    processed = []
    for img in sample_data:
        # Transpose kalau channel ada di depan (3, 64, 64) -> (64, 64, 3)
        if img.shape[0] == 3: 
            img = np.transpose(img, (1, 2, 0))
            
        # Resize ke 64x64 biar adil bandinginnya
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_resized = tf.image.resize(img_tensor, (64, 64))
        
        # Normalisasi ke 0-1 untuk display
        # Asumsi data asli mungkin 0-255 atau 0-1
        if tf.reduce_max(img_resized) > 1.0:
            img_resized = img_resized / 255.0
            
        processed.append(img_resized.numpy()) 
    
    return np.array(processed)

def main():
    print(f"📂 Memuat Model: {MODEL_PATH}...")
    try:
        generator = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Gagal load model: {e}")
        return

    # 1. GENERATE GAMBAR PALSU (DARI GENERATOR)
    print("🎨 Generator sedang menggambar...")
    noise = tf.random.normal([JUMLAH_SAMPEL, 100]) # 100 adalah NOISE_DIM
    fake_images = generator(noise, training=False)

    # DENORMALISASI PENTING! (Agar tidak hitam/gelap)
    # Karena output generator range [-1, 1], kita ubah ke [0, 1]
    fake_images = (fake_images + 1) / 2.0

    # 2. AMBIL GAMBAR ASLI (DARI TEST SET)
    real_images = load_real_test_sample(PATH_TEST_PICKLE, JUMLAH_SAMPEL)

    # 3. TAMPILKAN PERBANDINGAN
    print("🖥️ Menampilkan hasil...")
    plt.figure(figsize=(15, 5))

    # Baris 1: Gambar Buatan AI
    for i in range(JUMLAH_SAMPEL):
        plt.subplot(2, JUMLAH_SAMPEL, i+1)
        plt.imshow(fake_images[i])
        if i == 0: plt.title("Generator (Palsu)", fontsize=14, color='red')
        plt.axis('off')

    # Baris 2: Gambar Asli (Jika ada file testnya)
    if real_images is not None:
        for i in range(JUMLAH_SAMPEL):
            plt.subplot(2, JUMLAH_SAMPEL, JUMLAH_SAMPEL + i + 1)
            plt.imshow(real_images[i])
            if i == 0: plt.title("Dataset Asli (Test)", fontsize=14, color='green')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
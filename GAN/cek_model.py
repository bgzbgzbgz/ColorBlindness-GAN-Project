import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

nama_file = 'Train_images.pickle'
print(f"🕵️‍♂️ SEDANG MENJALANKAN DETEKTIF DATASET PADA: {nama_file}...")

try:
    with open(nama_file, 'rb') as f:
        data_raw = pickle.load(f)

    train_images = None
    
    # 1. CEK LEVEL PERTAMA
    print(f"\n📂 Tipe Data Terluar: {type(data_raw)}")
    if isinstance(data_raw, dict):
        print(f"🔑 Kunci Level 1: {list(data_raw.keys())}")
        
        # Masuk ke 'files_dic' kalau ada
        if 'files_dic' in data_raw:
            print("   ✅ Masuk ke dalam 'files_dic'...")
            isi_dalam = data_raw['files_dic']
            
            # 2. CEK ISI FILES_DIC (LEVEL KEDUA)
            print(f"   🔑 Kunci Level 2 (Isi files_dic): {list(isi_dalam.keys())}")
            
            for key in isi_dalam.keys():
                item = isi_dalam[key]
                # PAKSA JADI ARRAY BIAR KETAHUAN SHAPE-NYA
                item_arr = np.array(item)
                print(f"      - Cek Key '{key}': Shape = {item_arr.shape}")
                
                # Kalau dimensinya 4 (Jumlah, CH, H, W) atau (Jumlah, H, W, CH)
                if len(item_arr.shape) == 4:
                    train_images = item_arr
                    print(f"      🎉 KETEMU! Dataset ada di kunci: '{key}'")
                    break
        else:
            # Kalau gak ada files_dic, mungkin datanya langsung di level 1
            for key in data_raw.keys():
                item = np.array(data_raw[key])
                if len(item.shape) == 4:
                    train_images = item
                    break

    else:
        # Kalau bukan dictionary (langsung list/array)
        train_images = np.array(data_raw)

    # 3. VERIFIKASI AKHIR
    if train_images is None:
        raise ValueError("❌ TETAP GAGAL MENEMUKAN ARRAY GAMBAR 4 DIMENSI!")

    print(f"\n✅ Dataset Loaded. Shape Asli: {train_images.shape}")

    # 4. FIX FORMAT (PyTorch -> TensorFlow)
    # Jika channel (3) ada di depan (index 1), pindah ke belakang
    if train_images.shape[1] == 3:
        print("⚠️ Terdeteksi Channel First (PyTorch). Melakukan Transpose...")
        train_images = train_images.transpose(0, 2, 3, 1)

    # 5. TAMPILKAN GALERI
    print("🖼️ Menyiapkan Galeri...")
    
    # Normalisasi Visual (0-1)
    if train_images.min() < 0 or train_images.max() > 1:
        train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())

    plt.figure(figsize=(12, 12))
    indices = random.sample(range(len(train_images)), 25)
    
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i+1)
        plt.imshow(train_images[idx])
        plt.axis('off')
        plt.title(f"Img {idx}")

    plt.tight_layout()
    plt.show()
    print("✅ Selesai menampilkan gambar.")

except Exception as e:
    print("\n❌ TERJADI ERROR:")
    print(e)
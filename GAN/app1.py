import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ishihara GAN Generator", page_icon="🎨", layout="centered")

st.title("🎨 AI Pembuat Tes Buta Warna")
st.markdown("""
Aplikasi ini menggunakan **Deep Convolutional GAN (DCGAN)** untuk menghasilkan pola Ishihara secara sintetis.
Sistem akan secara **otomatis & acak** memilih generator (Angka 2, 5, atau 8) untuk setiap gambar.
""")

# --- FUNGSI LOAD SEMUA MODEL SEKALIGUS ---
# Kita cache supaya loading cuma sekali di awal, jadi pas klik tombol rasanya cepet
@st.cache_resource
def load_all_models():
    models = {}
    # Daftar file model yang kamu punya
    # Pastikan nama file ini sesuai dengan yang ada di folder kamu
    daftar_file = {
        "Angka 2": "generator_X_Plate_2.h5",
        "Angka 5": "generator_X_Plate_5.h5",
        "Angka 8": "generator_X_Plate_8.h5"
    }
    
    loaded_count = 0
    for label, path in daftar_file.items():
        if os.path.exists(path):
            try:
                models[label] = tf.keras.models.load_model(path)
                loaded_count += 1
            except:
                pass # Skip kalau file error
        else:
            # Opsional: Print di terminal kalau file gak ketemu (buat debug)
            print(f"⚠️ File model tidak ditemukan: {path}")

    return models

# --- LOAD MODEL DI BACKGROUND ---
with st.spinner("Sedang menyiapkan otak AI..."):
    available_models = load_all_models()

# Cek apakah ada model yang berhasil di-load
if not available_models:
    st.error("❌ Tidak ada model .h5 yang ditemukan! Pastikan file 'generator_X_Plate_*.h5' ada di folder yang sama.")
    st.stop()

# --- SIDEBAR (Sederhana Saja) ---
st.sidebar.header("🎛️ Pengaturan")
st.sidebar.success(f"✅ {len(available_models)} Model Siap Digunakan")
jumlah_gambar = st.sidebar.slider("Mau bikin berapa gambar?", 1, 9, 3)
tombol = st.sidebar.button("✨ GENERATE SEKARANG", type="primary")

# --- AREA UTAMA ---
if tombol:
    st.write("---")
    
    # Siapkan layout grid (maksimal 3 kolom)
    cols = st.columns(3)
    
    # Progress bar biar kerasa canggih
    my_bar = st.progress(0)

    for i in range(jumlah_gambar):
        # 1. Pilih Model secara ACAK untuk gambar ini
        label_terpilih = random.choice(list(available_models.keys()))
        model_terpilih = available_models[label_terpilih]

        # 2. Generate Gambar
        noise_dim = 100
        noise = tf.random.normal([1, noise_dim])
        
        generated_image = model_terpilih(noise, training=False)
        
        # 3. Denormalisasi [-1, 1] -> [0, 1]
        img_display = (generated_image[0] + 1) / 2.0
        img_display = img_display.numpy()

        # 4. Tampilkan di Streamlit
        # Pakai logika modulo untuk nentuin masuk kolom mana
        with cols[i % 3]:
            fig, ax = plt.subplots()
            ax.imshow(img_display)
            ax.axis('off')
            ax.set_title(f"Hasil: {label_terpilih}", fontsize=10, color='green')
            st.pyplot(fig)
            plt.close(fig) # Hemat memori
        
        # Update progress bar
        my_bar.progress((i + 1) / jumlah_gambar)

    st.success("🎉 Selesai! Coba klik tombol lagi untuk hasil berbeda.")

else:
    # Tampilan awal saat belum diklik
    st.info("👈 Atur jumlah gambar di menu kiri, lalu tekan tombol Generate.")
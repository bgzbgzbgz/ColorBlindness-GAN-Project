import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ishihara GAN Generator", page_icon="🎨")

st.title("🎨 AI Pembuat Tes Buta Warna")
st.write("Generator Ishihara menggunakan Deep Convolutional GAN (DCGAN).")

# --- FUNGSI LOAD MODEL (Dicache biar cepet) ---
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

# --- SIDEBAR: PILIH MODEL ---
st.sidebar.header("🎛️ Kontrol Panel")

# Nanti kalau kamu punya model Plate 5 & 8, tambah di sini
model_options = {
    "Angka 2 (Plate 2)": "generator_X_Plate_2.h5",
    "Angka 5 (Plate 5)": "generator_X_Plate_5.h5", # Aktifkan nanti
    "Angka 8 (Plate 8)": "generator_X_Plate_8.h5", # Aktifkan nanti
}

pilihan_model = st.sidebar.selectbox("Pilih Angka yang mau dibuat:", list(model_options.keys()))
file_path = model_options[pilihan_model]

# Load Model
try:
    model = load_model(file_path)
    st.sidebar.success(f"✅ Model {pilihan_model} dimuat!")
except Exception as e:
    st.sidebar.error(f"❌ Model belum ada: {file_path}")
    st.sidebar.info("Pastikan file .h5 sudah didownload dari server.")
    st.stop()

# --- TOMBOL GENERATE ---
col1, col2 = st.columns([1, 2])

with col1:
    jumlah_gambar = st.slider("Jumlah Gambar", 1, 9, 1)
    tombol = st.button("✨ Generate Baru")

# --- LOGIKA GENERATE ---
if tombol:
    noise_dim = 100
    noise = tf.random.normal([jumlah_gambar, noise_dim])
    
    with st.spinner("Sedang melukis titik-titik..."):
        generated_images = model(noise, training=False)
        
        # Rescale dari [-1, 1] ke [0, 1]
        generated_images = (generated_images + 1) / 2.0
        generated_images = generated_images.numpy()

    # Tampilkan Gambar
    st.subheader(f"Hasil Generasi: {pilihan_model}")
    
    # Atur Grid Tampilan
    cols = st.columns(3) # 3 Kolom
    for i in range(jumlah_gambar):
        with cols[i % 3]:
            fig, ax = plt.subplots()
            ax.imshow(generated_images[i])
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig)

else:
    st.info("Tekan tombol 'Generate Baru' untuk membuat gambar Ishihara.")
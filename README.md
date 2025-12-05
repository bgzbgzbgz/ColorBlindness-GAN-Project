# 👁️ ISHICARE: DCGAN Training Module

**ISHICARE** adalah proyek pengembangan model **Deep Convolutional Generative Adversarial Networks (DCGAN)** untuk menghasilkan pelat tes buta warna (Ishihara Plates) secara sintetis.

Repositori ini berisi kode sumber untuk melatih model generator agar dapat menciptakan variasi gambar tes buta warna baru.

---

## 👥 Tim Pengembang

**Dosen Pembimbing:**
Sritrusta Sukaridhoto ST, Ph.D

**Anggota Tim (Mahasiswa PENS):**
* **Muzaqi Indra Al Azhar** (5323600039)
* **Fadel Ilham Dzulkarnain** (5323600051)
* **Sufa Delila** (5323600053)
* **Bagas Andi Kurniawan** (5323600055)

---

## ⚙️ Persiapan (Setup)

Sebelum melatih model, pastikan Anda telah menyiapkan lingkungan kerja berikut:

### 1. Install Library
Gunakan perintah berikut untuk menginstal dependensi yang dibutuhkan:
```bash
pip install tensorflow numpy matplotlib
```
### 2. Siapkan Dataset
Karena ukuran file dataset besar, file dataset tidak disertakan di dalam repositori ini.

a. Download dataset "Ishihara Like MNIST" dari Kaggle: Link Dataset

b. Buat folder baru bernama pickle di dalam direktori GAN.

c. Masukkan file .pickle yang didownload (misal: X_Plate_2.pickle) ke dalam folder tersebut.

d. Struktur Folder yang Benar:
```text
ISHICARE/
└── GAN/
    ├── pickle/                <-- Buat folder ini
    │   └── X_Plate_2.pickle   <-- Simpan dataset disini
    ├── model/                 <-- Folder output model (.h5)
    └── train_gan_modular.py   <-- Script training
```
### 3. Cara Menjalankan Training
Ikuti langkah berikut untuk memulai proses pelatihan model:

a. Buka Terminal / CMD.

b. Masuk ke direktori GAN:
```bash
pip install tensorflow numpy matplotlib
```
c. Jalankan Script Training:
```bash
pip install tensorflow numpy matplotlib
```
### 4. Konfigurasi Training
Anda dapat mengubah pengaturan training langsung di dalam file train_gan_modular.py pada bagian AREA KONFIGURASI (Baris awal script)
| Variabel | Deskripsi |
| :--- | :--- |
| `NAMA_FILE_DATASET` | Nama file dataset target (misal: `'X_Plate_2.pickle'`). |
| `EPOCHS` | Jumlah perulangan training (Default: `100`). |
| `BATCH_SIZE` | Jumlah data per proses. Turunkan jika RAM/VRAM penuh (Default: `8` atau `16`). |


### 5. Output
Setelah training selesai, hasil akan disimpan otomatis:

a. Model: File .h5 akan tersimpan di folder output_lokal atau root folder sesuai konfigurasi.

b. Preview: Gambar perkembangan training tersimpan di folder hasil_training_[nama_dataset].

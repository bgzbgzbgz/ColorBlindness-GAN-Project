import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
import gc # Garbage Collector untuk bersih-bersih RAM

# ==================================================================
# 🔧 AREA KONFIGURASI (SUDAH DIOPTIMALKAN UNTUK SERVER ANDA)
# ==================================================================

# 1. Nama File Pickle (Pastikan file ini ada di folder pickle/)
NAMA_FILE_DATASET = 'X_Plate_3.pickle' 

# 2. Lokasi folder dataset
PATH_FOLDER_DATASET = './pickle'

# 3. Setting Training
EPOCHS = 100           
BATCH_SIZE = 8         # <--- SUDAH DITURUNKAN KE 8 (Agar GPU Kuat)
BUFFER_SIZE = 500      # <--- DITURUNKAN KE 500 (Agar RAM Server Aman)
NOISE_DIM = 100        
TARGET_SIZE = (112, 112) # Tetap Resolusi Tinggi

# ==================================================================

def setup_environment():
    # Folder output otomatis sesuai nama dataset
    nama_bersih = NAMA_FILE_DATASET.replace('.pickle', '').replace('.pkl', '')
    output_dir = f"hasil_training_{nama_bersih}"
    checkpoint_dir = os.path.join(output_dir, 'training_checkpoints')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        
    print(f"📁 Output gambar akan disimpan di: {output_dir}")
    return output_dir, checkpoint_dir

def load_dataset(filepath):
    print(f"🚀 Memuat dataset: {filepath}")
    
    # Load pickle pelan-pelan
    with open(filepath, 'rb') as f:
        data_raw = pickle.load(f)

    collected_images = [] 
    # Kita resize dulu ke target sementara untuk hemat RAM saat append
    # Nanti di preprocess baru di finalisasi
    
    print("   📉 Memproses gambar satu per satu (Hemat RAM)...")

    # --- FUNGSI BANTUAN ---
    def proses_dan_simpan(img_array):
        # 1. Fix Channel First (PyTorch) -> Last (TF)
        if img_array.shape[0] == 3: # Cek dimensi awal (3, 112, 112)
             img_array = np.transpose(img_array, (1, 2, 0))
        
        # 2. Resize Menggunakan TensorFlow (Biar cepat & rapi)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Resize langsung ke target size di sini biar array numpy-nya kecil
        img_resized = tf.image.resize(img_tensor, TARGET_SIZE).numpy()
        
        # 3. Masukkan ke list
        collected_images.append(img_resized)

    # --- LOGIKA BONGKAR DATA ---
    if isinstance(data_raw, dict):
        if 'files_dic' in data_raw:
            data_raw = data_raw['files_dic']
        
        total_items = len(data_raw)
        processed = 0
        
        for key, val in data_raw.items():
            if isinstance(val, list): val = np.array(val)
            
            if hasattr(val, 'shape'):
                if len(val.shape) == 4:
                    for img in val: proses_dan_simpan(img)
                elif len(val.shape) == 3:
                    proses_dan_simpan(val)
            
            processed += 1
            if processed % 2000 == 0:
                print(f"      -> Sudah memproses {processed}/{total_items}...")
                    
    elif isinstance(data_raw, list) or isinstance(data_raw, np.ndarray):
        data_raw = np.array(data_raw)
        if len(data_raw.shape) == 4:
            for img in data_raw:
                proses_dan_simpan(img)

    # --- HAPUS DATA MENTAH DARI RAM ---
    del data_raw
    gc.collect()

    # --- GABUNGKAN HASIL ---
    if len(collected_images) == 0:
        raise ValueError("❌ Gagal menemukan array gambar!")

    print(f"   📦 Menggabungkan {len(collected_images)} gambar...")
    train_images = np.stack(collected_images, axis=0)

    # --- [PENTING] MANUAL SHUFFLE UNTUK BUFFER KECIL ---
    print("   🎲 Mengacak data secara manual (Manual Shuffle)...")
    np.random.shuffle(train_images)
    # ---------------------------------------------------

    print(f"📊 Shape Final (Siap Training): {train_images.shape}")
    return train_images

def preprocess_data(images):
    print("   📉 Memulai Preprocessing & Normalisasi...")
    # Pastikan tipe float32
    if images.dtype != 'float32':
        images = images.astype('float32')

    # --- NORMALISASI CERDAS ---
    # GAN butuh range [-1, 1]
    if np.max(images) > 1.0:
        print("      -> Normalisasi dari [0, 255] ke [-1, 1]")
        images = (images - 127.5) / 127.5
    else:
        print("      -> Normalisasi dari [0, 1] ke [-1, 1]")
        images = (images * 2) - 1

    # Masukkan ke tf.data.Dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(images)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset

# --- MODEL GENERATOR ---
def make_generator_model():
    model = tf.keras.Sequential()
    
    # 1. Pondasi Awal (Start dari 7x7 atau 8x8 tergantung input)
    # Kita mulai dari 7x7 agar naik ke 112 (7 -> 14 -> 28 -> 56 -> 112)
    start_dim = TARGET_SIZE[0] // 16 # 112 / 16 = 7
    
    model.add(layers.Dense(start_dim * start_dim * 512, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((start_dim, start_dim, 512)))

    # Upsample 1 (7x7 -> 14x14)
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample 2 (14x14 -> 28x28)
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample 3 (28x28 -> 56x56)
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsample 4 (56x56 -> 112x112)
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# --- MODEL DISCRIMINATOR ---
def make_discriminator_model():
    model = tf.keras.Sequential()
    img_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
    
    # 112x112 -> 56x56
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # 56x56 -> 28x28
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # 28x28 -> 14x14
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    # 14x14 -> 7x7
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# --- LOSS & OPTIMIZER ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # Label smoothing (0.9)
    real_loss = cross_entropy(tf.ones_like(real_output) * 0.9, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# --- UTAMA ---
if __name__ == '__main__':
    # 1. Setup
    full_path = os.path.join(PATH_FOLDER_DATASET, NAMA_FILE_DATASET)
    output_folder, checkpoint_prefix = setup_environment()
    
    # 2. Load Data & Clean Memory
    try:
        raw_images = load_dataset(full_path)
        train_dataset = preprocess_data(raw_images)
        
        # Hapus raw images dari RAM karena sudah masuk dataset
        del raw_images
        gc.collect()
        print("   🧹 RAM dibersihkan sebelum training dimulai.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        exit()

    # 3. Init Model
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    seed = tf.random.normal([16, NOISE_DIM])

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def generate_and_save_images(model, epoch, test_input):
        predictions = model(test_input, training=False)
        predictions = (predictions + 1) / 2.0

        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i])
            plt.axis('off')

        save_path = os.path.join(output_folder, 'image_at_epoch_{:04d}.png'.format(epoch))
        plt.savefig(save_path)
        plt.close()

    # --- LOOP TRAINING ---
    print(f"\n🔥 MEMULAI TRAINING ({EPOCHS} Epochs)...")
    
    for epoch in range(EPOCHS):
        start = time.time()
        g_loss_avg = 0
        d_loss_avg = 0
        steps = 0

        for image_batch in train_dataset:
            g, d = train_step(image_batch)
            g_loss_avg += g
            d_loss_avg += d
            steps += 1

        print(f'Epoch {epoch + 1}/{EPOCHS} | G_Loss: {g_loss_avg/steps:.4f} | D_Loss: {d_loss_avg/steps:.4f} | Time: {time.time()-start:.1f}s')

        if (epoch + 1) == 1 or (epoch + 1) <= 10 or (epoch + 1) % 5 == 0:
            generate_and_save_images(generator, epoch + 1, seed)
            
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=os.path.join(checkpoint_prefix, "ckpt"))

    final_model_path = os.path.join(output_folder, f"generator_{NAMA_FILE_DATASET.replace('.pickle', '')}.h5")
    generator.save(final_model_path)
    print(f"\n✅ TRAINING SELESAI! Model disimpan di: {final_model_path}")
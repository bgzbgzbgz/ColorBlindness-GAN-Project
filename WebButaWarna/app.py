import os
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2 
from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import time

app = Flask(__name__)
app.secret_key = 'rahasia_banget'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///rumah_sakit.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'static/generated'
if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)

# --- 1. LOAD MODEL GAN ---
print("⏳ Loading Model GAN...")
models_gan = {}
try:
    models_gan['Plate_2'] = tf.keras.models.load_model('model/generator_X_Plate_2.h5')
    models_gan['Plate_5'] = tf.keras.models.load_model('model/generator_X_Plate_5.h5')
    models_gan['Plate_8'] = tf.keras.models.load_model('model/generator_X_Plate_8.h5')
    print("✅ GAN Loaded.")
except Exception as e:
    print(f"❌ Error Load GAN: {e}")

# --- 2. LOAD/BUAT MODEL JURI (CLASSIFIER) ---
print("⏳ Menyiapkan AI Juri (Classifier)...")

def create_judge_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

classifier_path = 'model/mnist_judge.h5'
if os.path.exists(classifier_path):
    judge_model = tf.keras.models.load_model(classifier_path)
    print("✅ AI Juri Siap (Loaded).")
else:
    print("⚠️ Model Juri belum ada, mendownload dataset MNIST & Training kilat...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    judge_model = create_judge_model()
    judge_model.fit(x_train, y_train, epochs=3) 
    judge_model.save(classifier_path)
    print("✅ AI Juri Selesai Training & Disimpan.")

# --- DATABASE MODEL ---
class Pasien(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nomor_id = db.Column(db.String(50), nullable=False)
    nama = db.Column(db.String(100), nullable=False)
    usia = db.Column(db.String(10), nullable=False)
    gender = db.Column(db.String(20), nullable=False)
    riwayat_mata = db.Column(db.String(200))
    skor = db.Column(db.String(100))
    tanggal = db.Column(db.DateTime, default=datetime.utcnow)

# --- LOGIC UTAMA ---
def generate_and_predict():
    keys = list(models_gan.keys())
    if not keys: return None, None
    selected_gan = models_gan[np.random.choice(keys)]

    noise = tf.random.normal([1, 100])
    generated_img = selected_gan(noise, training=False)
    
    img_display = (generated_img[0].numpy() + 1) / 2.0
    img_display = np.clip(img_display, 0, 1)

    img_resized = tf.image.resize(generated_img, (28, 28))
    img_resized = (img_resized[0].numpy() + 1) / 2.0 
    img_gray = np.mean(img_resized, axis=2) 
    p2, p98 = np.percentile(img_gray, (2, 98))
    img_gray = np.clip((img_gray - p2) / (p98 - p2), 0, 1)
    img_input_judge = img_gray.reshape(1, 28, 28, 1)

    prediksi = judge_model.predict(img_input_judge)
    angka_tebakan = np.argmax(prediksi)
    confidence = np.max(prediksi) * 100
    
    filename = f"test_{int(time.time())}.png"
    plt.imsave(os.path.join(UPLOAD_FOLDER, filename), img_display)
    
    # 🔥 DEBUG TERMINAL DINYALAKAN LAGI 🔥
    print(f"🤖 [JURI] Saya melihat angka: {angka_tebakan} (Yakin: {confidence:.1f}%)")
    
    return f"generated/{filename}", str(angka_tebakan)

# --- ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['nomor_id'] = request.form['nomor_id']
        session['nama'] = request.form['nama']
        session['usia'] = request.form['usia']
        session['gender'] = request.form['gender']
        session['riwayat'] = request.form['riwayat']
        
        session['skor_benar'] = 0
        session['soal_ke'] = 1
        session['max_soal'] = 5 
        
        return redirect(url_for('mulai_tes'))
    return render_template('index.html')

# --- ROUTE HALAMAN KREDIT (BARU) ---
@app.route('/kredit')
def kredit():
    return render_template('kredit.html')

# --- ROUTE LOGIN ADMIN ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin123':
            session['admin_logged_in'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('login.html', error="Password Salah!")
    return render_template('login.html')

@app.route('/admin')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect(url_for('login'))
    semua_pasien = Pasien.query.order_by(Pasien.tanggal.desc()).all()
    return render_template('admin.html', data=semua_pasien)

@app.route('/hapus/<int:id>')
def hapus_pasien(id):
    if not session.get('admin_logged_in'): return redirect(url_for('login'))
    p = Pasien.query.get(id)
    if p:
        db.session.delete(p)
        db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/logout')
def logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('home'))

@app.route('/tes', methods=['GET', 'POST'])
def mulai_tes():
    if 'soal_ke' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        input_user = request.form.get('tebakan', '')
        kunci_jawaban_asli = request.form.get('jawaban_asli_token', '')
        
        jawaban_user = str(input_user).strip()
        jawaban_ai   = str(kunci_jawaban_asli).strip()
        
        # 🔥 DEBUG LOGIC BENAR/SALAH DINYALAKAN LAGI 🔥
        if jawaban_user == jawaban_ai:
            session['skor_benar'] += 1
            print(f"✅ [User] Benar! Jawab {jawaban_user} (AI: {jawaban_ai})")
        else:
            print(f"❌ [User] Salah... Jawab {jawaban_user} (AI: {jawaban_ai})")

        if session['soal_ke'] >= session['max_soal']:
            return redirect(url_for('selesai'))
        else:
            session['soal_ke'] += 1
            return redirect(url_for('mulai_tes'))

    img_url, kunci_raw = generate_and_predict()
    kunci_str = str(kunci_raw).strip()
    
    return render_template('quiz.html', 
                           gambar=url_for('static', filename=img_url), 
                           bocoran=kunci_str,
                           no_soal=session['soal_ke'])

@app.route('/selesai')
def selesai():
    skor = session.get('skor_benar', 0)
    total = session.get('max_soal', 5)
    
    if skor >= (total - 1):
        status_akhir = "Normal"
    else:
        status_akhir = "Indikasi Buta Warna"
        
    data_baru = Pasien(
        nomor_id=session['nomor_id'], 
        nama=session['nama'],
        usia=session['usia'],
        gender=session['gender'],
        riwayat_mata=session['riwayat'],
        skor=f"{status_akhir} ({skor}/{total} Benar)"
    )
    db.session.add(data_baru)
    db.session.commit()
    return render_template('result.html', status=status_akhir, nama=session['nama'], skor=skor, total=total)

@app.route('/hasil/<status>')
def hasil(status):
    return render_template('result.html', status=status, nama=session.get('nama', 'User'))

if __name__ == "__main__":
    with app.app_context(): db.create_all()
    app.run(debug=True)
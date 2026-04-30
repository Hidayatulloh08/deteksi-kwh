import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

FILE = "data.csv"
MODEL = "model_lstm.keras"
SCALER = "scaler.save"
WINDOW = 14

# ===== LOAD DATA =====
if not os.path.exists(FILE):
    print("❌ data.csv tidak ditemukan")
    exit()

data = pd.read_csv(FILE, on_bad_lines='skip')

if 'biaya' not in data.columns:
    print("❌ Kolom biaya tidak ada")
    exit()

# ===== SMOOTHING =====
data['biaya'] = data['biaya'].rolling(window=3).mean()
data = data.dropna()

# ===== TAMBAH FITUR =====
if 'timestamp' in data.columns:
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
else:
    data['timestamp'] = pd.Timestamp.now()

data['hour'] = data['timestamp'].dt.hour.fillna(0)
data['day'] = data['timestamp'].dt.day.fillna(0)

if len(data) < WINDOW:
    print("❌ Data kurang dari", WINDOW)
    exit()

# ===== DATASET MULTI FEATURE =====
dataset = data[['biaya', 'hour', 'day']].values

# ===== LOAD SCALER =====
if not os.path.exists(SCALER):
    print("❌ scaler.save tidak ditemukan")
    exit()

scaler = joblib.load(SCALER)
scaled = scaler.transform(dataset)

# ===== LOAD MODEL =====
try:
    model = load_model(MODEL)
except:
    print("❌ Model tidak ditemukan")
    exit()

# ===== PREDIKSI =====
last = scaled[-WINDOW:]
X_test = np.array([last])

pred = model.predict(X_test, verbose=0)

# ambil hanya kolom biaya
pred_full = np.zeros((1, 3))
pred_full[0, 0] = pred[0][0]

pred_real = scaler.inverse_transform(pred_full)

hasil = float(pred_real[0][0])

# ===== ANALISIS =====
rata = data['biaya'].mean()
estimasi = rata * 30

# ===== ERROR =====
actual = data['biaya'].iloc[-1]
mae = abs(actual - hasil)

mape = 0
if actual != 0:
    mape = (mae / actual) * 100

# ===== CONFIDENCE =====
confidence = 100 - min(100, int(mape))

# ===== OUTPUT =====
print("\n===== HASIL PREDIKSI =====")
print("💰 Prediksi besok: Rp", int(hasil))
print("📊 Rata harian:", int(rata))
print("📈 Estimasi bulanan:", int(estimasi))

print("\n===== EVALUASI =====")
print("📉 MAE:", round(mae, 2))
print("📊 MAPE:", round(mape, 2), "%")
print("📊 Confidence:", confidence, "%")

# ===== DECISION =====
if estimasi > 350000:
    print("\n⚠️ Over Budget")
else:
    print("\n✅ Aman")
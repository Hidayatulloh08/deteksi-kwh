import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

FILE = "data.csv"
MODEL = "model_lstm.keras"
WINDOW = 14  # 🔥 samakan dengan training

# ===== LOAD DATA =====
if not os.path.exists(FILE):
    print("❌ data.csv tidak ditemukan")
    exit()

data = pd.read_csv(FILE, on_bad_lines='skip')

if 'biaya' not in data.columns:
    print("❌ Kolom biaya tidak ada")
    exit()

# 🔥 smoothing (harus sama dengan training)
data['biaya'] = data['biaya'].rolling(window=3).mean()
data = data.dropna()

if len(data) < WINDOW:
    print("❌ Data kurang dari", WINDOW)
    exit()

dataset = data[['biaya']].values

# ===== SCALER =====
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

# ===== LOAD MODEL =====
try:
    model = load_model(MODEL)
except:
    print("❌ Model tidak ditemukan")
    exit()

# ===== PREDIKSI =====
last = scaled[-WINDOW:]
X_test = np.array([last[:, 0]]).reshape((1, WINDOW, 1))

pred = model.predict(X_test, verbose=0)
pred_real = scaler.inverse_transform(pred)

hasil = float(pred_real[0][0])

# ===== ANALISIS =====
rata = dataset.mean()
estimasi = rata * 30

# ===== ERROR (OPSIONAL, UNTUK EVALUASI) =====
actual = dataset[-1][0]
mae = abs(actual - hasil)

mape = 0
if actual != 0:
    mape = (mae / actual) * 100

# ===== OUTPUT =====
print("\n===== HASIL PREDIKSI =====")
print("💰 Prediksi besok: Rp", int(hasil))
print("📊 Rata harian:", int(rata))
print("📈 Estimasi bulanan:", int(estimasi))

print("\n===== EVALUASI =====")
print("📉 MAE:", round(mae,2))
print("📊 MAPE:", round(mape,2), "%")

# ===== DECISION =====
if estimasi > 350000:
    print("\n⚠️ Over Budget")
else:
    print("\n✅ Aman")
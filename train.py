import numpy as np
import pandas as pd
import joblib
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# LOAD DATA
# =========================
FILE = "data/data.csv"

if not os.path.exists(FILE):
    raise Exception("❌ data.csv tidak ditemukan")

df = pd.read_csv(FILE, on_bad_lines='skip')

if 'biaya' not in df.columns:
    raise Exception("❌ Kolom 'biaya' tidak ditemukan")

# =========================
# FIX TIMESTAMP (ANTI ERROR)
# =========================
if 'timestamp' not in df.columns:
    print("⚠️ timestamp tidak ditemukan → generate otomatis")
    df['timestamp'] = pd.date_range(
        start='2024-01-01',
        periods=len(df),
        freq='H'
    )

df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# =========================
# FEATURE ENGINEERING
# =========================
df['hour'] = df['timestamp'].dt.hour.fillna(0)
df['day'] = df['timestamp'].dt.day.fillna(0)

# =========================
# CLEANING
# =========================
df = df.dropna(subset=["biaya"])

# smoothing biar stabil
df['biaya'] = df['biaya'].rolling(window=3, min_periods=1).mean()

# =========================
# DATASET (MULTI FEATURE)
# =========================
dataset = df[['biaya', 'hour', 'day']].values

# =========================
# SCALER
# =========================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.save")
print("✅ Scaler disimpan")

# =========================
# WINDOW
# =========================
WINDOW = 7

X, y = [], []

for i in range(WINDOW, len(scaled)):
    X.append(scaled[i-WINDOW:i])
    y.append(scaled[i][0])  # target hanya biaya

X = np.array(X)
y = np.array(y)

if len(X) < 10:
    raise Exception("❌ Data terlalu sedikit")

# =========================
# SPLIT DATA
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =========================
# MODEL LSTM
# =========================
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(WINDOW, 3)))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# =========================
# TRAIN
# =========================
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# PREDIKSI
# =========================
pred = model.predict(X_test, verbose=0)

# inverse scaling khusus biaya
dummy_pred = np.zeros((len(pred), 3))
dummy_pred[:, 0] = pred[:, 0]

dummy_real = np.zeros((len(y_test), 3))
dummy_real[:, 0] = y_test

pred_inv = scaler.inverse_transform(dummy_pred)[:, 0]
real_inv = scaler.inverse_transform(dummy_real)[:, 0]

# =========================
# EVALUASI
# =========================
mae = mean_absolute_error(real_inv, pred_inv)

real_safe = np.where(real_inv == 0, 1, real_inv)
mape = np.mean(np.abs((real_inv - pred_inv) / real_safe)) * 100

rmse = np.sqrt(mean_squared_error(real_inv, pred_inv))
r2 = r2_score(real_inv, pred_inv)

print("\n===== HASIL EVALUASI =====")
print("📉 MAE  :", round(mae, 2))
print("📊 MAPE :", round(mape, 2), "%")
print("📉 RMSE :", round(rmse, 2))
print("📊 R2   :", round(r2, 3))

# =========================
# PLOT
# =========================
plt.figure()
plt.plot(real_inv, label="Actual")
plt.plot(pred_inv, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("model/prediksi.png")
plt.close()

# =========================
# SAVE MODEL
# =========================
model.save("model/model_lstm.keras")

print("\n✅ Model + scaler + evaluasi + plot selesai")
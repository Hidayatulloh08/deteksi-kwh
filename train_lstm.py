import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')  # 🔥 penting untuk server/headless
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===== LOAD DATA =====
df = pd.read_csv("data_dummy.csv")

# ===== VALIDASI KOLOM =====
if 'biaya' not in df.columns:
    raise Exception("Kolom 'biaya' tidak ditemukan di CSV")

# ===== CLEANING =====
df = df.dropna(subset=["biaya"])

# 🔥 SMOOTHING (lebih stabil)
df['biaya'] = df['biaya'].rolling(window=3, min_periods=1).mean()

# ===== DATASET =====
dataset = df[['biaya']].values

# ===== SCALING =====
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

# 🔥 SIMPAN SCALER (WAJIB UNTUK SERVER)
joblib.dump(scaler, "scaler.save")
print("✅ Scaler disimpan")

# ===== WINDOW =====
window = 14

X, y = [], []

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)

# 🔥 VALIDASI DATA
if len(X) < 10:
    raise Exception("Data terlalu sedikit untuk training")

X = X.reshape((X.shape[0], X.shape[1], 1))

# ===== SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ===== MODEL =====
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(window,1)))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# ===== TRAIN =====
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# ===== PREDIKSI =====
pred = model.predict(X_test, verbose=0)

# 🔥 INVERSE SCALING
pred_inv = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# ===== EVALUASI =====
mae = mean_absolute_error(y_test_inv, pred_inv)

# 🔥 ANTI DIVISION BY ZERO
mape = np.mean(
    np.abs((y_test_inv - pred_inv) / np.maximum(y_test_inv, 1))
) * 100

print("📉 MAE:", round(mae, 2))
print("📊 MAPE:", round(mape, 2), "%")

# ===== PLOT =====
plt.figure()
plt.plot(y_test_inv, label="Actual")
plt.plot(pred_inv, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.tight_layout()
plt.savefig("prediksi.png")
plt.close()

# ===== SAVE MODEL =====
model.save("model_lstm.keras")

print("✅ Model selesai + evaluasi + plot")
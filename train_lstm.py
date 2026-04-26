import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ===== LOAD DATA =====
df = pd.read_csv("data.csv", on_bad_lines='skip')

# ===== CLEANING =====
df = df.dropna(subset=["biaya"])

# 🔥 SMOOTHING
df['biaya'] = df['biaya'].rolling(window=3).mean()

# hapus NaN hasil smoothing
df = df.dropna()

dataset = df[['biaya']].values

# ===== SCALING =====
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

# 🔥 SIMPAN SCALER (WAJIB UNTUK SERVER)
joblib.dump(scaler, "scaler.save")

# ===== WINDOW =====
window = 14

X, y = [], []

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)

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
    validation_data=(X_test, y_test)
)

# ===== PREDIKSI =====
pred = model.predict(X_test)

# inverse scaling
pred_inv = scaler.inverse_transform(pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

# ===== EVALUASI =====
mae = mean_absolute_error(y_test_inv, pred_inv)

mape = np.mean(np.abs((y_test_inv - pred_inv) / y_test_inv)) * 100

print("📉 MAE:", mae)
print("📊 MAPE:", mape, "%")

# ===== PLOT =====
plt.figure()
plt.plot(y_test_inv, label="Actual")
plt.plot(pred_inv, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.savefig("prediksi.png")
plt.close()

# ===== SAVE MODEL =====
model.save("model_lstm.keras")

print("✅ Model selesai + evaluasi + plot")

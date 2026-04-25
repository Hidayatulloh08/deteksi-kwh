import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# ===== LOAD DATA =====
df = pd.read_csv("data.csv", on_bad_lines='skip')

# 🔥 smoothing (penting!)
data['biaya'] = data['biaya'].rolling(window=3).mean()

# hapus NaN hasil smoothing
data = data.dropna()

dataset = data[['biaya']].values

# ===== SCALING =====
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

# ===== WINDOW LEBIH BESAR =====
window = 14

X, y = [], []

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

# ===== SPLIT DATA (WAJIB UNTUK JURNAL) =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ===== MODEL LSTM (LEBIH KUAT) =====
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

# ===== SAVE =====
model.save("model_lstm.keras")

print("✅ Model selesai (UPGRADE)")
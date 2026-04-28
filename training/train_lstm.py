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
df = pd.read_csv("data/data_dummy.csv")

# optional gabung data real
if os.path.exists("data/data.csv"):
    df_real = pd.read_csv("data/data.csv", on_bad_lines='skip')
    df = pd.concat([df, df_real])

# ===== CLEAN =====
df = df.dropna(subset=["biaya"])

# ===== SMOOTHING =====
df['biaya'] = df['biaya'].rolling(window=3, min_periods=1).mean()

# ===== TIME FEATURE =====
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
else:
    df['timestamp'] = pd.Timestamp.now()

df['hour'] = df['timestamp'].dt.hour.fillna(0)
df['day'] = df['timestamp'].dt.day.fillna(0)

# ===== DATASET =====
dataset = df[['biaya', 'hour', 'day']].values

# ===== SCALER =====
scaler = MinMaxScaler()
scaled = scaler.fit_transform(dataset)

joblib.dump(scaler, "scaler.save")

# ===== WINDOW =====
window = 14

X, y = [], []

for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ===== MODEL =====
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(window, 3)))
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
pred = model.predict(X_test)

# ===== INVERSE =====
pred_full = np.zeros((len(pred), 3))
pred_full[:, 0] = pred[:, 0]

y_full = np.zeros((len(y_test), 3))
y_full[:, 0] = y_test

pred_inv = scaler.inverse_transform(pred_full)[:, 0]
y_inv = scaler.inverse_transform(y_full)[:, 0]

# ===== METRICS =====
mae = mean_absolute_error(y_inv, pred_inv)

mape = np.mean(
    np.abs((y_inv - pred_inv) / np.maximum(y_inv, 1))
) * 100

print("📉 MAE:", round(mae, 2))
print("📊 MAPE:", round(mape, 2), "%")

# ===== SAVE MODEL =====
model.save("model_lstm.keras")

# ===== PLOT =====
plt.figure()
plt.plot(y_inv, label="Actual")
plt.plot(pred_inv, label="Predicted")
plt.legend()
plt.title("Prediction Result")
plt.savefig("prediksi.png")
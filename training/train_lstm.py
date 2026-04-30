import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data/data_dummy.csv")

if os.path.exists("data/data.csv"):
    df_real = pd.read_csv("data/data.csv", on_bad_lines='skip')
    df = pd.concat([df, df_real], ignore_index=True)

# =========================
# CLEAN DATA
# =========================
df = df.dropna(subset=["biaya"])
df = df.sort_values(by=df.columns[0])

# =========================
# TIME FEATURES
# =========================
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
else:
    df["timestamp"] = pd.Timestamp.now()

df["hour"] = df["timestamp"].dt.hour.fillna(0)
df["day"] = df["timestamp"].dt.day.fillna(0)

# =========================
# SAFE FEATURE ENGINEERING
# =========================
if "power" not in df.columns:
    df["power"] = 0

if "voltage" not in df.columns:
    df["voltage"] = 0

df["power_delta"] = df["power"].diff().fillna(0)
df["voltage_delta"] = df["voltage"].diff().fillna(0)

# =========================
# FEATURE MATRIX (TIDAK DIUBAH)
# =========================
dataset = df[['biaya', 'hour', 'day', 'power', 'power_delta']].values

# =========================
# TRAIN-TEST SPLIT (NO LEAKAGE)
# =========================
split_idx = int(len(dataset) * 0.8)

train_data = dataset[:split_idx]
test_data = dataset[split_idx:]

# =========================
# SCALER (TRAIN ONLY)
# =========================
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

joblib.dump(scaler, "scaler.save")

# =========================
# WINDOW FUNCTION
# =========================
window = 14

def create_dataset(data):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_scaled)
X_test, y_test = create_dataset(test_scaled)

# =========================
# MODEL LSTM (FIX INPUT SHAPE)
# =========================
model = Sequential()

# FIX: feature = 5 (biaya, hour, day, power, power_delta)
model.add(LSTM(64, return_sequences=True, input_shape=(window, 5)))
model.add(Dropout(0.2))

model.add(LSTM(32))
model.add(Dropout(0.2))

model.add(Dense(16, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# =========================
# TRAINING
# =========================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=1
)

# =========================
# PREDICTION
# =========================
pred = model.predict(X_test)

# =========================
# INVERSE TRANSFORM SAFE
# =========================
def inverse_transform_1d(values):
    temp = np.zeros((len(values), 5))
    temp[:, 0] = values
    return scaler.inverse_transform(temp)[:, 0]

y_inv = inverse_transform_1d(y_test)
pred_inv = inverse_transform_1d(pred[:, 0])

# =========================
# METRICS
# =========================
mae = mean_absolute_error(y_inv, pred_inv)

mape = np.mean(
    np.abs((y_inv - pred_inv) / np.maximum(y_inv, 1))
) * 100

print("📉 MAE:", round(mae, 2))
print("📊 MAPE:", round(mape, 2), "%")

# =========================
# SAVE MODEL
# =========================
model.save("model_lstm.keras")

# =========================
# PLOT RESULT (JURNAL READY)
# =========================
plt.figure()
plt.plot(y_inv, label="Actual")
plt.plot(pred_inv, label="Predicted")
plt.title("LSTM Energy Forecasting Result")
plt.legend()
plt.savefig("prediction_result.png")

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Training Loss Curve")
plt.savefig("loss_curve.png")
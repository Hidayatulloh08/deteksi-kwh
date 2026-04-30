import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

DATA_PATH = "data/data.csv"
MODEL_PATH = "model/model_lstm.keras"
SCALER_PATH = "model/scaler.save"

# =========================
# DRIFT DETECTION
# =========================
def check_drift(df):
    try:
        if len(df) < 10:
            return False

        std = df["power"].std()
        return std > 50  # threshold drift
    except:
        return False


# =========================
# RETRAIN
# =========================
def retrain():
    df = pd.read_csv(DATA_PATH)

    df = df.dropna(subset=["biaya"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day

    X = df[["biaya","power","voltage","current"]].fillna(0).values

    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X)

    window = 14
    X_train, y_train = [], []

    for i in range(window, len(Xs)):
        X_train.append(Xs[i-window:i])
        y_train.append(Xs[i,0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, X_train.shape[2])),
        LSTM(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=5, verbose=0)

    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("✅ MODEL RETRAINED")
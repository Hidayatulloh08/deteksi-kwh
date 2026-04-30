def train_model():
    import os, time
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    print("📊 Training model...")

    df_list = []

    if os.path.exists("data/data.csv"):
        df_list.append(pd.read_csv("data/data.csv", on_bad_lines='skip'))

    if len(df_list) == 0:
        print("⚠️ Tidak ada data")
        return

    df = pd.concat(df_list, ignore_index=True)

    if len(df) < 50:
        print("⏳ Data belum cukup")
        return

    df = df.dropna(subset=["biaya"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"] = df["timestamp"].dt.hour.fillna(0)
    df["day"] = df["timestamp"].dt.day.fillna(0)

    if "power" not in df.columns:
        df["power"] = 0

    df["power_delta"] = df["power"].diff().fillna(0)

    dataset = df[['biaya', 'hour', 'day', 'power', 'power_delta']].values

    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    joblib.dump(scaler, "scaler.save")

    window = 14

    if len(train_scaled) <= window or len(test_scaled) <= window:
        print("⚠️ Data kurang untuk window")
        return

    def create_dataset(data):
        X, y = [], []
        for i in range(window, len(data)):
            X.append(data[i-window:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_dataset(train_scaled)
    X_test, y_test = create_dataset(test_scaled)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(window, 5)))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    model.fit(X_train, y_train, epochs=20, batch_size=8, verbose=0)

    pred = model.predict(X_test)

    def inverse_transform(values):
        temp = np.zeros((len(values), 5))
        temp[:, 0] = values.flatten()
        return scaler.inverse_transform(temp)[:, 0]

    y_true = inverse_transform(y_test)
    y_pred = inverse_transform(pred)

    mae = mean_absolute_error(y_true, y_pred)

    print("📉 MAE:", round(mae, 2))

    model.save("model_lstm.keras")

    print("✅ Training selesai")
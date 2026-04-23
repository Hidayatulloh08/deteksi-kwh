from flask import Flask, request, jsonify
import pandas as pd
import os
from datetime import datetime
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import requests
import time

app = Flask(__name__)
FILE = "data.csv"

# ===== LOAD MODEL =====
model = None

def load_lstm():
    global model
    if model is None:
        print("Loading model...")
        model = load_model("model_lstm.keras")

# ===== TELEGRAM (AMAN) =====
TOKEN = os.environ.get("TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")

def kirim_notif(pesan):
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
        data = {"chat_id": CHAT_ID, "text": pesan}
        requests.post(url, data=data)
    except Exception as e:
        print("ERROR TELEGRAM:", e)

# ===== TIMER =====
last_notif_time = 0

@app.route('/data', methods=['POST'])
def receive_data():
    global last_notif_time

    data = request.json
    print("DATA MASUK:", data)

    # ===== TAMBAH WAKTU =====
    now = datetime.now()
    data['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
    data['day'] = now.day
    data['hour'] = now.hour

    columns = ["timestamp","day","hour","voltage","current","power","kwh","biaya"]
    df_new = pd.DataFrame([data])[columns]

    # ===== SIMPAN CSV =====
    if not os.path.exists(FILE):
        df_new.to_csv(FILE, index=False)
    else:
        df_new.to_csv(FILE, mode='a', header=False, index=False)

    # ===== LOAD DATA =====
    try:
        df = pd.read_csv(FILE)
    except:
        print("CSV rusak, reset ulang")
        os.remove(FILE)
        return jsonify({"status":"reset csv"})

    if len(df) < 5:
        return jsonify({"status":"data belum cukup"})

    # ===== ML =====
    dataset = df[['biaya']].values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    last_data = scaled_data[-3:]

    X_test = np.array([last_data[:,0]])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # ✅ LOAD MODEL DI SINI (BENAR)
    load_lstm()

    pred = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred)
    pred_value = float(pred_price[0][0])

    # ===== ANALISIS =====
    total = dataset.sum()
    rata = dataset.mean()
    pred_bulanan = rata * 30
    budget = 350000

    status = "⚠️ Over Budget" if pred_bulanan > budget else "✅ Aman"

    # ===== NOTIF TIAP 10 MENIT =====
    current_time = time.time()

    if current_time - last_notif_time > 600:
        kirim_notif(
            f"⚡ MONITORING LISTRIK\n\n"
            f"📅 {data['timestamp']}\n\n"
            f"🔌 {round(data['voltage'],1)} V\n"
            f"⚡ {round(data['current'],2)} A\n"
            f"💡 {round(data['power'],1)} W\n\n"
            f"💰 Total: Rp {int(total)}\n"
            f"📊 Harian: Rp {int(pred_value)}\n"
            f"📈 Bulanan: Rp {int(pred_bulanan)}\n\n"
            f"{status}"
        )
        last_notif_time = current_time

    # ===== OUTPUT =====
    result = {
        "prediksi_harian": int(pred_value),
        "total": int(total),
        "prediksi_bulanan": int(pred_bulanan),
        "status": status
    }

    print("HASIL AI:", result)

    return jsonify(result)


@app.route('/get-data', methods=['GET'])
def get_data():
    if not os.path.exists(FILE):
        return jsonify([])

    df = pd.read_csv(FILE)
    return df.tail(50).to_json(orient='records')


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
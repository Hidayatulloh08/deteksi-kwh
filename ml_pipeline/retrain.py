import time
import pandas as pd
import os

from training.train_lstm import train_model  # pastikan fungsi ini ada

DATA_PATH = "data/data.csv"

def start_retrain():
    print("🤖 ML RETRAIN THREAD STARTED")

    while True:
        try:
            if os.path.exists(DATA_PATH):
                df = pd.read_csv(DATA_PATH)

                # 🔥 syarat minimal data
                if len(df) > 200:
                    print("🔁 Retraining model...")
                    train_model()   # panggil training
                else:
                    print("⏳ Data belum cukup untuk training")

            else:
                print("⚠️ data.csv belum ada")

        except Exception as e:
            print("❌ ERROR RETRAIN:", e)

        time.sleep(3600)  # tiap 1 jam
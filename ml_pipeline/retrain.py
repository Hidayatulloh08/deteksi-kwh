import time
import pandas as pd
import os
from training.train_lstm import train_model

DATA_PATH = "data/data.csv"

def start_retrain():
    print("🤖 ML RETRAIN THREAD STARTED")

    while True:
        try:
            if not os.path.exists(DATA_PATH):
                print("⚠️ data.csv belum ada")
            else:
                df = pd.read_csv(DATA_PATH)

                if len(df) > 200:
                    print("🔁 Retraining...")
                    train_model()
                else:
                    print("⏳ Data belum cukup")

        except Exception as e:
            print("❌ ERROR RETRAIN:", e)

        time.sleep(3600)  # tiap 1 jam
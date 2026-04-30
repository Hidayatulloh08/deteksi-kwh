import pandas as pd
import os

def to_float(x):
    try:
        return float(x)
    except:
        return 0.0


def load_csv_safe(path):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)

        # pastikan kolom penting ada
        for col in ["voltage", "current", "power", "kwh", "biaya"]:
            if col not in df.columns:
                df[col] = 0

        return df

    except Exception as e:
        print("❌ ERROR LOAD CSV:", e)
        return pd.DataFrame()
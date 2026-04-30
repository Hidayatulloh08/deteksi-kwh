import pandas as pd
import os

# =========================
# SAFE FLOAT CONVERTER
# =========================
def to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except:
        return default


# =========================
# LOAD CSV AMAN
# =========================
def load_csv_safe(path):
    try:
        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)

        if df.empty:
            return pd.DataFrame()

        return df

    except Exception as e:
        print("❌ ERROR load_csv:", e)
        return pd.DataFrame()
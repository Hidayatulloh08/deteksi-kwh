import pandas as pd
import os

# =========================
# SAFE FLOAT
# =========================
def to_float(x):
    try:
        return float(str(x).replace(",", "."))
    except:
        return 0.0


# =========================
# SAFE CSV LOADER
# =========================
def load_csv_safe(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, on_bad_lines='skip')
    except Exception as e:
        print("🔥 CSV ERROR:", e)
        os.remove(path)
        return pd.DataFrame()
    print("✅ helper loaded")
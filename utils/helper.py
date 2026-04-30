import pandas as pd

def to_float(val):
    try:
        return float(val)
    except:
        return 0.0

def load_csv_safe(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()
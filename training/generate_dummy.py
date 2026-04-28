import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# pastikan folder data ada
os.makedirs("data", exist_ok=True)

rows = []
start = datetime.now() - timedelta(days=30)

for i in range(30 * 24):  # 30 hari per jam
    waktu = start + timedelta(hours=i)

    hour = waktu.hour

    # pola realistis
    if 0 <= hour <= 6:
        base = 200
    elif 7 <= hour <= 17:
        base = 400
    else:
        base = 700

    noise = np.random.randint(-50, 50)
    biaya = max(base + noise, 50)

    rows.append({
        "timestamp": waktu.strftime("%Y-%m-%d %H:%M:%S"),
        "biaya": biaya
    })

df = pd.DataFrame(rows)

# simpan ke data/data.csv (INI PENTING)
df.to_csv("data/data.csv", index=False)

print("✅ Data dummy berhasil dibuat di data/data.csv")
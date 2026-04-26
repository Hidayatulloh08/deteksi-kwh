import pandas as pd
import numpy as np

rows = []

for day in range(1, 31):  # 30 hari
    for hour in range(24):
        # pola realistis
        if 0 <= hour <= 6:
            base = 200
        elif 7 <= hour <= 17:
            base = 400
        else:
            base = 700

        noise = np.random.randint(-50, 50)
        biaya = base + noise

        rows.append({
            "biaya": max(biaya, 50)
        })

df = pd.DataFrame(rows)
df.to_csv("data_dummy.csv", index=False)

print("✅ Data dummy dibuat")
# =========================
# CONFIG SYSTEM
# =========================

# ===== AI =====
WINDOW = 7   # jumlah data untuk LSTM

# ===== THRESHOLD =====
THRESHOLD_DEFAULT = 500      # batas awal beban tinggi (W)
SPIKE_THRESHOLD = 200        # lonjakan mendadak (W)
LOW_POWER = 1                # dianggap mati (W)
MAX_POWER_NORMAL = 900       # batas normal alat rumah

# ===== NOTIF =====
NOTIF_INTERVAL = 600         # 10 menit (lebih realistis dari 300)

# ===== BIAYA =====
BUDGET_BULANAN = 300000
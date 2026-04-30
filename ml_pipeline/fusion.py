def fusion_engine(rule_label, anomaly, ai_pred, confidence):

    # =========================
    # PRIORITAS 1: kondisi kritis REAL
    # =========================
    if rule_label == "PLN_MATI":
        return "CRITICAL_OFF"

    if rule_label == "KONSLETING":
        return "CRITICAL_SHORT"

    # =========================
    # PRIORITAS 2: anomaly (harus valid + confidence tinggi)
    # =========================
    if anomaly and confidence >= 0.7:
        return "CRITICAL_ANOMALY"

    # =========================
    # PRIORITAS 3: confidence rendah → abaikan AI
    # =========================
    if confidence < 0.6:
        return rule_label  # 🔥 penting: jangan buat warning baru

    # =========================
    # PRIORITAS 4: prediksi AI ekstrem
    # =========================
    if ai_pred > 2000 and rule_label == "NORMAL" and confidence >= 0.7:
        return "POTENTIAL_SPIKE"

    return rule_label
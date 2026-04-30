def fusion_engine(rule_label, anomaly, ai_pred, confidence):

    # PRIORITAS 1: kondisi kritis
    if rule_label == "PLN_MATI":
        return "CRITICAL_OFF"

    if rule_label == "KONSLETING":
        return "CRITICAL_SHORT"

    # PRIORITAS 2: anomaly
    if anomaly:
        return "CRITICAL_ANOMALY"

    # PRIORITAS 3: confidence rendah
    if confidence < 0.6:
        return "LOW_CONFIDENCE_WARNING"

    # PRIORITAS 4: prediksi AI tinggi tapi rule normal
    if ai_pred > 2000 and rule_label == "NORMAL":
        return "POTENTIAL_SPIKE"

    return rule_label
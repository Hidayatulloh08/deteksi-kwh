def fusion_engine(rule_label, anomaly, ai_pred, confidence):

    if rule_label == "PLN_MATI":
        return "CRITICAL_OFF"

    if anomaly and rule_label == "WASPADA":
        return "CRITICAL_ANOMALY"

    if confidence < 0.6:
        return "LOW_CONFIDENCE_WARNING"

    return rule_label
def check_drift(df):
    if len(df) < 10:
        return False
    return df["power"].std() > 50


def detect_anomaly(power, voltage, mean_p, std_p, mean_v, std_v):
    z_p = (power - mean_p) / (std_p + 1e-6)
    z_v = (voltage - mean_v) / (std_v + 1e-6)

    return abs(z_p) > 3 or abs(z_v) > 2
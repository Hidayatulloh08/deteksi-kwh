def classify_load(power):

    if power < 80:
        return "FAN (KIPAS)"

    elif 80 <= power < 400:
        return "LIGHT_LOAD (LAMPU)"

    elif 400 <= power < 1000:
        return "IRON (SETRIKA)"

    else:
        return "HEAVY_LOAD (AC / HEATER)"
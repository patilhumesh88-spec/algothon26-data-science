"""Feature engineering. Uses ONLY inputs that exist before/independent of the failure label."""
import numpy as np
import pandas as pd

from .config import AIR, PROC, RPM, SENSOR_COLS, STRAIN_LIMIT, TORQUE, TYPE, TYPE_MAP, WEAR

RAW_FEATURES = SENSOR_COLS + ["Type_code"]
ENGINEERED_FEATURES = RAW_FEATURES + ["temp_diff", "power_w", "strain", "strain_ratio"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in SENSOR_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["Type_code"] = df[TYPE].map(TYPE_MAP).astype(float)
    X["temp_diff"] = X[PROC] - X[AIR]
    X["power_w"] = X[TORQUE] * X[RPM] * 2 * np.pi / 60.0
    X["strain"] = X[WEAR] * X[TORQUE]
    X["strain_ratio"] = X["strain"] / df[TYPE].map(STRAIN_LIMIT).astype(float)
    return X

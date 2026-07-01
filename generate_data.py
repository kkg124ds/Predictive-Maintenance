"""
generate_data.py
----------------
Synthetic dataset generator for Predictive Maintenance.
Simulates industrial motor sensor readings over time.

Based on real-world patterns from:
  - CWRU Bearing Dataset characteristics
  - ISO 10816 vibration standards
  - IEEE PHM 2012 challenge patterns

Run:
    python src/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
N_MACHINES   = 5
DAYS         = 365
READINGS_PER_DAY = 24          # hourly readings
TOTAL        = N_MACHINES * DAYS * READINGS_PER_DAY
FAILURE_RATE = 0.08            # ~8% failure events

def generate_sensor_data() -> pd.DataFrame:
    """
    Generates realistic sensor readings for N industrial motors.

    Features engineered from domain knowledge (5 yrs EE maintenance):
      - Temperature degrades bearing lubrication above 85°C
      - Vibration RMS > 7.1 mm/s (ISO 10816 Zone C) = Alarm
      - Current imbalance > 2% = Bearing wear indicator
      - Pressure drop signals seal failure
      - RPM deviation from setpoint = load anomaly

    Returns:
        pd.DataFrame with shape (~43,800, 13)
    """
    records = []

    for machine_id in range(1, N_MACHINES + 1):
        # Machine-specific base operating parameters
        base_temp     = np.random.uniform(55, 70)
        base_vib      = np.random.uniform(1.5, 3.5)
        base_current  = np.random.uniform(18, 25)
        base_pressure = np.random.uniform(4.5, 6.0)
        base_rpm      = np.random.choice([1450, 1480, 2900, 2950])

        # Inject failure events randomly
        failure_windows = []
        for _ in range(np.random.randint(2, 6)):
            start = np.random.randint(0, DAYS - 14)
            end   = start + np.random.randint(7, 14)  # degradation lasts 7-14 days
            failure_windows.append((start, end))

        for day in range(DAYS):
            for hour in range(READINGS_PER_DAY):
                timestamp = pd.Timestamp("2023-01-01") + \
                            pd.Timedelta(days=(machine_id - 1) * 0 + day, hours=hour)

                # ── Determine failure proximity ──────────────────────
                in_degradation = any(start <= day < end for start, end in failure_windows)
                at_failure     = any(abs(day - end) <= 1 for _, end in failure_windows)

                degradation_factor = 0.0
                if in_degradation:
                    # Linear degradation ramp
                    for start, end in failure_windows:
                        if start <= day < end:
                            degradation_factor = max(
                                degradation_factor,
                                (day - start) / (end - start)
                            )

                # ── Sensor readings with realistic noise ────────────
                temp_noise     = np.random.normal(0, 1.5)
                vib_noise      = np.random.normal(0, 0.3)
                current_noise  = np.random.normal(0, 0.5)
                pressure_noise = np.random.normal(0, 0.1)
                rpm_noise      = np.random.normal(0, 15)

                # Sensors degrade as failure approaches
                temperature = (
                    base_temp + temp_noise
                    + degradation_factor * np.random.uniform(20, 35)   # heating
                    + (15 if at_failure else 0)
                )
                vibration_rms = (
                    base_vib + abs(vib_noise)
                    + degradation_factor * np.random.uniform(4, 8)     # bearing wear
                    + (3.5 if at_failure else 0)
                )
                current_draw = (
                    base_current + current_noise
                    + degradation_factor * np.random.uniform(3, 7)     # increased load
                )
                pressure = (
                    base_pressure + pressure_noise
                    - degradation_factor * np.random.uniform(0.8, 1.5) # seal degradation
                )
                rpm = (
                    base_rpm + rpm_noise
                    - degradation_factor * np.random.uniform(50, 120)  # speed drop
                )

                # ── Derived / Engineered Features ───────────────────
                # Rolling-style calculations simulated per row
                current_imbalance = abs(np.random.normal(0, 0.3) + degradation_factor * 2.5)
                power_factor      = max(0.6, min(1.0,
                    0.92 - degradation_factor * 0.15 + np.random.normal(0, 0.02)
                ))
                vibration_peak    = vibration_rms * np.random.uniform(2.5, 3.5)
                bearing_temp      = temperature + np.random.normal(5, 2) + degradation_factor * 10

                # ── Label ────────────────────────────────────────────
                if at_failure and np.random.random() < 0.85:
                    failure_label = 1
                elif in_degradation and degradation_factor > 0.7 and np.random.random() < 0.4:
                    failure_label = 1
                else:
                    failure_label = 0

                # ── Failure type (multi-class) ───────────────────────
                if failure_label == 1:
                    failure_type = np.random.choice(
                        ["bearing_failure", "overheating", "electrical_fault", "seal_failure"],
                        p=[0.45, 0.25, 0.20, 0.10]
                    )
                else:
                    failure_type = "normal"

                records.append({
                    "timestamp":          timestamp,
                    "machine_id":         f"M{machine_id:03d}",
                    "temperature_c":      round(temperature, 2),
                    "vibration_rms":      round(max(0.1, vibration_rms), 3),
                    "vibration_peak":     round(max(0.1, vibration_peak), 3),
                    "current_draw_a":     round(max(1.0, current_draw), 2),
                    "current_imbalance":  round(max(0.0, current_imbalance), 3),
                    "pressure_bar":       round(max(0.5, pressure), 3),
                    "rpm":                round(max(100, rpm), 1),
                    "bearing_temp_c":     round(max(20, bearing_temp), 2),
                    "power_factor":       round(power_factor, 4),
                    "failure_label":      failure_label,
                    "failure_type":       failure_type,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    print(f"Dataset shape : {df.shape}")
    print(f"Failure rate  : {df['failure_label'].mean():.2%}")
    print(f"Failure types :\n{df['failure_type'].value_counts()}")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-window aggregation features per machine.
    These capture trend and volatility — critical for predictive maintenance.
    """
    feature_cols = [
        "temperature_c", "vibration_rms", "current_draw_a",
        "pressure_bar", "rpm", "bearing_temp_c"
    ]
    result = []

    for machine_id, group in df.groupby("machine_id"):
        group = group.copy().sort_values("timestamp")
        for col in feature_cols:
            group[f"{col}_roll6h_mean"]  = group[col].rolling(6,  min_periods=1).mean()
            group[f"{col}_roll24h_mean"] = group[col].rolling(24, min_periods=1).mean()
            group[f"{col}_roll24h_std"]  = group[col].rolling(24, min_periods=1).std().fillna(0)
        result.append(group)

    df_out = pd.concat(result).reset_index(drop=True)
    print(f"After rolling features: {df_out.shape[1]} columns")
    return df_out


if __name__ == "__main__":
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)

    df_raw = generate_sensor_data()
    df_raw.to_csv(out_dir / "sensor_data_raw.csv", index=False)
    print(f"\n✅ Raw data saved → data/sensor_data_raw.csv")

    df_feat = add_rolling_features(df_raw)
    df_feat.to_csv(out_dir / "sensor_data_features.csv", index=False)
    print(f"✅ Feature data saved → data/sensor_data_features.csv")

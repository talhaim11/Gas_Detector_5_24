"""
Data acquisition and logging. Saves automation scan data to CSV in required format.
Filename: scan_YYYY-MM-DD_HHMMSS_dir-R_step-<value>.csv
Columns: timestamp, scan_index, direction, step_size, position, sensor_1, sensor_2, ...
"""
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Union, Optional

from motor_app.config import (
    DATA_SAVE_DIR,
    CSV_FILENAME_PREFIX,
    CSV_FILENAME_DATE_FMT,
    CSV_FILENAME_TIME_FMT,
    CSV_PARTIAL_SUFFIX,
)


def _sensor_reading_to_row(index: int, direction: str, step_size: int, reading) -> List[str]:
    """Convert one SensorReading (or dict) to CSV row: timestamp, scan_index, direction, step_size, position, sensor_1, ..."""
    if hasattr(reading, "timestamp"):
        ts = reading.timestamp
        pos = reading.position
        vals = reading.values
    else:
        ts = reading.get("timestamp", "")
        pos = reading.get("position", "")
        vals = reading.get("values", [])
    return [ts, str(index), direction, str(step_size), pos] + [str(v) for v in vals]


def build_scan_filename(
    direction: str,
    step_size: int,
    partial: bool = False,
    save_dir: Optional[Path] = None,
) -> Path:
    """
    Build path: save_dir / scan_YYYY-MM-DD_HHMMSS_dir-<R|L>_step-<value>.csv
    If partial: ..._step-<value>_partial.csv
    """
    save_dir = save_dir or DATA_SAVE_DIR
    now = datetime.now()
    date_str = now.strftime(CSV_FILENAME_DATE_FMT)
    time_str = now.strftime(CSV_FILENAME_TIME_FMT)
    base = f"{CSV_FILENAME_PREFIX}_{date_str}_{time_str}_dir-{direction}_step-{step_size}"
    if partial:
        base += CSV_PARTIAL_SUFFIX
    return save_dir / f"{base}.csv"


def save_scan_csv(
    rows: List[Union[object, dict]],
    direction: str,
    step_size: int,
    partial: bool = False,
    save_dir: Optional[Path] = None,
    sensor_headers: Optional[List[str]] = None,
) -> Path:
    """
    Write CSV with columns: timestamp, scan_index, direction, step_size, position, sensor_1, sensor_2, ...
    rows: list of SensorReading or dict with timestamp, position, values.
    sensor_headers: optional names for sensor columns (sensor_1, sensor_2, ...).
    Returns path to written file.
    """
    path = build_scan_filename(direction, step_size, partial=partial, save_dir=save_dir)
    num_sensors = 0
    if rows:
        first = rows[0]
        if hasattr(first, "values"):
            num_sensors = len(first.values)
        else:
            num_sensors = len(first.get("values", []))
    headers = ["timestamp", "scan_index", "direction", "step_size", "position"]
    if sensor_headers:
        headers.extend(sensor_headers[:num_sensors])
    else:
        headers.extend([f"sensor_{i+1}" for i in range(num_sensors)])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for i, r in enumerate(rows, 1):
            w.writerow(_sensor_reading_to_row(i, direction, step_size, r))
    return path


class DataLogger:
    """
    Wraps save_scan_csv and optional live buffer for UI.
    Used by GUI/controller after automation run to save to C:\\Users\\...\\Downloads.
    """
    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir or DATA_SAVE_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_automation_scan(
        self,
        rows: List[Union[object, dict]],
        direction: str,
        step_size: int,
        partial: bool = False,
        sensor_headers: Optional[List[str]] = None,
    ) -> Path:
        """Save automation buffer to CSV. Returns path."""
        return save_scan_csv(
            rows=rows,
            direction=direction,
            step_size=step_size,
            partial=partial,
            save_dir=self.save_dir,
            sensor_headers=sensor_headers,
        )

    def save_ratio_measurements(self, rows: List[dict]) -> Path:
        """
        Save manual ratio measurements (live sensor data) to a simple CSV:
        columns: timestamp, signal_2.0/2.1_M, signal_2.3_M, ratio_raw, ratio_calibrated.
        """
        now = datetime.now()
        date_str = now.strftime(CSV_FILENAME_DATE_FMT)
        time_str = now.strftime(CSV_FILENAME_TIME_FMT)
        path = self.save_dir / f"ratio_{date_str}_{time_str}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "signal_2.0/2.1_M", "signal_2.3_M", "ratio_raw", "ratio_calibrated"])
            for r in rows:
                w.writerow(
                    [
                        r.get("timestamp", ""),
                        r.get("signal_2_0_2_1_M", ""),
                        r.get("signal_2_3_M", ""),
                        r.get("ratio_raw", ""),
                        r.get("ratio_calibrated", ""),
                    ]
                )
        return path

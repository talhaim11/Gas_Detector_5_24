"""
Entry point for the motor control + sensor application.
Run: python -m motor_app.main
Or: python motor_app/main.py
"""
import argparse
import sys

from motor_app.config import (
    DEFAULT_MOTOR_PORT,
    DEFAULT_MOTOR_BAUD,
    DEFAULT_SENSOR_PORT,
    DEFAULT_SENSOR_BAUD,
)
from motor_app.gui import run_gui


def main():
    parser = argparse.ArgumentParser(description="Motor control + sensor GUI (Manual / Automation modes)")
    parser.add_argument("--port", default=DEFAULT_MOTOR_PORT, help=f"Motor serial port (default: {DEFAULT_MOTOR_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_MOTOR_BAUD, help=f"Motor baud rate (default: {DEFAULT_MOTOR_BAUD})")
    parser.add_argument("--sensor-port", default=DEFAULT_SENSOR_PORT, help=f"Sensor serial port (default: {DEFAULT_SENSOR_PORT})")
    parser.add_argument("--sensor-baud", type=int, default=DEFAULT_SENSOR_BAUD, help=f"Sensor baud rate (default: {DEFAULT_SENSOR_BAUD})")
    args = parser.parse_args()
    run_gui(
        motor_port=args.port,
        motor_baud=args.baud,
        sensor_port=args.sensor_port,
        sensor_baud=args.sensor_baud,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

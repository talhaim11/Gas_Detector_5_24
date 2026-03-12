"""
Application configuration: paths, defaults, serial and protocol constants.
"""
import os
from pathlib import Path

# --- Paths ---
# Data is saved here in Automation mode (per requirement).
DATA_SAVE_DIR = Path(r"C:\Users\thaim\Downloads")
os.makedirs(DATA_SAVE_DIR, exist_ok=True)

# --- Serial ---
DEFAULT_MOTOR_PORT = "COM5"
DEFAULT_MOTOR_BAUD = 9600
DEFAULT_SENSOR_PORT = "COM8"
DEFAULT_SENSOR_BAUD = 230400
SERIAL_TIMEOUT = 0.5
SERIAL_WRITE_TIMEOUT = 2.0

# --- Protocol: Commands from PC to Arduino ---
CMD_PING = "PING"
CMD_STOP_ALL = "STOP ALL"
CMD_START = "START"       # START <axis> <dir> <speed>
CMD_STOP = "STOP"          # STOP <axis>
CMD_UPDATE_SPEED = "UPDATE_SPEED"  # UPDATE_SPEED <axis> <speed>
CMD_STEP = "STEP"          # STEP <axis> <dir> <step_size>
CMD_READ_SENSORS = "READ_SENSORS"

# --- Protocol: Responses from Arduino ---
RSP_ACK = "ACK"
RSP_ERR = "ERR"
RSP_MOTION_DONE = "MOTION_DONE"
RSP_SENSORS = "SENSORS"

# --- Axes / Directions ---
AXIS_1 = "1"
AXIS_2 = "2"
DIR_LEFT = "L"
DIR_RIGHT = "R"
DIR_UP = "U"
DIR_DOWN = "D"

# --- Defaults for GUI ---
DEFAULT_MANUAL_SPEED = 50
MIN_MANUAL_SPEED = 10   # Below 10 no perceptible difference for motor
MAX_MANUAL_SPEED = 80   # 80-100 too high for motor
DEFAULT_STEP_SIZE = 10
MIN_STEP_SIZE = 1
MAX_STEP_SIZE = 100
DEFAULT_NUM_STEPS = 20
MIN_NUM_STEPS = 1
MAX_NUM_STEPS = 1000

# Delay (ms) for automation step interval and Arduino SPEED mapping. Arduino accepts 50-5000 us.
# No Arduino/motor limit at 100-2000 ms; that was a previous GUI choice. Widened to full useful range.
MIN_DELAY_MS = 50
MAX_DELAY_MS = 5000
DEFAULT_DELAY_MS = 500

# --- CSV / Logging ---
CSV_FILENAME_PREFIX = "scan"
CSV_FILENAME_DATE_FMT = "%Y-%m-%d"
CSV_FILENAME_TIME_FMT = "%H%M%S"
CSV_PARTIAL_SUFFIX = "_partial"

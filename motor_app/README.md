# Motor Control + Sensor Application

Structured motor control with Manual (hold-to-move) and Automation (step scan) modes, state machine, and data logging.

## File Structure

```
motor_app/
  __init__.py
  config.py           # Paths, serial defaults, protocol constants
  state_machine.py    # AppState enum, MotorStateMachine, validated transitions
  serial_manager.py   # Serial I/O, protocol framing, command/response parsing
  motor_controller.py # Bridges state machine + serial; manual + automation logic
  data_logger.py      # CSV save: scan_YYYY-MM-DD_HHMMSS_dir-R_step-<value>.csv
  gui.py              # Tkinter GUI: mode switch, manual/auto controls, live sensor
  main.py             # Entry point
  README.md
```

## Run

From project root:

```bash
python -m motor_app.main
# Or with port/baud:
python -m motor_app.main --port COM5 --baud 9600
```

In the GUI, you can now choose **Motor COM** and **Sensor COM** from dropdowns at the top bar, then click **Connect**. Use **Refresh COM** after plugging in a device.

You can also choose **Connect mode**:
- **Motor + Signal**: full motor control + live sensor.
- **Signal Only**: sensor-only monitoring/logging without opening motor COM.

## Arduino Mount Firmware

**arduino_code_mount.ino** is the firmware used with the app. It supports:

- **PING** — replies `PONG` (connection check).
- **SPEED \<us\>** — step delay in microseconds (50–5000). Higher = slower.
- **R1/L1 \<steps\>** — axis 1 right/left (0 = stop).
- **U1/D1 \<steps\>** — axis 2 up/down (0 = stop).

The Python **legacy_motor_adapter** sends these commands. The GUI “Delay (ms)” is mapped to Arduino SPEED (50–5000 us) so higher delay = slower motor; the same value is used as the Python sleep between automation steps.

## Safety

- Global STOP always available; mode switch sends STOP ALL first.
- Serial disconnect triggers STOP and disables controls.
- Focus-out sends STOP when moving.

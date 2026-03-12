"""
Translates new app commands (START/STOP/STEP) into existing Arduino language:
R1 10, L1 10, U1 10, D1 10 for move; R1 0, L1 0, ... for stop.
Drop-in replacement for SerialManager when using current Arduino firmware.
Delays between commands so Arduino can process each one; stop sent multiple times.
"""
import threading
import time
from typing import Optional, Callable

import serial

from motor_app.serial_manager import ParsedResponse
from motor_app.config import (
    AXIS_1, AXIS_2, DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN,
    MIN_DELAY_MS, MAX_DELAY_MS,
    MIN_MANUAL_SPEED, MAX_MANUAL_SPEED,
    SERIAL_TIMEOUT, SERIAL_WRITE_TIMEOUT,
)


class LegacyMotorAdapter:
    """
    Presents the same connect/send/is_open interface as SerialManager but sends
    legacy commands: R1/L1/U1/D1 <n> with newline. No response parsing (get_response returns None).
    Sends repeated move commands while button is held for responsive control.
    """
    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        timeout: float = SERIAL_TIMEOUT,
        write_timeout: float = SERIAL_WRITE_TIMEOUT,
        on_disconnect: Optional[Callable[[], None]] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.write_timeout = write_timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._on_disconnect = on_disconnect
        # Legacy: send large move command for continuous movement (toggled on/off)
        self._move_distance = 1000  # Large value for continuous movement until stopped
        # No continuous movement thread needed - single command approach
        self._last_direction = None

    def is_open(self) -> bool:
        with self._lock:
            return self._ser is not None and self._ser.is_open

    def connect(self) -> bool:
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                return True
            try:
                self._ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=self.timeout,
                    write_timeout=self.write_timeout,
                )
                return True
            except Exception:
                return False

    def disconnect(self) -> None:
        # Clear buffers before closing
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                try:
                    self._ser.reset_input_buffer()
                    self._ser.reset_output_buffer()
                except Exception:
                    pass
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
        if self._on_disconnect:
            try:
                self._on_disconnect()
            except Exception:
                pass

    def _send_line(self, line: str) -> bool:
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                return False
            try:
                self._ser.write((line.strip() + "\n").encode("ascii"))
                self._ser.flush()
                time.sleep(0.03)  # Reduced delay for faster response
                return True
            except Exception:
                return False

    def send_ping(self) -> bool:
        return self._send_line("PING")

    def send_speed(self, delay_ms: int) -> bool:
        """Send SPEED command to Arduino. Arduino expects microseconds (50-5000). Higher = slower.
        Map GUI delay (MIN_DELAY_MS to MAX_DELAY_MS) to delay_us 50-5000."""
        ms = max(MIN_DELAY_MS, min(MAX_DELAY_MS, delay_ms)) if delay_ms is not None else 500
        delay_us = 50 + int((ms - MIN_DELAY_MS) * (5000 - 50) / max(1, MAX_DELAY_MS - MIN_DELAY_MS))
        delay_us = max(50, min(5000, delay_us))
        return self._send_line(f"SPEED {delay_us}")

    def send_stop_all(self) -> bool:
        """Stop both axes: clear buffers first, then send stop commands."""
        # Clear Arduino's serial buffers to discard any queued commands
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                try:
                    self._ser.reset_input_buffer()
                    self._ser.reset_output_buffer()
                except Exception:
                    pass
        
        # Axis 1: R1/L1; Axis 2: U2/D2 (Arduino expects axis number in command)
        commands = ("R1 0", "L1 0", "U2 0", "D2 0")
        ok = True
        for _ in range(3):  # Send multiple times for reliability
            for cmd in commands:
                if not self._send_line(cmd):
                    ok = False
        return ok

    def send_stop(self, axis: str, direction: Optional[str] = None) -> bool:
        """Stop: clear buffers first, then send stop commands."""
        # Clear Arduino's serial buffers to discard any queued commands
        with self._lock:
            if self._ser is not None and self._ser.is_open:
                try:
                    self._ser.reset_input_buffer()
                    self._ser.reset_output_buffer()
                except Exception:
                    pass
        
        # Determine stop command based on last direction. Axis 1: R1/L1; Axis 2: U2/D2.
        first_cmd = None
        if direction == DIR_RIGHT:
            first_cmd = "R1 0"
        elif direction == DIR_LEFT:
            first_cmd = "L1 0"
        elif direction == DIR_UP:
            first_cmd = "U2 0"
        elif direction == DIR_DOWN:
            first_cmd = "D2 0"
        
        # Send stop commands - prioritize the moving direction
        commands = ("R1 0", "L1 0", "U2 0", "D2 0")
        if first_cmd:
            order = [first_cmd] + [c for c in commands if c != first_cmd]
        else:
            order = list(commands)
        
        ok = True
        for _ in range(3):  # Send multiple times for reliability
            for cmd in order:
                if not self._send_line(cmd):
                    ok = False
        return ok

    def send_start(self, axis: str, direction: str, speed: int) -> bool:
        """Manual move: set SPEED from slider (bar = speed only), then send large fixed step count so motor runs until stop."""
        # Bar = speed only: map slider (MIN_MANUAL_SPEED-MAX_MANUAL_SPEED) to Arduino 50-5000 us (higher = slower)
        speed_val = max(MIN_MANUAL_SPEED, min(MAX_MANUAL_SPEED, speed)) if speed is not None else 50
        delay_us = 5000 - int((speed_val - MIN_MANUAL_SPEED) * (5000 - 50) / max(1, MAX_MANUAL_SPEED - MIN_MANUAL_SPEED))
        delay_us = max(50, min(5000, delay_us))
        self._send_line(f"SPEED {delay_us}")
        # Large fixed step count so hold-to-move runs until user releases (not tied to bar)
        dist = 30000
        command = None
        if axis == AXIS_1:
            if direction == DIR_RIGHT:
                command = f"R1 {dist}"
            elif direction == DIR_LEFT:
                command = f"L1 {dist}"
        elif axis == AXIS_2:
            if direction == DIR_UP:
                command = f"U2 {dist}"
            elif direction == DIR_DOWN:
                command = f"D2 {dist}"
        
        if command is None:
            return False
        
        self._last_direction = direction
        return self._send_line(command)

    def send_update_speed(self, axis: str, speed: int) -> bool:
        """Speed changes require stopping and restarting (not supported in toggle mode)."""
        return True  # No-op in toggle mode

    def send_step(self, axis: str, direction: str, step_size: int) -> bool:
        """One step: send R1 <step_size> or L1 <step_size> etc."""
        if axis == AXIS_1:
            if direction == DIR_RIGHT:
                return self._send_line(f"R1 {step_size}")
            if direction == DIR_LEFT:
                return self._send_line(f"L1 {step_size}")
        if axis == AXIS_2:
            if direction == DIR_UP:
                return self._send_line(f"U2 {step_size}")
            if direction == DIR_DOWN:
                return self._send_line(f"D2 {step_size}")
        return False

    def send_read_sensors(self) -> bool:
        """No-op for legacy; sensors come from separate sensor port."""
        return True

    def get_response(self, block: bool = False, timeout: float = 0.1) -> Optional[ParsedResponse]:
        """Legacy Arduino does not use ACK/MOTION_DONE protocol; return None."""
        return None

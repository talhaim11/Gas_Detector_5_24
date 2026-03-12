"""
Serial communication manager. Handles protocol framing and non-blocking I/O.
All motion commands require acknowledgment (handled by motor_controller / Arduino).
"""
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Any
import serial

from motor_app.config import (
    CMD_PING, CMD_STOP_ALL, CMD_START, CMD_STOP, CMD_UPDATE_SPEED, CMD_STEP, CMD_READ_SENSORS,
    RSP_ACK, RSP_ERR, RSP_MOTION_DONE, RSP_SENSORS,
    SERIAL_TIMEOUT, SERIAL_WRITE_TIMEOUT,
)


@dataclass
class ParsedResponse:
    """Parsed line from Arduino."""
    kind: str  # ACK, ERR, MOTION_DONE, SENSORS
    raw: str = ""
    payload: List[str] = field(default_factory=list)
    # For MOTION_DONE: payload = [axis, position]
    # For SENSORS: payload = [timestamp, position, val1, val2, ...]
    # For ACK: payload = [command_echo]
    # For ERR: payload = [code] or [code, message]


class SerialManager:
    """
    Manages serial connection and protocol. Runs a reader thread that pushes
    parsed responses into a queue. Send is synchronous (with timeout).
    """
    LINE_TERM = b"\r\n"
    LINE_TERM_ALT = b"\n"
    RECV_BUFFER_SIZE = 1024

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
        self._reader_thread: Optional[threading.Thread] = None
        self._running = False
        self._on_disconnect = on_disconnect
        self._response_queue: queue.Queue[ParsedResponse] = queue.Queue()
        self._line_buffer = bytearray()

    def is_open(self) -> bool:
        with self._lock:
            return self._ser is not None and self._ser.is_open

    def connect(self) -> bool:
        """Open serial port and start reader thread. Returns True on success."""
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
                self._running = True
                self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._reader_thread.start()
                return True
            except Exception:
                return False

    def disconnect(self) -> None:
        """Close port and stop reader. Safe to call multiple times."""
        self._running = False
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

    def send_line(self, line: str) -> bool:
        """
        Send a single line (no newline in line; we add \\r\\n).
        Returns True if written successfully. Thread-safe.
        """
        if not line.strip():
            return False
        with self._lock:
            if self._ser is None or not self._ser.is_open:
                return False
            try:
                to_send = (line.strip() + "\r\n").encode("ascii", errors="replace")
                self._ser.write(to_send)
                self._ser.flush()
                return True
            except Exception:
                return False

    # --- Protocol: Commands from PC (see config for exact strings) ---

    def send_ping(self) -> bool:
        return self.send_line(CMD_PING)

    def send_stop_all(self) -> bool:
        return self.send_line(CMD_STOP_ALL)

    def send_stop(self, axis: str, direction: Optional[str] = None) -> bool:
        return self.send_line(f"{CMD_STOP} {axis}")

    def send_start(self, axis: str, direction: str, speed: int) -> bool:
        return self.send_line(f"{CMD_START} {axis} {direction} {speed}")

    def send_update_speed(self, axis: str, speed: int) -> bool:
        return self.send_line(f"{CMD_UPDATE_SPEED} {axis} {speed}")

    def send_step(self, axis: str, direction: str, step_size: int) -> bool:
        return self.send_line(f"{CMD_STEP} {axis} {direction} {step_size}")

    def send_read_sensors(self) -> bool:
        return self.send_line(CMD_READ_SENSORS)

    # --- Response handling ---

    def get_response(self, block: bool = False, timeout: float = 0.1) -> Optional[ParsedResponse]:
        """Get next parsed response from queue. Non-blocking by default."""
        try:
            if block:
                return self._response_queue.get(timeout=timeout)
            return self._response_queue.get_nowait()
        except queue.Empty:
            return None

    def _reader_loop(self) -> None:
        """Background: read bytes, split lines, parse and put ParsedResponse."""
        while self._running:
            ser = None
            with self._lock:
                ser = self._ser
            if ser is None or not ser.is_open:
                time.sleep(0.05)
                continue
            try:
                if ser.in_waiting:
                    data = ser.read(min(ser.in_waiting, self.RECV_BUFFER_SIZE))
                    self._line_buffer.extend(data)
                else:
                    time.sleep(0.01)
            except Exception:
                break
            self._flush_lines()

        self.disconnect()

    def _flush_lines(self) -> None:
        """Process complete lines from _line_buffer and push ParsedResponse."""
        while True:
            idx = self._line_buffer.find(self.LINE_TERM)
            term_len = len(self.LINE_TERM)
            if idx == -1:
                idx = self._line_buffer.find(self.LINE_TERM_ALT)
                term_len = len(self.LINE_TERM_ALT)
            if idx == -1:
                break
            line_bytes = bytes(self._line_buffer[:idx])
            del self._line_buffer[: idx + term_len]
            line = line_bytes.decode("ascii", errors="replace").strip()
            if not line:
                continue
            parsed = self._parse_line(line)
            if parsed:
                self._response_queue.put(parsed)

    def _parse_line(self, line: str) -> Optional[ParsedResponse]:
        """Turn one line from Arduino into ParsedResponse. TODO: align with actual Arduino format."""
        parts = line.split()
        if not parts:
            return None
        kind = parts[0].upper()
        payload = parts[1:] if len(parts) > 1 else []
        if kind == RSP_ACK:
            return ParsedResponse(kind=RSP_ACK, raw=line, payload=payload)
        if kind == RSP_ERR:
            return ParsedResponse(kind=RSP_ERR, raw=line, payload=payload)
        if kind == RSP_MOTION_DONE:
            return ParsedResponse(kind=RSP_MOTION_DONE, raw=line, payload=payload)
        if kind == RSP_SENSORS:
            return ParsedResponse(kind=RSP_SENSORS, raw=line, payload=payload)
        # Unknown; still push so caller can log
        return ParsedResponse(kind="UNKNOWN", raw=line, payload=parts)

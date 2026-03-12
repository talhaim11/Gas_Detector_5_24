"""
Live sensor reader: 16-byte packets (0xFF), Ch1 and Ch3.
Samples the ~6µs photon flash (once per 0.5s): accumulate samples, median of values > 70k.
Saturation > ~8M displayed as 'sat'. get_latest() -> ((ch1_millions, sat1), (ch3_millions, sat3)).
"""
import threading
import time
from datetime import datetime
from typing import Optional, List, Tuple
import serial

from motor_app.motor_controller import SensorReading


def _process_channel(channel_bytes: bytes) -> int:
    """3-byte channel, byte swap, big-endian (matches reference)."""
    if len(channel_bytes) < 3:
        return 0
    swapped = channel_bytes[2:3] + channel_bytes[1:2] + channel_bytes[0:1]
    return int.from_bytes(swapped, byteorder="big")


def _process_packet(packet: bytes) -> List[int]:
    """Packet: 16 bytes, first 4 header then 12 bytes = 4 channels x 3 bytes."""
    if len(packet) < 16:
        return []
    data = packet[4:16]
    return [
        _process_channel(data[i : i + 3])
        for i in range(0, 12, 3)
    ]


def _median_of_above(arr: list, threshold: int) -> float:
    """Median of elements > threshold. Empty -> 0."""
    above = [x for x in arr if x > threshold]
    if not above:
        return 0.0
    above.sort()
    return float(above[len(above) // 2])


# Flash sampling: ~500 samples/sec, process every N samples (align with ~0.5s flash period).
THRESHOLD_MIN = 70_000          # ignore tiny values when computing median
THRESHOLD_SATURATION = 8_000_000
DEFAULT_SAMPLES_PER_UPDATE = 500


class SensorReader:
    """
    Parses 16-byte packets, accumulates Ch1/Ch3, then median of values > 70k (flash sampling).
    Saturation > 8M: display as 'sat'. get_latest() -> ((m1, sat1), (m3, sat3)).
    """
    def __init__(self, port: str, baudrate: int = 230400, timeout: float = 0.01):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._c1_list: List[int] = []
        self._c3_list: List[int] = []
        self._ch1_med: float = 0.0
        self._ch3_med: float = 0.0
        self._ch1_sat: bool = False
        self._ch3_sat: bool = False
        self._last_ts: str = ""
        self._samples_per_update: int = DEFAULT_SAMPLES_PER_UPDATE

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
                )
                self._running = True
                self._thread = threading.Thread(target=self._reader_loop, daemon=True)
                self._thread.start()
                return True
            except Exception:
                return False

    def disconnect(self) -> None:
        self._running = False
        with self._lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
        self._c1_list.clear()
        self._c3_list.clear()
        self._ch1_med = 0.0
        self._ch3_med = 0.0
        self._ch1_sat = False
        self._ch3_sat = False

    def set_samples_per_update(self, n: int) -> None:
        """Adjust how many packets are used per median update (controls responsiveness vs noise)."""
        with self._lock:
            n = max(50, min(5000, int(n or DEFAULT_SAMPLES_PER_UPDATE)))
            self._samples_per_update = n

    def _reader_loop(self) -> None:
        while self._running:
            ser = None
            with self._lock:
                ser = self._ser
            if ser is None or not ser.is_open:
                time.sleep(0.05)
                continue
            try:
                n = ser.in_waiting
                if n >= 16:
                    data = ser.read(min(n, 1024))
                    packets = [
                        data[i : i + 16]
                        for i in range(len(data) - 15)
                        if data[i] == 0xFF and len(data[i : i + 16]) == 16
                    ]
                    for pkt in packets:
                        chs = _process_packet(pkt)
                        if len(chs) >= 3:
                            self._c1_list.append(chs[0])
                            self._c3_list.append(chs[2])
                    if len(self._c1_list) >= self._samples_per_update:
                        c1_med = _median_of_above(self._c1_list, THRESHOLD_MIN)
                        c3_med = _median_of_above(self._c3_list, THRESHOLD_MIN)
                        low1 = c1_med == 0.0
                        low3 = c3_med == 0.0
                        # Use Ch3 as sync: if Ch3 says "no signal", force Ch1 to "no signal" too.
                        if low3:
                            low1 = True
                        self._ch1_sat = (not low1) and (c1_med > THRESHOLD_SATURATION)
                        self._ch3_sat = (not low3) and (c3_med > THRESHOLD_SATURATION)
                        if self._ch1_sat or low1:
                            c1_med = 0.0
                        if self._ch3_sat or low3:
                            c3_med = 0.0
                        self._ch1_med = c1_med
                        self._ch3_med = c3_med
                        self._c1_list = []
                        self._c3_list = []
                        self._last_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            except Exception:
                break
            time.sleep(0.01)

    def get_latest(self) -> Tuple[Tuple[float, bool], Tuple[float, bool]]:
        """Return ((ch1_millions, ch1_saturated), (ch3_millions, ch3_saturated)) for display. Saturated -> show 'sat'."""
        with self._lock:
            m1 = (self._ch1_med / 1e6) if self._ch1_med > 0 else 0.0
            m3 = (self._ch3_med / 1e6) if self._ch3_med > 0 else 0.0
            # Final safeguard: if Ch3 is effectively zero, force Ch1 display to zero as well.
            if m3 == 0.0:
                m1 = 0.0
            return ((m1, self._ch1_sat and m3 > 0.0), (m3, self._ch3_sat))

    def get_latest_reading(self, position: str = "0") -> SensorReading:
        """Return SensorReading for automation CSV: timestamp, position, [ch1_m, ch3_m] (0 when saturated)."""
        with self._lock:
            ts = self._last_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            m1 = (self._ch1_med / 1e6) if self._ch1_med > 0 else 0.0
            m3 = (self._ch3_med / 1e6) if self._ch3_med > 0 else 0.0
            return SensorReading(timestamp=ts, position=position, values=[m1, m3])

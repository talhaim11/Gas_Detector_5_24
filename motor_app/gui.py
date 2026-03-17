"""
GUI: Manual / Automation modes, live sensor display, motor controls.
Event-driven; serial processing on timer. Global STOP always active.
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, font as tkfont
from typing import Optional, Callable, Tuple
from collections import deque
import threading
import time
from datetime import datetime
from serial.tools import list_ports

from motor_app.state_machine import AppState, MotorStateMachine
from motor_app.motor_controller import MotorController
from motor_app.data_logger import DataLogger
from motor_app.legacy_motor_adapter import LegacyMotorAdapter
from motor_app.sensor_reader import SensorReader
from motor_app.config import (
    DEFAULT_MOTOR_PORT, DEFAULT_MOTOR_BAUD,
    DEFAULT_SENSOR_PORT, DEFAULT_SENSOR_BAUD,
    AXIS_1, AXIS_2, DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN,
    DEFAULT_MANUAL_SPEED, MIN_MANUAL_SPEED, MAX_MANUAL_SPEED,
    DEFAULT_STEP_SIZE, MIN_STEP_SIZE, MAX_STEP_SIZE,
    DEFAULT_NUM_STEPS, MIN_NUM_STEPS, MAX_NUM_STEPS,
    MIN_DELAY_MS, MAX_DELAY_MS, DEFAULT_DELAY_MS,
    DATA_SAVE_DIR,
)


# --- Live sensor visualization: simple graph + numeric ---
SENSOR_GRAPH_LEN = 100
SENSOR_UPDATE_MS = 150
SERIAL_POLL_MS = 50
AVG_SIGNAL_WINDOW = 5
AVG_APPEND_MS = 500
LIVE_RECORD_APPEND_MS = 200


# Channel display names
CHANNEL_LABELS = ("CH1", "CH2", "CH3", "CH4")
CHANNEL_COUNT_OPTIONS = ("1", "2", "4")
CONNECT_MODE_MOTOR_AND_SIGNAL = "Motor + Signal"
CONNECT_MODE_SIGNAL_ONLY = "Signal Only"


class SensorDisplay(ttk.Frame):
    """Scrolling graph for CH1-CH4. Values can be (float, is_sat)."""
    Y_MARGIN = 46
    X_MARGIN = 36
    AXIS_COLOR = "#888888"
    FONT_SIZE = 9

    def __init__(self, parent, num_channels: int = 2, **kw):
        super().__init__(parent, **kw)
        self.num_channels = num_channels
        self.buffers = [deque(maxlen=SENSOR_GRAPH_LEN) for _ in range(num_channels)]
        self.labels = []
        self.canvas = None
        self._build_ui()

    def _build_ui(self):
        self.canvas = tk.Canvas(self, height=200, width=600, bg="#1e1e1e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, pady=4)

    def update_values(self, values: list):
        """values: list of (float, bool) per channel: (value_millions, is_saturated). For graph use value (0 when sat)."""
        for i, item in enumerate(values):
            if i >= self.num_channels:
                break
            if item is None:
                continue
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                x, is_sat = float(item[0]), bool(item[1])
            else:
                try:
                    x = float(item)
                except (TypeError, ValueError):
                    x = 0.0
            self.buffers[i].append(x)
        self._redraw_graph()

    def _redraw_graph(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 420
        h = self.canvas.winfo_height() or 140
        gx0 = self.Y_MARGIN
        gy0 = 10
        gx1 = w - 8
        gy1 = h - self.X_MARGIN
        gw = gx1 - gx0
        gh = gy1 - gy0
        colors = ["#00ff00", "#ffcc00", "#00aaff", "#ff6a00"]
        all_vals = []
        for ch in range(self.num_channels):
            if ch < len(self.buffers) and self.buffers[ch]:
                all_vals.extend(self.buffers[ch])
        if all_vals:
            mx = max(all_vals)
            mn = min(all_vals)
            if mx <= mn:
                mx = mn + 1
        else:
            mn, mx = 0.0, 1.0
        self.canvas.create_text(gx0 - 6, gy0, text=f"{mx:.2f}", anchor=tk.E, fill=self.AXIS_COLOR, font=("Consolas", self.FONT_SIZE))
        self.canvas.create_text(gx0 - 6, gy1, text=f"{mn:.2f}", anchor=tk.E, fill=self.AXIS_COLOR, font=("Consolas", self.FONT_SIZE))
        mid = (mn + mx) / 2
        self.canvas.create_text(gx0 - 6, (gy0 + gy1) / 2, text=f"{mid:.2f}", anchor=tk.E, fill=self.AXIS_COLOR, font=("Consolas", self.FONT_SIZE))
        self.canvas.create_text(gx0, gy1 + 14, text="0", anchor=tk.N, fill=self.AXIS_COLOR, font=("Consolas", self.FONT_SIZE))
        self.canvas.create_text(gx1, gy1 + 14, text="N", anchor=tk.N, fill=self.AXIS_COLOR, font=("Consolas", self.FONT_SIZE))
        self.canvas.create_text((gx0 + gx1) / 2, h - 4, text="samples", anchor=tk.N, fill=self.AXIS_COLOR, font=("Consolas", self.FONT_SIZE - 1))
        # Legend for sensor traces
        self.canvas.create_text(gx1 - 110, gy0 + 10, text="CH1", anchor=tk.W, fill="#00ff00", font=("Consolas", self.FONT_SIZE))
        self.canvas.create_text(gx1 - 110, gy0 + 24, text="CH2", anchor=tk.W, fill="#ffcc00", font=("Consolas", self.FONT_SIZE))
        self.canvas.create_text(gx1 - 60, gy0 + 10, text="CH3", anchor=tk.W, fill="#00aaff", font=("Consolas", self.FONT_SIZE))
        self.canvas.create_text(gx1 - 60, gy0 + 24, text="CH4", anchor=tk.W, fill="#ff6a00", font=("Consolas", self.FONT_SIZE))
        scale = gh / (mx - mn) if (mx - mn) > 0 else 1
        for ch, color in enumerate(colors):
            if ch >= len(self.buffers) or not self.buffers[ch]:
                continue
            buf = list(self.buffers[ch])
            if not buf:
                continue
            ox = gw / max(len(buf) - 1, 1)
            points = []
            for i, v in enumerate(buf):
                x = gx0 + i * ox
                y = gy1 - (v - mn) * scale
                points.append((x, y))
            if len(points) >= 2:
                self.canvas.create_line(points, fill=color, width=1)


class MainApp(ttk.Frame):
    """
    Main window: mode switch, Manual controls (direction + speed + STOP),
    Automation controls (step size, num steps, Scan L/R, progress, STOP),
    live sensor display, global STOP.
    """
    def __init__(
        self,
        parent,
        motor_port: str = DEFAULT_MOTOR_PORT,
        motor_baud: int = DEFAULT_MOTOR_BAUD,
        sensor_port: str = DEFAULT_SENSOR_PORT,
        sensor_baud: int = DEFAULT_SENSOR_BAUD,
        **kw,
    ):
        super().__init__(parent, **kw)
        self.motor_port = motor_port
        self.motor_baud = motor_baud
        self.sensor_port = sensor_port
        self.sensor_baud = sensor_baud
        self._connection_mode_var = tk.StringVar(value=CONNECT_MODE_MOTOR_AND_SIGNAL)
        self._channel_count_var = tk.StringVar(value="1")
        self._motor_port_var = tk.StringVar(value=motor_port)
        self._sensor_port_var = tk.StringVar(value=sensor_port)
        self._motor_adapter: Optional[LegacyMotorAdapter] = None
        self._sensor_reader: Optional[SensorReader] = None
        self._controller: Optional[MotorController] = None
        self._logger = DataLogger()
        self._manual_speed_var = tk.IntVar(value=DEFAULT_MANUAL_SPEED)
        self._step_size_var = tk.IntVar(value=DEFAULT_STEP_SIZE)
        self._num_steps_var = tk.IntVar(value=DEFAULT_NUM_STEPS)
        self._progress_var = tk.StringVar(value="Idle")
        self._state_var = tk.StringVar(value=AppState.DISCONNECTED.name)
        # Live channels / ratios
        self._ratio_var = tk.StringVar(value="--")
        self._ratio2_var = tk.StringVar(value="--")
        self._channel_value_vars = {ch: tk.StringVar(value="off") for ch in range(1, 5)}
        self._channel_avg_vars = {ch: tk.StringVar(value="none") for ch in range(1, 5)}
        self._channel_avg_buffers = {ch: [] for ch in range(1, 5)}
        self._last_avg_append_ts = 0.0
        self._record_rate_var = tk.StringVar(value="0.0 Hz")
        self._record_rate_times = deque()
        self._last_live_record_ts = 0.0
        self._sample_window_var = tk.IntVar(value=500)
        # Continuous live recording
        self._live_recording = False
        self._live_records = []
        self._ratio_cal_ref: Optional[float] = None
        self._ratio_measurements = []  # list of dicts
        self._latest_channels = {ch: 0.0 for ch in range(1, 5)}
        self._latest_sats = {ch: False for ch in range(1, 5)}
        self._last_ratio_norm: Optional[float] = None
        self._current_ratio_raw: Optional[float] = None
        self._current_ratio2_raw: Optional[float] = None
        self._last_scan_direction = DIR_RIGHT
        self._last_scan_step_size = DEFAULT_STEP_SIZE
        self._auto_delay_var = tk.IntVar(value=DEFAULT_DELAY_MS)
        self._auto_delay_var.trace_add("write", self._on_delay_change)  # Send SPEED command when change
        self._connection_mode_combo = None
        self._channel_count_combo = None
        self._motor_port_combo = None
        self._sensor_port_combo = None
        self._build_ui()
        self._refresh_com_ports()
        self._on_connection_mode_change()
        self._on_channel_count_change()
        self._after_id = None
        self._movement_timer_1 = None  # Track movement timeout for axis 1
        self._movement_timer_2 = None  # Track movement timeout for axis 2
        self._bind_focus_stop()
        self._bind_keyboard_controls()

    def _build_ui(self):
        # Top row 1: COM selectors + Connect
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=4, pady=(4, 1))
        ttk.Label(top, text="Connect mode:").pack(side=tk.LEFT, padx=(0, 2))
        self._connection_mode_combo = ttk.Combobox(
            top,
            textvariable=self._connection_mode_var,
            values=(CONNECT_MODE_MOTOR_AND_SIGNAL, CONNECT_MODE_SIGNAL_ONLY),
            width=14,
            state="readonly",
        )
        self._connection_mode_combo.pack(side=tk.LEFT, padx=(0, 6))
        self._connection_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_connection_mode_change())
        ttk.Label(top, text="Motor COM:").pack(side=tk.LEFT, padx=(0, 2))
        self._motor_port_combo = ttk.Combobox(top, textvariable=self._motor_port_var, width=8, state="readonly")
        self._motor_port_combo.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(top, text="Sensor COM:").pack(side=tk.LEFT, padx=(0, 2))
        self._sensor_port_combo = ttk.Combobox(top, textvariable=self._sensor_port_var, width=8, state="readonly")
        self._sensor_port_combo.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Label(top, text="Channels:").pack(side=tk.LEFT, padx=(0, 2))
        self._channel_count_combo = ttk.Combobox(
            top,
            textvariable=self._channel_count_var,
            values=CHANNEL_COUNT_OPTIONS,
            width=3,
            state="readonly",
        )
        self._channel_count_combo.pack(side=tk.LEFT, padx=(0, 6))
        self._channel_count_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_channel_count_change())
        ttk.Button(top, text="Refresh COM", command=self._refresh_com_ports).pack(side=tk.LEFT, padx=2)

        # Top row 2: State + Connect
        top2 = ttk.Frame(self)
        top2.pack(fill=tk.X, padx=4, pady=(1, 4))
        ttk.Label(top2, text="State:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(top2, textvariable=self._state_var, font=("Consolas", 10, "bold")).pack(side=tk.LEFT, padx=2)
        ttk.Button(top2, text="Connect", command=self._on_connect).pack(side=tk.RIGHT, padx=2)

        # Main body: left = motor controls, right = live sensor data
        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        left_panel = ttk.Frame(body)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        right_panel = ttk.Frame(body)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # Mode switch (left panel)
        mode_f = ttk.LabelFrame(left_panel, text="Mode")
        mode_f.pack(fill=tk.X, pady=4)
        self._manual_mode_btn = ttk.Button(mode_f, text="Manual (Hold-to-Move)", command=self._on_manual_mode)
        self._manual_mode_btn.pack(side=tk.LEFT, padx=2, pady=2)
        self._auto_mode_btn = ttk.Button(mode_f, text="Automation (Step Scan)", command=self._on_automation_mode)
        self._auto_mode_btn.pack(side=tk.LEFT, padx=2, pady=2)
        stop_btn = ttk.Button(mode_f, text="(space) stop", command=self._on_stop_all)
        stop_btn.pack(side=tk.RIGHT, padx=4, pady=2)
        self._stop_btn = stop_btn

        # Manual controls - Axis 1 (Left/Right)
        manual_f = ttk.LabelFrame(left_panel, text="Manual: Axis 1 (Left/Right)")
        manual_f.pack(fill=tk.X, pady=4)
        dir_f = ttk.Frame(manual_f)
        dir_f.pack(anchor=tk.W)
        self._btn_left = ttk.Button(dir_f, text="◄ Left", width=8)
        self._btn_left.pack(side=tk.LEFT, padx=2)
        self._btn_left.bind("<ButtonPress-1>", lambda e: self._on_dir_btn_press(AXIS_1, DIR_LEFT))
        self._btn_left.bind("<ButtonRelease-1>", lambda e: self._on_dir_btn_release(AXIS_1, DIR_LEFT))
        self._btn_left.bind("<Leave>", lambda e: self._on_dir_btn_leave(AXIS_1, DIR_LEFT))
        self._btn_right = ttk.Button(dir_f, text="Right ►", width=8)
        self._btn_right.pack(side=tk.LEFT, padx=2)
        self._btn_right.bind("<ButtonPress-1>", lambda e: self._on_dir_btn_press(AXIS_1, DIR_RIGHT))
        self._btn_right.bind("<ButtonRelease-1>", lambda e: self._on_dir_btn_release(AXIS_1, DIR_RIGHT))
        self._btn_right.bind("<Leave>", lambda e: self._on_dir_btn_leave(AXIS_1, DIR_RIGHT))
        ttk.Label(dir_f, text="  Use Arrow Keys ← → or hold buttons", font=("Consolas", 8)).pack(side=tk.LEFT, padx=20)
        self._moving_direction_1 = None

        # Manual controls - Axis 2 (Up/Down)
        manual2_f = ttk.LabelFrame(left_panel, text="Manual: Axis 2 (Up/Down)")
        manual2_f.pack(fill=tk.X, pady=4)
        dir2_f = ttk.Frame(manual2_f)
        dir2_f.pack(anchor=tk.W)
        self._btn_up = ttk.Button(dir2_f, text="▲ Up", width=8)
        self._btn_up.pack(side=tk.LEFT, padx=2)
        self._btn_up.bind("<ButtonPress-1>", lambda e: self._on_dir_btn_press(AXIS_2, DIR_UP))
        self._btn_up.bind("<ButtonRelease-1>", lambda e: self._on_dir_btn_release(AXIS_2, DIR_UP))
        self._btn_up.bind("<Leave>", lambda e: self._on_dir_btn_leave(AXIS_2, DIR_UP))
        self._btn_down = ttk.Button(dir2_f, text="Down ▼", width=8)
        self._btn_down.pack(side=tk.LEFT, padx=2)
        self._btn_down.bind("<ButtonPress-1>", lambda e: self._on_dir_btn_press(AXIS_2, DIR_DOWN))
        self._btn_down.bind("<ButtonRelease-1>", lambda e: self._on_dir_btn_release(AXIS_2, DIR_DOWN))
        self._btn_down.bind("<Leave>", lambda e: self._on_dir_btn_leave(AXIS_2, DIR_DOWN))
        ttk.Label(dir2_f, text="  Use Arrow Keys ↑ ↓ or hold buttons", font=("Consolas", 8)).pack(side=tk.LEFT, padx=20)
        self._moving_direction_2 = None
        speed_f = ttk.Frame(manual_f)
        speed_f.pack(anchor=tk.W, pady=4)
        ttk.Label(speed_f, text="Speed:").pack(side=tk.LEFT, padx=(0, 4))
        speed_slider = ttk.Scale(
            speed_f, from_=MIN_MANUAL_SPEED, to=MAX_MANUAL_SPEED,
            variable=self._manual_speed_var, orient=tk.HORIZONTAL, length=200,
            command=lambda v: self._on_manual_speed_change(int(float(v))),
        )
        speed_slider.pack(side=tk.LEFT, padx=4)
        ttk.Label(speed_f, textvariable=self._manual_speed_var, width=4).pack(side=tk.LEFT)

        # Automation controls
        auto_f = ttk.LabelFrame(left_panel, text="Automation: Step Scan")
        auto_f.pack(fill=tk.X, pady=4)
        row1 = ttk.Frame(auto_f)
        row1.pack(anchor=tk.W)
        ttk.Label(row1, text="Step size:").pack(side=tk.LEFT, padx=2)
        tk.Spinbox(row1, from_=MIN_STEP_SIZE, to=MAX_STEP_SIZE, width=6, textvariable=self._step_size_var).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="Num steps:").pack(side=tk.LEFT, padx=(8, 2))
        tk.Spinbox(row1, from_=MIN_NUM_STEPS, to=MAX_NUM_STEPS, width=6, textvariable=self._num_steps_var).pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="Delay (ms):").pack(side=tk.LEFT, padx=(8, 2))
        tk.Spinbox(row1, from_=MIN_DELAY_MS, to=MAX_DELAY_MS, width=6, textvariable=self._auto_delay_var).pack(side=tk.LEFT, padx=2)
        row2 = ttk.Frame(auto_f)
        row2.pack(anchor=tk.W, pady=4)
        # Scan buttons use same physical axes as manual controls.
        self._scan_up_btn = ttk.Button(row2, text="Scan Up", command=lambda: self._start_scan(DIR_UP))
        self._scan_up_btn.pack(side=tk.LEFT, padx=2)
        self._scan_down_btn = ttk.Button(row2, text="Scan Down", command=lambda: self._start_scan(DIR_DOWN))
        self._scan_down_btn.pack(side=tk.LEFT, padx=2)
        self._scan_left_btn = ttk.Button(row2, text="Scan Left", command=lambda: self._start_scan(DIR_LEFT))
        self._scan_left_btn.pack(side=tk.LEFT, padx=2)
        self._scan_right_btn = ttk.Button(row2, text="Scan Right", command=lambda: self._start_scan(DIR_RIGHT))
        self._scan_right_btn.pack(side=tk.LEFT, padx=2)
        ttk.Label(auto_f, text="Progress:").pack(anchor=tk.W, padx=2)
        ttk.Label(auto_f, textvariable=self._progress_var, font=("Consolas", 9)).pack(anchor=tk.W, padx=2)

        # Activity log (left panel, below automation)
        log_f = ttk.LabelFrame(left_panel, text="Activity Log")
        log_f.pack(fill=tk.BOTH, expand=True, pady=4)
        _log_container = ttk.Frame(log_f)
        _log_container.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._activity_log = tk.Text(
            _log_container, height=6, state="disabled",
            font=("Consolas", 8), bg="#1e1e1e", fg="#cccccc", wrap=tk.WORD,
        )
        _log_sb = ttk.Scrollbar(_log_container, orient=tk.VERTICAL, command=self._activity_log.yview)
        self._activity_log.configure(yscrollcommand=_log_sb.set)
        _log_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._activity_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Live sensor (right panel)
        sensor_f = ttk.LabelFrame(right_panel, text="Live Sensor Data")
        sensor_f.pack(fill=tk.BOTH, expand=True, pady=4)

        # Indicators box: ratios + channel values
        indicators_f = ttk.LabelFrame(sensor_f, text="Indicators")
        indicators_f.pack(fill=tk.X, padx=4, pady=(4, 2))
        top_row = ttk.Frame(indicators_f)
        top_row.pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(top_row, text="Ratio1 (CH1/CH3):").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(top_row, textvariable=self._ratio_var, font=("Consolas", 10, "bold"), width=8).pack(side=tk.LEFT)
        ttk.Label(top_row, text="Ratio2 (CH2/CH4):").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(top_row, textvariable=self._ratio2_var, font=("Consolas", 10, "bold"), width=8).pack(side=tk.LEFT)
        ch_row1 = ttk.Frame(indicators_f)
        ch_row1.pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(ch_row1, text="CH1:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(ch_row1, textvariable=self._channel_value_vars[1], font=("Consolas", 9), width=8).pack(side=tk.LEFT)
        ttk.Label(ch_row1, text="CH2:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(ch_row1, textvariable=self._channel_value_vars[2], font=("Consolas", 9), width=8).pack(side=tk.LEFT)
        ch_row2 = ttk.Frame(indicators_f)
        ch_row2.pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(ch_row2, text="CH3:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(ch_row2, textvariable=self._channel_value_vars[3], font=("Consolas", 9), width=8).pack(side=tk.LEFT)
        ttk.Label(ch_row2, text="CH4:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(ch_row2, textvariable=self._channel_value_vars[4], font=("Consolas", 9), width=8).pack(side=tk.LEFT)
        avg_row1 = ttk.Frame(indicators_f)
        avg_row1.pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(avg_row1, text="AVG CH1:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(avg_row1, textvariable=self._channel_avg_vars[1], font=("Consolas", 9), width=8).pack(side=tk.LEFT)
        ttk.Label(avg_row1, text="AVG CH2:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(avg_row1, textvariable=self._channel_avg_vars[2], font=("Consolas", 9), width=8).pack(side=tk.LEFT)
        avg_row2 = ttk.Frame(indicators_f)
        avg_row2.pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(avg_row2, text="AVG CH3:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(avg_row2, textvariable=self._channel_avg_vars[3], font=("Consolas", 9), width=8).pack(side=tk.LEFT)
        ttk.Label(avg_row2, text="AVG CH4:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(avg_row2, textvariable=self._channel_avg_vars[4], font=("Consolas", 9), width=8).pack(side=tk.LEFT)

        # Controls / legend box
        controls_f = ttk.LabelFrame(sensor_f, text="Controls")
        controls_f.pack(fill=tk.X, padx=4, pady=(2, 4))
        btn_row = ttk.Frame(controls_f)
        btn_row.pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(btn_row, text="(c) Calibrate (→1)", command=self._on_calibrate_ratio).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="(a) Add point", command=self._on_add_ratio_measurement).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text="(s) Save XLSX", command=self._on_save_ratio_measurements).pack(side=tk.LEFT, padx=2)
        record_row = ttk.Frame(controls_f)
        record_row.pack(anchor=tk.W, pady=(0, 2))
        self._record_btn = ttk.Button(record_row, text="(r) Record", command=self._on_live_record_start)
        self._record_btn.pack(side=tk.LEFT, padx=2)
        self._stop_record_btn = ttk.Button(record_row, text="(t) Stop", command=self._on_live_record_stop, state="disabled")
        self._stop_record_btn.pack(side=tk.LEFT, padx=2)
        ttk.Label(record_row, text="Actual record rate:").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Label(record_row, textvariable=self._record_rate_var, font=("Consolas", 9, "bold"), width=8).pack(side=tk.LEFT)
        # Sample window (controls responsiveness vs noise)
        sample_row = ttk.Frame(controls_f)
        sample_row.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(sample_row, text="Samples / update:").pack(side=tk.LEFT, padx=(0, 4))
        sample_slider = ttk.Scale(
            sample_row,
            from_=100,
            to=2000,
            orient=tk.HORIZONTAL,
            length=160,
            variable=self._sample_window_var,
            command=lambda v: self._on_sample_window_change(int(float(v))),
        )
        sample_slider.pack(side=tk.LEFT, padx=2)
        ttk.Label(sample_row, textvariable=self._sample_window_var, width=5, font=("Consolas", 8)).pack(side=tk.LEFT, padx=(4, 0))

        # Second row for numeric entry so text doesn't get cropped by frame border
        sample_row2 = ttk.Frame(controls_f)
        sample_row2.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(sample_row2, text="Set:", font=("Consolas", 8)).pack(side=tk.LEFT, padx=(0, 2))
        self._sample_window_entry = ttk.Entry(sample_row2, width=6, font=("Consolas", 8))
        self._sample_window_entry.insert(0, str(self._sample_window_var.get()))
        self._sample_window_entry.pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(sample_row2, text="Apply", command=self._on_sample_window_apply).pack(side=tk.LEFT, padx=(2, 0))
        legend_row = ttk.Frame(controls_f)
        legend_row.pack(anchor=tk.W, pady=(0, 2))
        legend_row.pack_forget()

        # Live graph
        self._sensor_display = SensorDisplay(sensor_f, num_channels=4)
        self._sensor_display.pack(fill=tk.BOTH, expand=True, padx=2, pady=(2, 4))

    def _list_com_ports(self):
        ports = []
        try:
            ports = sorted([p.device for p in list_ports.comports()])
        except Exception:
            ports = []
        return ports

    def _refresh_com_ports(self):
        ports = self._list_com_ports()
        selected_motor = self._motor_port_var.get()
        selected_sensor = self._sensor_port_var.get()
        merged_ports = list(ports)
        for candidate in (selected_motor, selected_sensor, self.motor_port, self.sensor_port):
            if candidate and candidate not in merged_ports:
                merged_ports.append(candidate)
        merged_ports = sorted(set(merged_ports))
        if self._motor_port_combo is not None:
            self._motor_port_combo["values"] = merged_ports
        if self._sensor_port_combo is not None:
            self._sensor_port_combo["values"] = merged_ports
        if selected_motor:
            self._motor_port_var.set(selected_motor)
        elif merged_ports:
            self._motor_port_var.set(merged_ports[0])
        if selected_sensor:
            self._sensor_port_var.set(selected_sensor)
        elif merged_ports:
            self._sensor_port_var.set(merged_ports[0])

    def _on_connection_mode_change(self):
        signal_only = self._connection_mode_var.get() == CONNECT_MODE_SIGNAL_ONLY
        if self._motor_port_combo is not None:
            self._motor_port_combo.configure(state="disabled" if signal_only else "readonly")

    def _active_channels(self):
        mode = self._channel_count_var.get().strip()
        if mode == "1":
            return {3}
        if mode == "2":
            return {1, 3}
        return {1, 2, 3, 4}

    def _on_channel_count_change(self):
        active = self._active_channels()
        for ch in range(1, 5):
            if ch not in active:
                self._channel_avg_buffers[ch].clear()
                self._channel_value_vars[ch].set("off")
                self._channel_avg_vars[ch].set("off")
                if hasattr(self, "_sensor_display") and self._sensor_display is not None:
                    self._sensor_display.buffers[ch - 1].clear()
        if not ({1, 3} <= active):
            self._ratio_var.set("--")
        if not ({2, 4} <= active):
            self._ratio2_var.set("--")

    def _bind_keyboard_controls(self):
        """Bind keyboard keys for hold-to-move continuous control."""
        root = self.winfo_toplevel()
        root.bind("<KeyPress-Left>", self._on_key_press)
        root.bind("<KeyRelease-Left>", self._on_key_release)
        root.bind("<KeyPress-Right>", self._on_key_press)
        root.bind("<KeyRelease-Right>", self._on_key_release)
        root.bind("<KeyPress-Up>", self._on_key_press)
        root.bind("<KeyRelease-Up>", self._on_key_release)
        root.bind("<KeyPress-Down>", self._on_key_press)
        root.bind("<KeyRelease-Down>", self._on_key_release)
        root.bind("<KeyPress-space>", self._on_key_press)
        # Ratio shortcuts: c = calibrate, a = add point, s = save xlsx, r = record, t = stop
        root.bind("<KeyPress-c>", self._on_ratio_key)
        root.bind("<KeyPress-a>", self._on_ratio_key)
        root.bind("<KeyPress-s>", self._on_ratio_key)
        root.bind("<KeyPress-r>", self._on_ratio_key)
        root.bind("<KeyPress-t>", self._on_ratio_key)

    def _bind_focus_stop(self):
        """Bind FocusOut event to stop motor when window loses focus."""
        self.bind("<FocusOut>", self._on_focus_out)
    
    def _on_key_press(self, event):
        """Handle key press for continuous movement. Swapped: Up/Down = Axis 1, Left/Right = Axis 2. Space = STOP."""
        if event.keysym == "space":
            if self._controller:
                self._controller.stop_all()
            self._moving_direction_1 = None
            self._moving_direction_2 = None
            return
        if not self._controller or not self._controller.state_machine.can_manual_move():
            return
        speed = self._manual_speed_var.get()
        if event.keysym == "Left":
            self._controller.manual_start(AXIS_2, DIR_DOWN, speed)
            self._moving_direction_1 = DIR_LEFT
        elif event.keysym == "Right":
            self._controller.manual_start(AXIS_2, DIR_UP, speed)
            self._moving_direction_1 = DIR_RIGHT
        elif event.keysym == "Up":
            self._controller.manual_start(AXIS_1, DIR_RIGHT, speed)
            self._moving_direction_2 = DIR_UP
        elif event.keysym == "Down":
            self._controller.manual_start(AXIS_1, DIR_LEFT, speed)
            self._moving_direction_2 = DIR_DOWN
    
    def _on_key_release(self, event):
        """Handle key release to stop continuous movement."""
        if not self._controller:
            return
        if event.keysym == "Left" and self._moving_direction_1 == DIR_LEFT:
            self._controller.manual_stop()
            self._moving_direction_1 = None
        elif event.keysym == "Right" and self._moving_direction_1 == DIR_RIGHT:
            self._controller.manual_stop()
            self._moving_direction_1 = None
        elif event.keysym == "Up" and self._moving_direction_2 == DIR_UP:
            self._controller.manual_stop()
            self._moving_direction_2 = None
        elif event.keysym == "Down" and self._moving_direction_2 == DIR_DOWN:
            self._controller.manual_stop()
            self._moving_direction_2 = None

    def _on_dir_btn_press(self, axis: str, direction: str):
        """Handle direction button press. Swapped: Axis 1 (Left/Right) -> Axis 2, Axis 2 (Up/Down) -> Axis 1."""
        if not self._controller or not self._controller.state_machine.can_manual_move():
            return
        speed = self._manual_speed_var.get()
        # Swap: GUI Left/Right (Axis 1) -> motor Axis 2; GUI Up/Down (Axis 2) -> motor Axis 1
        if axis == AXIS_1:
            real_axis, real_dir = AXIS_2, (DIR_DOWN if direction == DIR_LEFT else DIR_UP)
        else:
            real_axis, real_dir = AXIS_1, (DIR_RIGHT if direction == DIR_UP else DIR_LEFT)
        self._controller.manual_start(real_axis, real_dir, speed)
        if axis == AXIS_1:
            self._moving_direction_1 = direction
        else:
            self._moving_direction_2 = direction

    def _on_dir_btn_release(self, axis: str, direction: str):
        """Handle direction button release: stop moving."""
        if not self._controller:
            return
        if axis == AXIS_1 and self._moving_direction_1 == direction:
            self._controller.manual_stop()
            self._moving_direction_1 = None
        elif axis == AXIS_2 and self._moving_direction_2 == direction:
            self._controller.manual_stop()
            self._moving_direction_2 = None

    def _on_dir_btn_leave(self, axis: str, direction: str):
        """Handle mouse leaving button: stop if this button was moving."""
        if not self._controller:
            return
        if axis == AXIS_1 and self._moving_direction_1 == direction:
            self._controller.manual_stop()
            self._moving_direction_1 = None
        elif axis == AXIS_2 and self._moving_direction_2 == direction:
            self._controller.manual_stop()
            self._moving_direction_2 = None

    def _on_focus_out(self, event):
        if self._controller and self._controller.state_machine.can_stop():
            self._controller.stop_all()
            self._moving_direction_1 = None
            self._moving_direction_2 = None

    def _on_connect(self):
        if (self._controller and self._controller.state_machine.is_connected()) or (
            self._sensor_reader and self._sensor_reader.is_open()
        ):
            return
        connect_mode = self._connection_mode_var.get()
        connect_motor = connect_mode != CONNECT_MODE_SIGNAL_ONLY
        self.motor_port = self._motor_port_var.get().strip() or self.motor_port
        self.sensor_port = self._sensor_port_var.get().strip() or self.sensor_port
        if connect_motor:
            self._motor_adapter = LegacyMotorAdapter(
                port=self.motor_port,
                baudrate=self.motor_baud,
                on_disconnect=self._on_serial_disconnect,
            )
            self._controller = MotorController(
                serial_manager=self._motor_adapter,
                on_state_change=self._on_state_change,
                on_sensor_reading=self._on_sensor_reading,
                on_automation_progress=self._on_automation_progress,
                on_automation_complete=self._on_automation_complete,
                on_error=self._on_error,
            )
            if not self._motor_adapter.connect():
                messagebox.showerror("Connection", f"Failed to open motor port {self.motor_port}")
                self._controller = None
                self._motor_adapter = None
                return
            if not self._controller.connect():
                messagebox.showerror("Connection", "Failed to start controller")
                try:
                    self._motor_adapter.disconnect()
                except Exception:
                    pass
                self._controller = None
                self._motor_adapter = None
                return
        else:
            self._controller = None
            self._motor_adapter = None
        self._sensor_reader = SensorReader(port=self.sensor_port, baudrate=self.sensor_baud)
        if not self._sensor_reader.connect():
            if connect_motor and self._controller:
                self._controller.disconnect()
                self._controller = None
                self._motor_adapter = None
            messagebox.showerror("Connection", f"Failed to open sensor port {self.sensor_port}")
            return
        try:
            self._sensor_reader.set_samples_per_update(self._sample_window_var.get())
        except Exception:
            pass
        if self._controller:
            self._state_var.set(self._controller.state.name)
        else:
            self._state_var.set("SIGNAL_ONLY")
        self._update_controls()
        self._start_poll()

    def _on_serial_disconnect(self):
        if self._controller:
            try:
                self._controller.state_machine.transition_to(AppState.DISCONNECTED, force=True)
            except ValueError:
                pass
        self._state_var.set(AppState.DISCONNECTED.name)
        self._update_controls()

    def _on_disconnect(self):
        if self._sensor_reader:
            self._sensor_reader.disconnect()
            self._sensor_reader = None
        if self._controller:
            self._controller.disconnect()
            self._controller = None
        self._motor_adapter = None
        self._state_var.set(AppState.DISCONNECTED.name)
        for ch in range(1, 5):
            self._channel_avg_buffers[ch].clear()
            self._channel_value_vars[ch].set("off")
            self._channel_avg_vars[ch].set("off")
            self._latest_channels[ch] = 0.0
            self._latest_sats[ch] = False
        self._last_avg_append_ts = 0.0
        self._record_rate_times.clear()
        self._last_live_record_ts = 0.0
        self._record_rate_var.set("0.0 Hz")
        self._ratio_var.set("--")
        self._ratio2_var.set("--")
        self._update_controls()
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def _start_poll(self):
        def poll():
            if self._controller:
                self._controller.process_responses()
            if self._sensor_reader and self._sensor_reader.is_open():
                ch_data = self._sensor_reader.get_latest()
                active = self._active_channels()
                for idx, (m, sat) in enumerate(ch_data, start=1):
                    self._latest_channels[idx] = m
                    self._latest_sats[idx] = sat
                # Ratio1 = CH1/CH3
                ratio_raw = None
                if {1, 3} <= active:
                    if (not self._latest_sats[1]) and (not self._latest_sats[3]) and self._latest_channels[1] > 0 and self._latest_channels[3] > 0:
                        ratio_raw = self._latest_channels[1] / self._latest_channels[3]
                self._current_ratio_raw = ratio_raw
                if ratio_raw is not None and self._ratio_cal_ref and self._ratio_cal_ref > 0:
                    ratio_norm = ratio_raw / self._ratio_cal_ref
                else:
                    ratio_norm = ratio_raw
                self._last_ratio_norm = ratio_norm
                if {1, 3} <= active:
                    self._ratio_var.set(f"{ratio_norm:.3f}" if ratio_norm is not None else "sat")
                else:
                    self._ratio_var.set("--")
                # Ratio2 = CH2/CH4
                ratio2_raw = None
                if {2, 4} <= active:
                    if (not self._latest_sats[2]) and (not self._latest_sats[4]) and self._latest_channels[2] > 0 and self._latest_channels[4] > 0:
                        ratio2_raw = self._latest_channels[2] / self._latest_channels[4]
                self._current_ratio2_raw = ratio2_raw
                self._ratio2_var.set(f"{ratio2_raw:.3f}" if ratio2_raw is not None else ("sat" if {2, 4} <= active else "--"))

                # Update channel indicators (respect saturation and active-channel mode)
                txt_by_ch = {}
                for ch in range(1, 5):
                    if ch not in active:
                        txt_by_ch[ch] = "off"
                        self._channel_value_vars[ch].set("off")
                        self._channel_avg_vars[ch].set("off")
                        self._channel_avg_buffers[ch].clear()
                        continue
                    v = self._latest_channels[ch]
                    sat = self._latest_sats[ch]
                    if sat:
                        txt = "sat"
                    elif v > 0:
                        txt = f"{v:.3f}"
                    else:
                        txt = "0"
                    txt_by_ch[ch] = txt
                    self._channel_value_vars[ch].set(txt)
                now = time.monotonic()
                if (now - self._last_avg_append_ts) * 1000 >= AVG_APPEND_MS:
                    self._last_avg_append_ts = now
                    for ch in active:
                        txt = txt_by_ch.get(ch, "0")
                        if txt not in ("sat", "0", "off"):
                            try:
                                self._channel_avg_buffers[ch].append(float(txt))
                            except ValueError:
                                pass
                for ch in active:
                    if len(self._channel_avg_buffers[ch]) == AVG_SIGNAL_WINDOW:
                        self._channel_avg_vars[ch].set(f"{(sum(self._channel_avg_buffers[ch]) / AVG_SIGNAL_WINDOW):.3f}")
                        self._channel_avg_buffers[ch].clear()
                    elif self._channel_avg_vars[ch].get() == "off":
                        self._channel_avg_vars[ch].set("none")
                # Continuous live recording: append one row per poll when enabled
                if self._live_recording:
                    if (now - self._last_live_record_ts) * 1000 >= LIVE_RECORD_APPEND_MS:
                        self._last_live_record_ts = now
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        ratio1_for_save = ratio_raw if {1, 3} <= active else ""
                        ratio1_cal_for_save = ratio_norm if {1, 3} <= active else ""
                        ratio2_for_save = ratio2_raw if {2, 4} <= active else ""
                        self._live_records.append(
                            {
                                "timestamp": ts,
                                "active_channels": sorted(active),
                                "ch1_M": self._latest_channels[1] if 1 in active else "off",
                                "ch2_M": self._latest_channels[2] if 2 in active else "off",
                                "ch3_M": self._latest_channels[3] if 3 in active else "off",
                                "ch4_M": self._latest_channels[4] if 4 in active else "off",
                                "ratio1_raw": ratio1_for_save if ratio1_for_save is not None else "sat",
                                "ratio1_calibrated": ratio1_cal_for_save if ratio1_cal_for_save is not None else "sat",
                                "ratio2_raw": ratio2_for_save if ratio2_for_save is not None else "sat",
                            }
                        )
                        self._record_rate_times.append(now)
                while self._record_rate_times and now - self._record_rate_times[0] > 1.0:
                    self._record_rate_times.popleft()
                self._record_rate_var.set(f"{len(self._record_rate_times):.1f} Hz")
                graph_values = [item if (idx + 1) in active else None for idx, item in enumerate(ch_data)]
                self._sensor_display.update_values(graph_values)
            self._after_id = self.after(SERIAL_POLL_MS, poll)
        self._after_id = self.after(SERIAL_POLL_MS, poll)

    def _on_ratio_key(self, event):
        """Keyboard shortcuts for ratio tools: c=calibrate, a=add, s=save, r=record, t=stop."""
        if event.keysym == "c":
            self._on_calibrate_ratio()
        elif event.keysym == "a":
            self._on_add_ratio_measurement()
        elif event.keysym == "s":
            self._on_save_ratio_measurements()
        elif event.keysym == "r":
            self._on_live_record_start()
        elif event.keysym == "t":
            self._on_live_record_stop()

    def _log_action(self, msg: str):
        if not hasattr(self, "_activity_log"):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self._activity_log.config(state="normal")
        self._activity_log.insert(tk.END, f"[{ts}] {msg}\n")
        self._activity_log.see(tk.END)
        self._activity_log.config(state="disabled")

    def _on_state_change(self, old: AppState, new: AppState):
        self.after(0, lambda: self._state_var.set(new.name))
        self.after(0, self._update_controls)

    def _update_controls(self):
        state = self._controller.state if self._controller else AppState.DISCONNECTED
        motor_connected = state != AppState.DISCONNECTED and state != AppState.ERROR
        manual_ok = state in (AppState.IDLE_MANUAL, AppState.MANUAL_MOVING)
        auto_ok = state in (AppState.AUTO_IDLE, AppState.AUTO_RUNNING, AppState.AUTO_SAVING)
        self._stop_btn.state(["!disabled"] if motor_connected else ["disabled"])
        self._manual_mode_btn.state(["!disabled"] if motor_connected and state == AppState.AUTO_IDLE else ["disabled"])
        self._auto_mode_btn.state(["!disabled"] if motor_connected and state == AppState.IDLE_MANUAL else ["disabled"])
        self._btn_left.state(["!disabled"] if manual_ok else ["disabled"])
        self._btn_right.state(["!disabled"] if manual_ok else ["disabled"])
        self._btn_up.state(["!disabled"] if manual_ok else ["disabled"])
        self._btn_down.state(["!disabled"] if manual_ok else ["disabled"])
        self._scan_left_btn.state(["!disabled"] if auto_ok and state == AppState.AUTO_IDLE else ["disabled"])
        self._scan_right_btn.state(["!disabled"] if auto_ok and state == AppState.AUTO_IDLE else ["disabled"])
        self._scan_up_btn.state(["!disabled"] if auto_ok and state == AppState.AUTO_IDLE else ["disabled"])
        self._scan_down_btn.state(["!disabled"] if auto_ok and state == AppState.AUTO_IDLE else ["disabled"])

    def _on_stop_all(self):
        if self._controller:
            self._controller.stop_all()
            self._moving_direction_1 = None
            self._moving_direction_2 = None

    def _on_manual_mode(self):
        if self._controller and self._controller.switch_to_manual_mode():
            self._state_var.set(self._controller.state.name)
            self._update_controls()

    def _on_automation_mode(self):
        if self._controller and self._controller.switch_to_automation_mode():
            self._state_var.set(self._controller.state.name)
            self._update_controls()

    def _manual_start(self, axis: str, direction: str):
        """Start moving in the specified direction and axis (button or keyboard press)."""
        if not self._controller or not self._controller.state_machine.can_manual_move():
            return
        speed = self._manual_speed_var.get()
        self._controller.manual_start(axis, direction, speed)
        if axis == AXIS_1:
            self._moving_direction_1 = direction
        elif axis == AXIS_2:
            self._moving_direction_2 = direction
    
    def _manual_toggle(self, direction: str):
        """Start moving in the specified direction (single button press)."""
        if not self._controller or not self._controller.state_machine.can_manual_move():
            return
        speed = self._manual_speed_var.get()
        self._controller.manual_start(AXIS_1, direction, speed)
        self._moving_direction_1 = direction
    
    def _update_button_colors(self):
        """Update button appearance (optional visual feedback)."""
        pass  # Can add visual feedback here if desired

    def _on_manual_speed_change(self, speed: int):
        if self._controller:
            self._controller.manual_update_speed(speed)

    def _on_delay_change(self, var, index, mode):
        """Send SPEED command to Arduino when delay spinbox changes."""
        if self._motor_adapter:
            try:
                delay_ms = self._auto_delay_var.get()
                if delay_ms:
                    self._motor_adapter.send_speed(delay_ms)
            except (TypeError, ValueError, tk.TclError):
                pass  # Ignore invalid spinbox values during editing

    def _on_sample_window_change(self, value: int):
        """Update sensor reader sample window (number of packets per median update)."""
        self._sample_window_var.set(value)
        # Keep entry in sync with slider
        try:
            if hasattr(self, "_sample_window_entry"):
                self._sample_window_entry.delete(0, tk.END)
                self._sample_window_entry.insert(0, str(int(value)))
        except Exception:
            pass
        if self._sensor_reader and self._sensor_reader.is_open():
            try:
                self._sensor_reader.set_samples_per_update(value)
            except Exception:
                pass

    def _on_sample_window_apply(self):
        """User typed a value for sample window; clamp, sync slider, and apply."""
        try:
            raw = self._sample_window_entry.get()
            n = int(raw)
        except Exception:
            return
        n = max(100, min(2000, n))
        self._on_sample_window_change(n)

    def _on_live_record_start(self):
        """Start continuous logging of live sensor values to in-memory list."""
        if self._live_recording:
            return
        if not self._sensor_reader or not self._sensor_reader.is_open():
            messagebox.showwarning("Record", "Cannot start recording: sensor not connected.")
            return
        self._live_records = []
        self._live_recording = True
        self._record_rate_times.clear()
        self._last_live_record_ts = 0.0
        self._record_rate_var.set("0.0 Hz")
        if hasattr(self, "_record_btn") and hasattr(self, "_stop_record_btn"):
            self._record_btn.state(["disabled"])
            self._stop_record_btn.state(["!disabled"])
        messagebox.showinfo("Record", "Live recording started. Press '(t) Stop' to save.")

    def _prompt_save_filename(self) -> Tuple[bool, Optional[str]]:
        """Ask user for default filename or custom filename before saving XLSX."""
        choice = messagebox.askyesnocancel(
            "Save XLSX",
            "Use default file name?\nYes = default\nNo = custom name\nCancel = abort save",
        )
        if choice is None:
            return False, None
        if choice:
            return True, None
        suggested = datetime.now().strftime("ratio_%Y-%m-%d_%H%M%S")
        custom_name = simpledialog.askstring(
            "Custom file name",
            "Enter file name (without extension):",
            initialvalue=suggested,
            parent=self,
        )
        if custom_name is None:
            return False, None
        custom_name = custom_name.strip()
        if not custom_name:
            return True, None
        return True, custom_name

    def _on_live_record_stop(self):
        """Stop continuous logging and save to CSV via DataLogger."""
        if not self._live_recording:
            return
        self._live_recording = False
        self._record_rate_times.clear()
        self._last_live_record_ts = 0.0
        self._record_rate_var.set("0.0 Hz")
        if hasattr(self, "_record_btn") and hasattr(self, "_stop_record_btn"):
            self._record_btn.state(["!disabled"])
            self._stop_record_btn.state(["disabled"])
        if not self._live_records:
            messagebox.showinfo("Record", "No samples were recorded.")
            return
        try:
            proceed, custom_name = self._prompt_save_filename()
            if not proceed:
                return
            path = self._logger.save_ratio_measurements(self._live_records, file_name=custom_name)
            messagebox.showinfo("Record", f"Recorded {len(self._live_records)} samples to:\n{path}")
            self._live_records = []
        except Exception as exc:
            messagebox.showerror("Record", f"Failed to save recording:\n{exc}")

    def _on_calibrate_ratio(self):
        """Set current Ratio1 (CH1/CH3) as reference so future values are ~1."""
        if self._current_ratio_raw is None or self._current_ratio_raw <= 0:
            messagebox.showwarning("Calibrate ratio", "Cannot calibrate Ratio1: CH1/CH3 is not available (channels may be 0 or saturated).")
            return
        self._ratio_cal_ref = self._current_ratio_raw
        self._log_action(f"Calibrated Ratio1: raw {self._current_ratio_raw:.3f} -> 1.0")

    def _on_add_ratio_measurement(self):
        """Add current ratio point to in-memory list."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        active = self._active_channels()
        entry = {
            "timestamp": ts,
            "active_channels": sorted(active),
            "ch1_M": self._latest_channels[1] if 1 in active else "off",
            "ch2_M": self._latest_channels[2] if 2 in active else "off",
            "ch3_M": self._latest_channels[3] if 3 in active else "off",
            "ch4_M": self._latest_channels[4] if 4 in active else "off",
            "ratio1_raw": (self._current_ratio_raw if self._current_ratio_raw is not None else "sat") if {1, 3} <= active else "",
            "ratio1_calibrated": (self._last_ratio_norm if self._last_ratio_norm is not None else "sat") if {1, 3} <= active else "",
            "ratio2_raw": (self._current_ratio2_raw if self._current_ratio2_raw is not None else "sat") if {2, 4} <= active else "",
        }
        self._ratio_measurements.append(entry)
        self._log_action(f"Measurement #{len(self._ratio_measurements)} stored.")

    def _on_save_ratio_measurements(self):
        """Save collected ratio measurements to CSV via DataLogger."""
        if not self._ratio_measurements:
            messagebox.showwarning("Save XLSX", "No measurements to save.")
            return
        try:
            count = len(self._ratio_measurements)
            proceed, custom_name = self._prompt_save_filename()
            if not proceed:
                return
            path = self._logger.save_ratio_measurements(self._ratio_measurements, file_name=custom_name)
            self._ratio_measurements = []
            self._log_action(f"Saved {count} measurements → {path.name}")
            messagebox.showinfo("Save", f"Saved {count} measurements to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Save", f"Failed to save measurements:\n{exc}")

    def _start_scan(self, direction: str):
        if not self._controller or self._controller.state != AppState.AUTO_IDLE:
            return
        try:
            step_size = int(self._step_size_var.get())
            num_steps = int(self._num_steps_var.get())
            delay_ms = int(self._auto_delay_var.get())
        except (TypeError, ValueError, tk.TclError):
            step_size = DEFAULT_STEP_SIZE
            num_steps = DEFAULT_NUM_STEPS
            delay_ms = 500
        step_size = max(MIN_STEP_SIZE, min(MAX_STEP_SIZE, step_size))
        num_steps = max(MIN_NUM_STEPS, min(MAX_NUM_STEPS, num_steps))
        delay_ms = max(MIN_DELAY_MS, min(MAX_DELAY_MS, delay_ms))
        self._last_scan_direction = direction
        self._last_scan_step_size = step_size
        # Map GUI directions to same physical axes as manual controls.
        if direction in (DIR_LEFT, DIR_RIGHT):
            gui_axis = AXIS_1
        else:
            gui_axis = AXIS_2
        if gui_axis == AXIS_1:
            axis, real_dir = (AXIS_2, (DIR_DOWN if direction == DIR_LEFT else DIR_UP))
        else:
            axis, real_dir = (AXIS_1, (DIR_RIGHT if direction == DIR_UP else DIR_LEFT))
        get_sensor = None
        if self._sensor_reader and self._sensor_reader.is_open():
            reader = self._sensor_reader
            get_sensor = lambda step: reader.get_latest_reading(str(step))
        self._controller.start_automation(
            axis, real_dir, step_size, num_steps, get_sensor_reading=get_sensor, delay_ms=delay_ms
        )
        self._progress_var.set(f"Running 0 / {num_steps}")
        self._update_controls()

    def _on_automation_progress(self, current: int, total: int):
        self.after(0, lambda: self._progress_var.set(f"Running {current} / {total}"))

    def _on_automation_complete(self, completed_ok: bool):
        def do():
            self._progress_var.set("Saving..." if completed_ok else "Stopped (saving partial)")
            rows = self._controller.get_automation_buffer() if self._controller else []
            step_size = self._last_scan_step_size
            direction = self._last_scan_direction
            path = self._logger.save_automation_scan(
                rows=rows,
                direction=direction,
                step_size=step_size,
                partial=not completed_ok,
            )
            self._controller.automation_finish_saving()
            self._state_var.set(self._controller.state.name)
            self._progress_var.set(f"Saved: {path.name}")
            self._update_controls()
        self.after(0, do)

    def _on_sensor_reading(self, reading):
        self.after(0, lambda: self._sensor_display.update_values(reading.values))

    def _on_error(self, msg: str):
        self.after(0, lambda: messagebox.showerror("Error", msg))
        self.after(0, lambda: self._state_var.set(AppState.ERROR.name))
        self.after(0, self._update_controls)


def run_gui(
    motor_port: str = DEFAULT_MOTOR_PORT,
    motor_baud: int = DEFAULT_MOTOR_BAUD,
    sensor_port: str = DEFAULT_SENSOR_PORT,
    sensor_baud: int = DEFAULT_SENSOR_BAUD,
):
    root = tk.Tk()
    root.title("Motor Control + Sensor")
    # Larger default window and fonts for better readability
    root.geometry("900x650")
    try:
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(size=11)
        tkfont.nametofont("TkTextFont").configure(size=11)
        tkfont.nametofont("TkHeadingFont").configure(size=12, weight="bold")
    except Exception:
        # If default fonts aren't available, skip font customization
        pass
    app = MainApp(
        root,
        motor_port=motor_port,
        motor_baud=motor_baud,
        sensor_port=sensor_port,
        sensor_baud=sensor_baud,
    )
    app.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._on_disconnect(), root.destroy()))
    root.mainloop()

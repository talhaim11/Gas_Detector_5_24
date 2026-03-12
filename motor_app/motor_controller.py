"""
Motor controller: bridges state machine, serial manager, and automation loop.
Handles Manual (hold-to-move) and Automation (step scan) with safe transitions.
"""
import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass

from motor_app.state_machine import AppState, MotorStateMachine
from motor_app.serial_manager import ParsedResponse
from motor_app.config import (
    AXIS_1, AXIS_2, DIR_LEFT, DIR_RIGHT, DIR_UP, DIR_DOWN,
    MIN_DELAY_MS, MAX_DELAY_MS,
    RSP_ACK, RSP_ERR, RSP_MOTION_DONE, RSP_SENSORS,
)


@dataclass
class SensorReading:
    """One row of sensor data from Arduino (READ_SENSORS response)."""
    timestamp: str
    position: str
    values: list  # [sensor_1, sensor_2, ...]


# Callback types
OnStateChangeCB = Callable[[AppState, AppState], None]
OnSensorReadingCB = Callable[[SensorReading], None]
OnAutomationProgressCB = Callable[[int, int], None]  # current_step, total_steps
OnAutomationCompleteCB = Callable[[bool], None]     # completed_ok (False = stopped partial)
OnErrorCB = Callable[[str], None]
GetSensorReadingCB = Callable[[int], SensorReading]  # step_index -> SensorReading for CSV


class MotorController:
    """
    Coordinates state machine and serial. Manual: press/release sends START/STOP.
    Automation: runs step loop in worker thread; each step sends STEP, waits MOTION_DONE, then READ_SENSORS.
    """
    def __init__(
        self,
        serial_manager,  # SerialManager or LegacyMotorAdapter (duck typing)
        on_state_change: Optional[OnStateChangeCB] = None,
        on_sensor_reading: Optional[OnSensorReadingCB] = None,
        on_automation_progress: Optional[OnAutomationProgressCB] = None,
        on_automation_complete: Optional[OnAutomationCompleteCB] = None,
        on_error: Optional[OnErrorCB] = None,
    ):
        self._serial = serial_manager
        self._sm = MotorStateMachine(on_state_change=on_state_change)
        self._on_sensor = on_sensor_reading
        self._on_automation_progress = on_automation_progress
        self._on_automation_complete = on_automation_complete
        self._on_error = on_error
        self._lock = threading.RLock()
        # Manual: which axis/dir is currently held (for UPDATE_SPEED and STOP on release)
        self._manual_axis: Optional[str] = None
        self._manual_direction: Optional[str] = None
        self._manual_speed = 50
        # Automation worker
        self._auto_thread: Optional[threading.Thread] = None
        self._auto_stop_requested = False
        self._auto_steps_total = 0
        self._auto_steps_done = 0
        self._auto_axis = AXIS_1
        self._auto_direction = DIR_RIGHT
        self._auto_step_size = 10
        self._auto_delay_ms = 250
        self._auto_data_buffer: list = []  # list of SensorReading or dict rows for CSV
        self._auto_get_sensor: Optional[GetSensorReadingCB] = None

    @property
    def state(self) -> AppState:
        return self._sm.state

    @property
    def state_machine(self) -> MotorStateMachine:
        return self._sm

    def connect(self) -> bool:
        """Connect serial and transition to IDLE_MANUAL (user can then switch to AUTO_IDLE)."""
        if self._serial.connect():
            try:
                self._sm.transition_to(AppState.IDLE_MANUAL)
                return True
            except ValueError:
                pass
        return False

    def disconnect(self) -> None:
        """Send STOP ALL, disconnect serial, go to DISCONNECTED."""
        self.request_automation_stop()
        self._serial.send_stop_all()
        self._serial.disconnect()
        try:
            self._sm.transition_to(AppState.DISCONNECTED, force=True)
        except ValueError:
            self._sm.transition_to(AppState.DISCONNECTED)
        self._manual_axis = None
        self._manual_direction = None

    def process_responses(self) -> None:
        """
        Process any pending serial responses. Call from GUI timer or main loop.
        Handles ACK, ERR, MOTION_DONE, SENSORS. Non-blocking.
        """
        while True:
            rsp = self._serial.get_response(block=False)
            if rsp is None:
                break
            self._handle_response(rsp)

    def _handle_response(self, rsp: ParsedResponse) -> None:
        if rsp.kind == RSP_ERR:
            if self._on_error:
                self._on_error("Arduino: " + (rsp.raw or " ".join(rsp.payload)))
            try:
                self._sm.transition_to(AppState.ERROR, force=True)
            except ValueError:
                pass
            return
        if rsp.kind == RSP_MOTION_DONE:
            # Automation thread may be waiting for this; we signal via shared state or queue.
            # TODO: Arduino integration - automation loop can wait on MOTION_DONE per step.
            pass
        if rsp.kind == RSP_SENSORS:
            # payload: timestamp, position, val1, val2, ...
            if len(rsp.payload) >= 2:
                reading = SensorReading(
                    timestamp=rsp.payload[0],
                    position=rsp.payload[1],
                    values=list(rsp.payload[2:]),
                )
                if self._on_sensor:
                    self._on_sensor(reading)
                self._auto_data_buffer.append(reading)

    # --- Manual mode: hold-to-move ---

    def manual_start(self, axis: str, direction: str, speed: int) -> bool:
        """
        Called on direction button press. Send START <axis> <dir> <speed>.
        Transition IDLE_MANUAL -> MANUAL_MOVING.
        """
        with self._lock:
            if not self._sm.can_manual_move():
                return False
            if not self._serial.send_start(axis, direction, speed):
                return False
            try:
                self._sm.transition_to(AppState.MANUAL_MOVING)
            except ValueError:
                return False
            self._manual_axis = axis
            self._manual_direction = direction
            self._manual_speed = speed
            return True

    def manual_stop(self, axis: Optional[str] = None) -> bool:
        """
        Called on button release or global STOP. Send STOP (with direction so adapter sends it first).
        Transition MANUAL_MOVING -> IDLE_MANUAL.
        """
        with self._lock:
            ax = axis or self._manual_axis
            direction = self._manual_direction
            if ax and self._serial.is_open():
                self._serial.send_stop(ax, direction)
            try:
                self._sm.transition_to(AppState.IDLE_MANUAL)
            except ValueError:
                pass
            self._manual_axis = None
            self._manual_direction = None
            return True

    def manual_update_speed(self, speed: int) -> bool:
        """While moving, update speed. Send UPDATE_SPEED <axis> <speed>."""
        with self._lock:
            if self._sm.state != AppState.MANUAL_MOVING or not self._manual_axis:
                return False
            self._manual_speed = speed
            return self._serial.send_update_speed(self._manual_axis, speed)

    # --- Global STOP (Manual or Automation) ---

    def stop_all(self) -> bool:
        """Send STOP ALL. If MANUAL_MOVING send current direction's stop first, then stop all."""
        with self._lock:
            if not self._serial.is_open():
                return False
            if self._sm.state == AppState.MANUAL_MOVING and self._manual_axis and self._manual_direction:
                self._serial.send_stop(self._manual_axis, self._manual_direction)
            else:
                self._serial.send_stop_all()
            if self._sm.state == AppState.MANUAL_MOVING:
                self._manual_axis = None
                self._manual_direction = None
                try:
                    self._sm.transition_to(AppState.IDLE_MANUAL)
                except ValueError:
                    pass
            elif self._sm.state == AppState.AUTO_RUNNING:
                self._auto_stop_requested = True
            return True

    def request_automation_stop(self) -> None:
        """Ask automation loop to stop after current step."""
        self._auto_stop_requested = True

    # --- Mode switch: must send STOP ALL first ---

    def switch_to_manual_mode(self) -> bool:
        """From AUTO_IDLE, transition to IDLE_MANUAL after STOP ALL."""
        with self._lock:
            if not self._sm.is_safe_to_switch_mode():
                return False
            if self._sm.state == AppState.AUTO_IDLE:
                self._serial.send_stop_all()
                try:
                    self._sm.transition_to(AppState.IDLE_MANUAL)
                    return True
                except ValueError:
                    pass
            return self._sm.state == AppState.IDLE_MANUAL

    def switch_to_automation_mode(self) -> bool:
        """From IDLE_MANUAL, transition to AUTO_IDLE after STOP ALL."""
        with self._lock:
            if not self._sm.is_safe_to_switch_mode():
                return False
            if self._sm.state == AppState.IDLE_MANUAL:
                self._serial.send_stop_all()
                try:
                    self._sm.transition_to(AppState.AUTO_IDLE)
                    return True
                except ValueError:
                    pass
            return self._sm.state == AppState.AUTO_IDLE

    # --- Automation: step scan ---

    def start_automation(
        self,
        axis: str,
        direction: str,
        step_size: int,
        num_steps: int,
        get_sensor_reading: Optional[GetSensorReadingCB] = None,
        delay_ms: int = 250,
    ) -> bool:
        """
        Start automation scan. Transitions AUTO_IDLE -> AUTO_RUNNING and starts worker thread.
        Worker: for each step send STEP, wait delay_ms, then get_sensor_reading(step) and append to buffer.
        On finish: AUTO_SAVING (save CSV), then AUTO_IDLE. On STOP: save partial, AUTO_IDLE.
        """
        with self._lock:
            if self._sm.state != AppState.AUTO_IDLE:
                return False
            try:
                self._sm.transition_to(AppState.AUTO_RUNNING)
            except ValueError:
                return False
            self._auto_stop_requested = False
            self._auto_data_buffer = []
            self._auto_axis = axis
            self._auto_direction = direction
            self._auto_step_size = step_size
            self._auto_steps_total = num_steps
            self._auto_steps_done = 0
            self._auto_get_sensor = get_sensor_reading
            self._auto_delay_ms = max(MIN_DELAY_MS, min(MAX_DELAY_MS, delay_ms))
            self._auto_thread = threading.Thread(
                target=self._automation_loop,
                args=(axis, direction, step_size, num_steps),
                daemon=True,
            )
            self._auto_thread.start()
            return True

    def _automation_loop(
        self,
        axis: str,
        direction: str,
        step_size: int,
        num_steps: int,
    ) -> None:
        """
        Run in background. For each step: send STEP (legacy: R1/L1 step_size), wait,
        then get_sensor_reading(step) and append to buffer. Sensors from separate sensor port.
        """
        completed_ok = False
        try:
            for step in range(num_steps):
                if self._auto_stop_requested:
                    break
                if not self._serial.is_open():
                    break
                self._serial.send_step(axis, direction, step_size)
                time.sleep(self._auto_delay_ms / 1000.0)
                if self._auto_get_sensor:
                    reading = self._auto_get_sensor(step + 1)
                    with self._lock:
                        self._auto_data_buffer.append(reading)
                self._auto_steps_done = step + 1
                if self._on_automation_progress:
                    self._on_automation_progress(step + 1, num_steps)
            completed_ok = not self._auto_stop_requested and self._auto_steps_done == num_steps
        finally:
            try:
                self._sm.transition_to(AppState.AUTO_SAVING)
            except ValueError:
                pass
            if self._on_automation_complete:
                self._on_automation_complete(completed_ok)
            # AUTO_SAVING -> AUTO_IDLE is done by data_logger/gui after save
            self._auto_thread = None

    def get_automation_buffer(self) -> list:
        """Return collected sensor rows for current/last automation run (for CSV save)."""
        with self._lock:
            return list(self._auto_data_buffer)

    def automation_finish_saving(self) -> None:
        """Call after CSV save done. Transition AUTO_SAVING -> AUTO_IDLE."""
        try:
            self._sm.transition_to(AppState.AUTO_IDLE)
        except ValueError:
            pass

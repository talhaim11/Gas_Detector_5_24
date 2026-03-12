"""
Motor control state machine. All transitions are explicit and validated.
Prevents unsafe transitions and uncontrolled motion.
"""
from enum import Enum, auto
from typing import Optional, Set, Callable, Any
import threading


class AppState(Enum):
    """Main application states. Transitions must use transition_to()."""
    DISCONNECTED = auto()
    IDLE_MANUAL = auto()
    MANUAL_MOVING = auto()
    AUTO_IDLE = auto()
    AUTO_RUNNING = auto()
    AUTO_SAVING = auto()
    ERROR = auto()


# Allowed transitions: from_state -> set of to_states
_TRANSITIONS: dict[AppState, Set[AppState]] = {
    AppState.DISCONNECTED: {AppState.IDLE_MANUAL, AppState.AUTO_IDLE, AppState.ERROR},
    AppState.IDLE_MANUAL: {AppState.MANUAL_MOVING, AppState.AUTO_IDLE, AppState.DISCONNECTED, AppState.ERROR},
    AppState.MANUAL_MOVING: {AppState.IDLE_MANUAL, AppState.DISCONNECTED, AppState.ERROR},
    AppState.AUTO_IDLE: {AppState.AUTO_RUNNING, AppState.IDLE_MANUAL, AppState.DISCONNECTED, AppState.ERROR},
    AppState.AUTO_RUNNING: {AppState.AUTO_SAVING, AppState.AUTO_IDLE, AppState.DISCONNECTED, AppState.ERROR},
    AppState.AUTO_SAVING: {AppState.AUTO_IDLE, AppState.ERROR},
    AppState.ERROR: {AppState.DISCONNECTED, AppState.IDLE_MANUAL, AppState.AUTO_IDLE},
}


def _is_allowed(from_state: AppState, to_state: AppState) -> bool:
    """Return True if transition from_state -> to_state is allowed."""
    allowed = _TRANSITIONS.get(from_state)
    return allowed is not None and to_state in allowed


class MotorStateMachine:
    """
    Thread-safe state machine for motor control.
    Call transition_to() for every state change; invalid transitions raise.
    """
    _lock = threading.RLock()

    def __init__(self, on_state_change: Optional[Callable[[AppState, AppState], None]] = None):
        self._state = AppState.DISCONNECTED
        self._on_state_change = on_state_change  # (old_state, new_state)

    @property
    def state(self) -> AppState:
        with self._lock:
            return self._state

    def transition_to(self, new_state: AppState, force: bool = False) -> bool:
        """
        Transition to new_state if allowed. Returns True if transitioned, False if no-op.
        Raises ValueError if transition is not allowed and force is False.
        """
        with self._lock:
            old = self._state
            if old == new_state:
                return False
            if not force and not _is_allowed(old, new_state):
                raise ValueError(
                    f"Invalid transition: {old.name} -> {new_state.name}. "
                    f"Allowed from {old.name}: {[s.name for s in _TRANSITIONS.get(old, set())]}"
                )
            self._state = new_state
            if self._on_state_change:
                try:
                    self._on_state_change(old, new_state)
                except Exception:
                    pass
            return True

    def can_manual_move(self) -> bool:
        """True if manual motion commands are allowed in current state."""
        with self._lock:
            return self._state == AppState.IDLE_MANUAL

    def can_stop(self) -> bool:
        """True if STOP is meaningful (moving or running automation)."""
        with self._lock:
            return self._state in (
                AppState.MANUAL_MOVING,
                AppState.AUTO_RUNNING,
                AppState.AUTO_SAVING,
            )

    def is_manual_mode(self) -> bool:
        with self._lock:
            return self._state in (AppState.IDLE_MANUAL, AppState.MANUAL_MOVING)

    def is_automation_mode(self) -> bool:
        with self._lock:
            return self._state in (AppState.AUTO_IDLE, AppState.AUTO_RUNNING, AppState.AUTO_SAVING)

    def is_connected(self) -> bool:
        with self._lock:
            return self._state != AppState.DISCONNECTED and self._state != AppState.ERROR

    def is_safe_to_switch_mode(self) -> bool:
        """True if we can switch between Manual and Automation (must send STOP ALL first)."""
        with self._lock:
            return self._state in (AppState.IDLE_MANUAL, AppState.AUTO_IDLE)

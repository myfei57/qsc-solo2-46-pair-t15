"""Travel limits, work-zone profiles, multi-axis arbitration and latch release generations."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Mapping, Sequence

from .audit import Audit
from .errors import (
    AxesMoving,
    LimitProfileInvalid,
    LimitProfileUnknown,
    LimitSwitchBusy,
    StaleRelease,
    StoreNotFound,
)
from .hoist import Hoist, limit_stop as hoist_limit_stop
from .slew import Slew, limit_stop as slew_limit_stop
from .store import Store
from .trolley import Trolley, limit_stop as trolley_limit_stop

AXIS_SLEW = "slew"
AXIS_TROLLEY = "trolley"
AXIS_HOIST = "hoist"
AXES = (AXIS_SLEW, AXIS_TROLLEY, AXIS_HOIST)

DEFAULT_MINIMUM = -100.0
DEFAULT_MAXIMUM = 100.0

PROFILE_OPEN = "open"
PROFILE_WALL = "wall"

PROFILES_KEY = "limit/profiles"
MAX_SWITCH_LEDGER = 50


class Limit:
    """Mutable state of the travel limits of every axis.

    Besides the trip/stop bookkeeping the limit owns the named work-zone
    profiles (one travel range set per zone), the profile currently in
    force and the ledger that keeps every profile switch traceable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tripped: dict[str, bool] = {}
        self.stopped: dict[str, bool] = {}
        self.latch = False
        self.generation = 0
        self.sequence = 0
        self.profiles = default_profiles()
        self.active = PROFILE_OPEN
        self.switching = False
        self.switch_sequence = 0
        self.switches: list[ProfileSwitch] = []

    def is_tripped(self, axis: str) -> bool:
        with self._lock:
            return self.tripped.get(axis, False)

    def is_stopped(self, axis: str) -> bool:
        with self._lock:
            return self.stopped.get(axis, False)

    def is_latched(self) -> bool:
        with self._lock:
            return self.latch

    def generation_now(self) -> int:
        with self._lock:
            return self.generation


@dataclass(frozen=True)
class ProfileSwitch:
    """One recorded activation of a limit profile, kept as the switch trail."""

    sequence: int
    previous: str
    active: str
    at: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "previous": self.previous,
            "active": self.active,
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProfileSwitch":
        return cls(
            sequence=int(payload["sequence"]),
            previous=str(payload["previous"]),
            active=str(payload["active"]),
            at=str(payload["at"]),
        )


@dataclass(frozen=True)
class Envelope:
    """A single axis envelope described by a live value reader."""

    axis: str
    minimum: float
    maximum: float
    value: Callable[[], float]

    def exceeded(self) -> bool:
        current = self.value()
        return current < self.minimum or current > self.maximum

    def margin(self) -> float:
        current = self.value()
        if current < self.minimum:
            return self.minimum - current
        if current > self.maximum:
            return current - self.maximum
        return 0.0


@dataclass(frozen=True)
class Arbitration:
    """The result of arbitrating every tripped axis in one pass.

    ``stops`` lists the axes that actually stopped; any axis whose stop
    failed stays tripped and is reported under ``failed`` so the batch
    never pretends a half-finished stop completed.
    """

    axes: tuple[str, ...]
    stops: dict[str, bool]
    sequence: int
    failed: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "axes": list(self.axes),
            "stops": self.stops,
            "sequence": self.sequence,
            "failed": self.failed,
        }


def default_profiles() -> dict[str, dict[str, tuple[float, float]]]:
    """The two built-in work-zone profiles: open field and wall-side work."""

    return {
        PROFILE_OPEN: {axis: (DEFAULT_MINIMUM, DEFAULT_MAXIMUM) for axis in AXES},
        PROFILE_WALL: {
            AXIS_SLEW: (-45.0, 45.0),
            AXIS_TROLLEY: (-60.0, 60.0),
            AXIS_HOIST: (-50.0, 50.0),
        },
    }


def _validate_ranges(ranges: Mapping[str, Sequence[float]]) -> dict[str, tuple[float, float]]:
    if not isinstance(ranges, Mapping):
        raise LimitProfileInvalid("ranges must map every axis to [minimum, maximum]")
    cleaned: dict[str, tuple[float, float]] = {}
    for axis in AXES:
        raw = ranges.get(axis)
        if raw is None:
            raise LimitProfileInvalid(f"range missing for axis {axis}")
        try:
            minimum, maximum = float(raw[0]), float(raw[1])
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise LimitProfileInvalid(f"range malformed for axis {axis}") from exc
        if minimum >= maximum:
            raise LimitProfileInvalid(f"range inverted for axis {axis}")
        cleaned[axis] = (minimum, maximum)
    unknown = sorted(str(axis) for axis in ranges if axis not in AXES)
    if unknown:
        raise LimitProfileInvalid("unknown axes: " + ",".join(unknown))
    return cleaned


def define_profile(
    limit: Limit,
    name: str,
    ranges: Mapping[str, Sequence[float]],
    recorder: Audit | None = None,
) -> None:
    """Create or replace one work-zone profile; every axis needs a range."""

    cleaned_name = str(name).strip()
    if not cleaned_name:
        raise LimitProfileInvalid("profile name required")
    cleaned = _validate_ranges(ranges)
    with limit._lock:  # noqa: SLF001 - module-level functions share limit state
        limit.profiles[cleaned_name] = cleaned
    if recorder is not None:
        recorder.record("system", "limit-profile-define", cleaned_name)


def activate_profile(
    limit: Limit,
    name: str,
    slew: Slew,
    trolley: Trolley,
    hoist: Hoist,
    recorder: Audit | None = None,
) -> ProfileSwitch:
    """Switch the active profile; every axis must stand still for the switch.

    The ``switching`` flag is held for the whole swap so motion commands
    issued meanwhile are refused, and the switch is appended to the ledger
    (and the audit log) so the trail of who-worked-under-which-range is
    never lost.
    """

    with limit._lock:  # noqa: SLF001
        if limit.switching:
            raise LimitSwitchBusy("profile switch already in progress")
        if name not in limit.profiles:
            raise LimitProfileUnknown("unknown limit profile: " + str(name))
        limit.switching = True
        try:
            if slew.is_moving() or trolley.is_moving() or hoist.is_moving():
                raise AxesMoving("axes must stand still while the profile switches")
            previous = limit.active
            limit.active = name
            limit.switch_sequence += 1
            switch = ProfileSwitch(
                sequence=limit.switch_sequence,
                previous=previous,
                active=name,
                at=datetime.now(timezone.utc).isoformat(),
            )
            limit.switches.append(switch)
            if len(limit.switches) > MAX_SWITCH_LEDGER:
                del limit.switches[: len(limit.switches) - MAX_SWITCH_LEDGER]
        finally:
            limit.switching = False
    if recorder is not None:
        recorder.record(
            "system",
            "limit-profile-switch",
            f"{switch.previous} -> {switch.active} seq={switch.sequence}",
        )
    return switch


def is_switching(limit: Limit) -> bool:
    with limit._lock:  # noqa: SLF001
        return limit.switching


def active_profile(limit: Limit) -> str:
    with limit._lock:  # noqa: SLF001
        return limit.active


def switch_sequence_now(limit: Limit) -> int:
    with limit._lock:  # noqa: SLF001
        return limit.switch_sequence


def switch_history(limit: Limit) -> list[ProfileSwitch]:
    with limit._lock:  # noqa: SLF001
        return list(limit.switches)


def profiles_snapshot(limit: Limit) -> dict[str, dict[str, tuple[float, float]]]:
    with limit._lock:  # noqa: SLF001
        return {name: dict(ranges) for name, ranges in limit.profiles.items()}


def active_envelopes(limit: Limit, readers: Mapping[str, Callable[[], float]]) -> list[Envelope]:
    """Build envelopes for ``readers`` using the ranges of the active profile."""

    with limit._lock:  # noqa: SLF001
        ranges = dict(limit.profiles.get(limit.active, {}))
    envelopes: list[Envelope] = []
    for axis in AXES:
        reader = readers.get(axis)
        if reader is None:
            continue
        minimum, maximum = ranges.get(axis, (DEFAULT_MINIMUM, DEFAULT_MAXIMUM))
        envelopes.append(Envelope(axis=axis, minimum=minimum, maximum=maximum, value=reader))
    return envelopes


def persist_limit(limit: Limit, store: Store) -> None:
    """Write the profiles, the active selection and the switch ledger."""

    with limit._lock:  # noqa: SLF001
        payload = json.dumps(
            {
                "active": limit.active,
                "switch_sequence": limit.switch_sequence,
                "profiles": {
                    name: {axis: [span[0], span[1]] for axis, span in ranges.items()}
                    for name, ranges in limit.profiles.items()
                },
                "switches": [switch.to_dict() for switch in limit.switches],
            },
            ensure_ascii=False,
        ).encode("utf-8")
    store.save(PROFILES_KEY, payload)


def restore_limit(limit: Limit, store: Store) -> None:
    """Reload the persisted profiles on boot; missing state keeps defaults."""

    try:
        payload = store.load(PROFILES_KEY)
    except StoreNotFound:
        return
    snapshot = json.loads(payload.decode("utf-8"))
    with limit._lock:  # noqa: SLF001
        profiles = {
            str(name): {
                str(axis): (float(span[0]), float(span[1])) for axis, span in ranges.items()
            }
            for name, ranges in snapshot.get("profiles", {}).items()
        }
        if profiles:
            limit.profiles = profiles
        active = str(snapshot.get("active", PROFILE_OPEN))
        limit.active = active if active in limit.profiles else PROFILE_OPEN
        limit.switch_sequence = int(snapshot.get("switch_sequence", 0))
        limit.switches = [ProfileSwitch.from_dict(item) for item in snapshot.get("switches", [])]


def trip(limit: Limit, axis: str) -> None:
    with limit._lock:  # noqa: SLF001 - module-level functions share limit state
        limit.tripped[axis] = True


def latch(limit: Limit) -> int:
    with limit._lock:  # noqa: SLF001
        limit.latch = True
        limit.generation += 1
        return limit.generation


def release(limit: Limit, generation: int) -> bool:
    """Release the latch for ``generation``; older generations are rejected."""

    with limit._lock:  # noqa: SLF001
        if generation < limit.generation:
            raise StaleRelease("stale release generation")
        limit.latch = False
        limit.generation = generation
        return True


def arbitrate(limit: Limit, slew: Slew, trolley: Trolley, hoist: Hoist) -> Arbitration:
    """Stop every axis that has tripped a limit in a single pass.

    All axes tripped in this batch are stopped together under one sequence.
    A stop that raises never spares the remaining axes: the failed axis
    stays tripped (and is reported under ``failed``) so a later pass can
    retry it, instead of leaving half the batch running unnoticed.
    """

    with limit._lock:  # noqa: SLF001
        limit.sequence += 1
        sequence = limit.sequence
        axes = sorted(limit.tripped)
        stops: dict[str, bool] = {}
        failed: dict[str, str] = {}
        for axis in axes:
            try:
                if axis == AXIS_SLEW:
                    slew_limit_stop(slew, axis, sequence)
                elif axis == AXIS_TROLLEY:
                    trolley_limit_stop(trolley)
                elif axis == AXIS_HOIST:
                    hoist_limit_stop(hoist)
                else:
                    failed[axis] = "unknown axis"
                    continue
            except Exception as exc:  # noqa: BLE001 - one bad stop must not spare the others
                failed[axis] = str(exc)
                continue
            stops[axis] = True
            limit.stopped[axis] = True
            del limit.tripped[axis]
    return Arbitration(axes=tuple(axes), stops=stops, sequence=sequence, failed=failed)


def reset_arbitration(limit: Limit) -> None:
    with limit._lock:  # noqa: SLF001
        limit.stopped = {}


def position_envelope(axis: str, value: Callable[[], float], minimum: float = -100.0, maximum: float = 100.0) -> Envelope:
    return Envelope(axis=axis, minimum=minimum, maximum=maximum, value=value)


def enforce_envelopes(limit: Limit, envelopes: list[Envelope]) -> list[str]:
    """Trip every envelope that is out of range and return the axes."""

    exceeded: list[str] = []
    for envelope in envelopes:
        if envelope.exceeded():
            trip(limit, envelope.axis)
            exceeded.append(envelope.axis)
    return exceeded


def envelope_margins(envelopes: list[Envelope]) -> dict[str, float]:
    return {envelope.axis: envelope.margin() for envelope in envelopes}


def slew_position(x: float) -> Callable[[], float]:
    return lambda: x


def trolley_position(y: float) -> Callable[[], float]:
    return lambda: y

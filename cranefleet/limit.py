"""Travel limits, multi-axis arbitration and latch release generations.

Each work zone owns a named limit profile (one set of per-axis travel
ranges), so wall-side work and mid-field work no longer share a single
fixed envelope. Switching the active profile is audited, persisted and
refused while any axis is moving. When a check trips several axes at
once, :func:`enforce_and_stop` stops that exact group in a single pass so
no axis is left running behind the others.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

from .errors import (
    InvalidProfile,
    ProfileNotFound,
    ProfileSwitchBlocked,
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

PROFILES_KEY = "limit/profiles"


class Limit:
    """Mutable state of the travel limits of every axis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tripped: dict[str, bool] = {}
        self.stopped: dict[str, bool] = {}
        self.latch = False
        self.generation = 0
        self.sequence = 0

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
    """The result of arbitrating every tripped axis in one pass."""

    axes: tuple[str, ...]
    stops: dict[str, bool]
    sequence: int
    profile: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "axes": list(self.axes),
            "stops": self.stops,
            "sequence": self.sequence,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class AxisRange:
    """The travel range of one axis inside a limit profile."""

    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, float]:
        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AxisRange":
        try:
            minimum = float(payload["minimum"])  # type: ignore[index]
            maximum = float(payload["maximum"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidProfile("axis range requires numeric minimum and maximum") from exc
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            raise InvalidProfile("axis range bounds must be finite")
        if minimum >= maximum:
            raise InvalidProfile("axis range minimum must be below maximum")
        return cls(minimum=minimum, maximum=maximum)


DEFAULT_RANGE = AxisRange(-100.0, 100.0)


@dataclass(frozen=True)
class LimitProfile:
    """One named set of per-axis travel ranges for a work zone."""

    profile_id: str
    name: str
    ranges: dict[str, AxisRange]

    def range_for(self, axis: str) -> AxisRange:
        return self.ranges.get(axis, DEFAULT_RANGE)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.profile_id,
            "name": self.name,
            "ranges": {axis: self.ranges[axis].to_dict() for axis in sorted(self.ranges)},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LimitProfile":
        profile_id = str(payload.get("id") or "").strip()
        if not profile_id:
            raise InvalidProfile("profile id required")
        name = str(payload.get("name") or profile_id)
        raw_ranges = payload.get("ranges")
        if not isinstance(raw_ranges, Mapping) or not raw_ranges:
            raise InvalidProfile("profile ranges required")
        ranges: dict[str, AxisRange] = {}
        for axis, raw in raw_ranges.items():
            axis_name = str(axis)
            if axis_name not in AXES:
                raise InvalidProfile("unknown axis " + axis_name)
            if not isinstance(raw, Mapping):
                raise InvalidProfile("axis range must be an object")
            ranges[axis_name] = AxisRange.from_dict(raw)
        return cls(profile_id=profile_id, name=name, ranges=ranges)


class LimitProfiles:
    """Mutable registry of the limit profiles and the active selection."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.profiles: dict[str, LimitProfile] = {}
        self.active = ""
        self.switch_count = 0


def register_profile(profiles: LimitProfiles, profile: LimitProfile) -> None:
    """Add or replace a profile; the first one registered becomes active."""

    with profiles._lock:  # noqa: SLF001 - module-level functions share profile state
        profiles.profiles[profile.profile_id] = profile
        if not profiles.active:
            profiles.active = profile.profile_id


def profiles_snapshot(profiles: LimitProfiles) -> list[LimitProfile]:
    with profiles._lock:  # noqa: SLF001
        return [profiles.profiles[key] for key in sorted(profiles.profiles)]


def active_profile(profiles: LimitProfiles) -> LimitProfile | None:
    with profiles._lock:  # noqa: SLF001
        return profiles.profiles.get(profiles.active)


def active_profile_id(profiles: LimitProfiles) -> str:
    with profiles._lock:  # noqa: SLF001
        return profiles.active


def switch_count(profiles: LimitProfiles) -> int:
    with profiles._lock:  # noqa: SLF001
        return profiles.switch_count


def activate(
    profiles: LimitProfiles,
    profile_id: str,
    moving_axes: Callable[[], list[str]],
) -> LimitProfile:
    """Switch the active profile; refused while any axis is moving.

    The motion check runs inside the registry lock so the decision and the
    swap are one atomic step: the crane must be standing still for the
    whole switch.
    """

    with profiles._lock:  # noqa: SLF001
        profile = profiles.profiles.get(profile_id)
        if profile is None:
            raise ProfileNotFound(profile_id)
        moving = sorted(axis for axis in moving_axes() if axis)
        if moving:
            raise ProfileSwitchBlocked("axes moving: " + ",".join(moving))
        profiles.active = profile_id
        profiles.switch_count += 1
        return profile


def envelopes_for_profile(
    profile: LimitProfile | None,
    readers: Mapping[str, Callable[[], float]],
) -> list[Envelope]:
    """Build the envelopes of the active profile for the axes with readers."""

    envelopes: list[Envelope] = []
    for axis in AXES:
        reader = readers.get(axis)
        if reader is None:
            continue
        travel = profile.range_for(axis) if profile is not None else DEFAULT_RANGE
        envelopes.append(
            Envelope(axis=axis, minimum=travel.minimum, maximum=travel.maximum, value=reader)
        )
    return envelopes


def persist_profiles(profiles: LimitProfiles, store: Store) -> None:
    """Write every profile and the active selection so a restart keeps them."""

    with profiles._lock:  # noqa: SLF001
        payload = json.dumps(
            {
                "active": profiles.active,
                "switch_count": profiles.switch_count,
                "profiles": [
                    profiles.profiles[key].to_dict() for key in sorted(profiles.profiles)
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
    store.save(PROFILES_KEY, payload)


def restore_profiles(profiles: LimitProfiles, store: Store) -> None:
    """Reload persisted profiles over the built-in defaults on boot."""

    try:
        payload = store.load(PROFILES_KEY)
    except StoreNotFound:
        return
    data = json.loads(payload.decode("utf-8"))
    restored = [LimitProfile.from_dict(item) for item in data.get("profiles") or []]
    with profiles._lock:  # noqa: SLF001
        for profile in restored:
            profiles.profiles[profile.profile_id] = profile
        active = str(data.get("active") or "")
        if active in profiles.profiles:
            profiles.active = active
        try:
            profiles.switch_count = max(profiles.switch_count, int(data.get("switch_count") or 0))
        except (TypeError, ValueError):
            pass


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
    """Stop every axis that has tripped a limit in a single pass."""

    with limit._lock:  # noqa: SLF001
        limit.sequence += 1
        sequence = limit.sequence
        axes = sorted(limit.tripped)
        stops: dict[str, bool] = {}
        for axis in axes:
            if axis == AXIS_SLEW:
                slew_limit_stop(slew, axis, sequence)
            elif axis == AXIS_TROLLEY:
                trolley_limit_stop(trolley)
            elif axis == AXIS_HOIST:
                hoist_limit_stop(hoist)
            stops[axis] = True
            limit.stopped[axis] = True
            del limit.tripped[axis]
    return Arbitration(axes=tuple(axes), stops=stops, sequence=sequence)


def reset_arbitration(limit: Limit) -> None:
    with limit._lock:  # noqa: SLF001
        limit.stopped = {}


def stop_together(
    limit: Limit,
    slew: Slew,
    trolley: Trolley,
    hoist: Hoist,
    axes: Iterable[str],
    profile: str = "",
) -> Arbitration:
    """Stop every axis in ``axes`` as one group in a single pass.

    Every axis is attempted even if one stop fails, so the group is never
    left half stopped. An axis whose stop fails stays tripped so a later
    pass retries it.
    """

    with limit._lock:  # noqa: SLF001
        limit.sequence += 1
        sequence = limit.sequence
        group = sorted({axis for axis in axes if axis in AXES})
        stops: dict[str, bool] = {}
        for axis in group:
            try:
                if axis == AXIS_SLEW:
                    slew_limit_stop(slew, axis, sequence)
                elif axis == AXIS_TROLLEY:
                    trolley_limit_stop(trolley)
                elif axis == AXIS_HOIST:
                    hoist_limit_stop(hoist)
            except Exception:  # noqa: BLE001 - never leave the group half stopped
                stops[axis] = False
                limit.tripped[axis] = True
                continue
            stops[axis] = True
            limit.stopped[axis] = True
            limit.tripped.pop(axis, None)
    return Arbitration(axes=tuple(group), stops=stops, sequence=sequence, profile=profile)


def enforce_and_stop(
    limit: Limit,
    slew: Slew,
    trolley: Trolley,
    hoist: Hoist,
    envelopes: list[Envelope],
    profile: str = "",
) -> Arbitration | None:
    """Stop the axes tripped by this exact check as one group.

    Returns ``None`` when every envelope is inside its range, otherwise the
    arbitration of the axes that exceeded in this pass.
    """

    exceeded = [envelope.axis for envelope in envelopes if envelope.exceeded()]
    if not exceeded:
        return None
    return stop_together(limit, slew, trolley, hoist, exceeded, profile)


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

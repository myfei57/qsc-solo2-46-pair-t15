"""Unit and integration tests for limit profiles and the group stop."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cranefleet import limit as limit_module
from cranefleet.console import Console, Request
from cranefleet.errors import InvalidProfile, ProfileNotFound, ProfileSwitchBlocked
from cranefleet.hoist import Hoist
from cranefleet.limit import (
    AXES,
    AXIS_HOIST,
    AXIS_SLEW,
    AXIS_TROLLEY,
    DEFAULT_RANGE,
    AxisRange,
    Limit,
    LimitProfile,
    LimitProfiles,
    activate,
    active_profile,
    active_profile_id,
    enforce_and_stop,
    envelopes_for_profile,
    persist_profiles,
    position_envelope,
    profiles_snapshot,
    register_profile,
    restore_profiles,
    stop_together,
    switch_count,
    trip,
)
from cranefleet.slew import Slew
from cranefleet.store import MemoryStore
from cranefleet.trolley import Trolley
from cranefleet.wiring import build_app

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def dispatch(console: Console, method: str, path: str, body: dict | None = None):  # noqa: ANN201
    request = Request(method=method, path=path, query={}, body=body or {}, body_valid=True)
    return console.dispatch(request)


def make_profiles() -> LimitProfiles:
    profiles = LimitProfiles()
    register_profile(
        profiles,
        LimitProfile(
            profile_id="mid-field",
            name="场地中间",
            ranges={axis: AxisRange(-100.0, 100.0) for axis in AXES},
        ),
    )
    register_profile(
        profiles,
        LimitProfile(
            profile_id="wall-side",
            name="靠墙作业",
            ranges={
                AXIS_SLEW: AxisRange(-60.0, 60.0),
                AXIS_TROLLEY: AxisRange(0.0, 80.0),
                AXIS_HOIST: AxisRange(-100.0, 100.0),
            },
        ),
    )
    return profiles


class ProfileRegistryTests(unittest.TestCase):
    def test_first_registered_profile_becomes_active(self) -> None:
        profiles = make_profiles()
        self.assertEqual(active_profile_id(profiles), "mid-field")
        self.assertEqual([p.profile_id for p in profiles_snapshot(profiles)], ["mid-field", "wall-side"])
        self.assertEqual(active_profile(profiles).name, "场地中间")

    def test_activate_switches_and_counts(self) -> None:
        profiles = make_profiles()
        profile = activate(profiles, "wall-side", lambda: [])
        self.assertEqual(profile.name, "靠墙作业")
        self.assertEqual(active_profile_id(profiles), "wall-side")
        self.assertEqual(switch_count(profiles), 1)

    def test_activate_unknown_profile(self) -> None:
        profiles = make_profiles()
        with self.assertRaises(ProfileNotFound):
            activate(profiles, "roof-top", lambda: [])
        self.assertEqual(active_profile_id(profiles), "mid-field")

    def test_activate_refused_while_any_axis_moves(self) -> None:
        profiles = make_profiles()
        with self.assertRaises(ProfileSwitchBlocked) as caught:
            activate(profiles, "wall-side", lambda: [AXIS_TROLLEY])
        self.assertIn(AXIS_TROLLEY, str(caught.exception))
        self.assertEqual(active_profile_id(profiles), "mid-field")
        self.assertEqual(switch_count(profiles), 0)

    def test_profile_validation(self) -> None:
        with self.assertRaises(InvalidProfile):
            AxisRange.from_dict({"minimum": 10.0, "maximum": 10.0})
        with self.assertRaises(InvalidProfile):
            AxisRange.from_dict({"minimum": 0.0, "maximum": float("inf")})
        with self.assertRaises(InvalidProfile):
            AxisRange.from_dict({"minimum": "wide"})
        with self.assertRaises(InvalidProfile):
            LimitProfile.from_dict({"id": "", "ranges": {AXIS_SLEW: {"minimum": 0, "maximum": 1}}})
        with self.assertRaises(InvalidProfile):
            LimitProfile.from_dict({"id": "p1"})
        with self.assertRaises(InvalidProfile):
            LimitProfile.from_dict({"id": "p1", "ranges": {"sleew": {"minimum": 0, "maximum": 1}}})
        with self.assertRaises(InvalidProfile):
            LimitProfile.from_dict({"id": "p1", "ranges": {AXIS_SLEW: "wide"}})
        profile = LimitProfile.from_dict(
            {"id": "p1", "ranges": {AXIS_SLEW: {"minimum": -5, "maximum": 5}}}
        )
        self.assertEqual(profile.name, "p1")
        self.assertEqual(profile.range_for(AXIS_TROLLEY), DEFAULT_RANGE)

    def test_envelopes_follow_the_active_profile(self) -> None:
        profiles = make_profiles()
        activate(profiles, "wall-side", lambda: [])
        readers = {AXIS_SLEW: limit_module.slew_position(70.0)}
        envelopes = envelopes_for_profile(active_profile(profiles), readers)
        self.assertEqual(len(envelopes), 1)
        self.assertTrue(envelopes[0].exceeded())
        mid_field = envelopes_for_profile(
            profiles_snapshot(profiles)[0], readers
        )
        self.assertFalse(mid_field[0].exceeded())
        defaults = envelopes_for_profile(None, readers)
        self.assertFalse(defaults[0].exceeded())

    def test_profiles_survive_persistence(self) -> None:
        store = MemoryStore()
        profiles = make_profiles()
        register_profile(
            profiles,
            LimitProfile(
                profile_id="night-shift",
                name="夜间作业",
                ranges={AXIS_SLEW: AxisRange(-10.0, 10.0)},
            ),
        )
        activate(profiles, "night-shift", lambda: [])
        persist_profiles(profiles, store)

        restored = make_profiles()
        restore_profiles(restored, store)
        self.assertEqual(active_profile_id(restored), "night-shift")
        ids = [p.profile_id for p in profiles_snapshot(restored)]
        self.assertEqual(ids, ["mid-field", "night-shift", "wall-side"])
        night = active_profile(restored)
        self.assertEqual(night.range_for(AXIS_SLEW), AxisRange(-10.0, 10.0))

    def test_restore_without_snapshot_keeps_defaults(self) -> None:
        profiles = make_profiles()
        restore_profiles(profiles, MemoryStore())
        self.assertEqual(active_profile_id(profiles), "mid-field")


class GroupStopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limit = Limit()
        self.slew = Slew("crane-1")
        self.trolley = Trolley("crane-1")
        self.hoist = Hoist("crane-1")

    def test_enforce_and_stop_returns_none_when_inside(self) -> None:
        envelopes = [position_envelope(AXIS_SLEW, limit_module.slew_position(10.0))]
        self.assertIsNone(
            enforce_and_stop(self.limit, self.slew, self.trolley, self.hoist, envelopes)
        )

    def test_triggered_axes_stop_together(self) -> None:
        self.slew.moving = True
        self.trolley.moving = True
        self.hoist.moving = True
        envelopes = [
            position_envelope(AXIS_SLEW, limit_module.slew_position(150.0)),
            position_envelope(AXIS_TROLLEY, limit_module.trolley_position(-150.0)),
            position_envelope(AXIS_HOIST, lambda: 500.0, -50.0, 50.0),
        ]
        arbitration = enforce_and_stop(
            self.limit, self.slew, self.trolley, self.hoist, envelopes, "wall-side"
        )
        self.assertEqual(arbitration.axes, (AXIS_HOIST, AXIS_SLEW, AXIS_TROLLEY))
        self.assertEqual(
            arbitration.stops, {AXIS_HOIST: True, AXIS_SLEW: True, AXIS_TROLLEY: True}
        )
        self.assertEqual(arbitration.profile, "wall-side")
        self.assertFalse(self.slew.is_moving())
        self.assertFalse(self.trolley.is_moving())
        self.assertFalse(self.hoist.is_moving())
        for axis in arbitration.axes:
            self.assertTrue(self.limit.is_stopped(axis))
            self.assertFalse(self.limit.is_tripped(axis))

    def test_group_stop_never_leaves_an_axis_running(self) -> None:
        self.slew.moving = True
        self.trolley.moving = True
        with mock.patch(
            "cranefleet.limit.trolley_limit_stop", side_effect=RuntimeError("drive fault")
        ):
            arbitration = stop_together(
                self.limit, self.slew, self.trolley, self.hoist, [AXIS_SLEW, AXIS_TROLLEY]
            )
        self.assertFalse(self.slew.is_moving())
        self.assertTrue(arbitration.stops[AXIS_SLEW])
        self.assertFalse(arbitration.stops[AXIS_TROLLEY])
        self.assertTrue(self.limit.is_stopped(AXIS_SLEW))
        self.assertFalse(self.limit.is_stopped(AXIS_TROLLEY))
        self.assertTrue(self.limit.is_tripped(AXIS_TROLLEY))

    def test_group_stop_ignores_unknown_axes(self) -> None:
        arbitration = stop_together(
            self.limit, self.slew, self.trolley, self.hoist, [AXIS_SLEW, "winch"]
        )
        self.assertEqual(arbitration.axes, (AXIS_SLEW,))

    def test_earlier_trip_is_not_swept_into_this_pass(self) -> None:
        trip(self.limit, AXIS_HOIST)
        arbitration = stop_together(
            self.limit, self.slew, self.trolley, self.hoist, [AXIS_SLEW]
        )
        self.assertEqual(arbitration.axes, (AXIS_SLEW,))
        self.assertTrue(self.limit.is_tripped(AXIS_HOIST))
        self.assertFalse(self.limit.is_stopped(AXIS_HOIST))


class LimitProfileApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_app("memory", web_root=str(WEB_ROOT))
        self.console = self.app.console

    def tearDown(self) -> None:
        self.app.shutdown()

    def test_default_profiles_listed(self) -> None:
        status, payload = dispatch(self.console, "GET", "/api/limit/profiles")
        self.assertEqual(status, 200)
        data = payload["data"]
        self.assertEqual(data["active"], "mid-field")
        self.assertEqual(data["switch_count"], 0)
        ids = [profile["id"] for profile in data["profiles"]]
        self.assertEqual(ids, ["mid-field", "wall-side"])

    def test_activate_leaves_an_audit_trail(self) -> None:
        status, payload = dispatch(
            self.console, "POST", "/api/limit/activate", {"profile": "wall-side"}
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["active"], "wall-side")
        self.assertEqual(payload["data"]["switch_count"], 1)
        entries = [
            entry
            for entry in self.console.audit.snapshot()
            if entry.action == "limit-profile-switch"
        ]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].detail, "mid-field->wall-side")
        status, payload = dispatch(self.console, "GET", "/api/limit/state")
        self.assertEqual(payload["data"]["active_profile"], "wall-side")
        self.assertEqual(payload["data"]["profile_name"], "靠墙作业")

    def test_activate_is_refused_while_an_axis_moves(self) -> None:
        status, _ = dispatch(self.console, "POST", "/api/slew/start")
        self.assertEqual(status, 200)
        status, payload = dispatch(
            self.console, "POST", "/api/limit/activate", {"profile": "wall-side"}
        )
        self.assertEqual(status, 409)
        self.assertIn("slew", payload["error"])
        self.assertEqual(active_profile_id(self.console.profiles), "mid-field")
        blocked = [
            entry
            for entry in self.console.audit.snapshot()
            if entry.action == "limit-switch-blocked"
        ]
        self.assertEqual(len(blocked), 1)

    def test_activate_unknown_or_missing_profile(self) -> None:
        status, _ = dispatch(self.console, "POST", "/api/limit/activate", {"profile": "roof"})
        self.assertEqual(status, 404)
        status, _ = dispatch(self.console, "POST", "/api/limit/activate", {})
        self.assertEqual(status, 400)

    def test_register_profile_validation_and_audit(self) -> None:
        status, payload = dispatch(
            self.console,
            "POST",
            "/api/limit/profile",
            {"id": "bad", "ranges": {"slew": {"minimum": 5, "maximum": 5}}},
        )
        self.assertEqual(status, 400)
        status, payload = dispatch(
            self.console,
            "POST",
            "/api/limit/profile",
            {
                "id": "night-shift",
                "name": "夜间作业",
                "ranges": {"slew": {"minimum": -10, "maximum": 10}},
            },
        )
        self.assertEqual(status, 200)
        ids = [profile["id"] for profile in payload["data"]["profiles"]]
        self.assertIn("night-shift", ids)
        actions = [entry.action for entry in self.console.audit.snapshot()]
        self.assertIn("limit-profile-register", actions)

    def test_limit_check_uses_the_active_profile(self) -> None:
        dispatch(
            self.console, "POST", "/api/cluster/survey", {"cranes": {"crane-2": {"x": 0, "y": 90}}}
        )
        status, payload = dispatch(self.console, "POST", "/api/limit/check")
        self.assertEqual(payload["data"]["exceeded"], [])
        self.assertEqual(payload["data"]["profile"], "mid-field")
        dispatch(self.console, "POST", "/api/limit/activate", {"profile": "wall-side"})
        status, payload = dispatch(self.console, "POST", "/api/limit/check")
        self.assertEqual(payload["data"]["exceeded"], ["trolley"])
        self.assertEqual(payload["data"]["profile"], "wall-side")

    def test_limit_check_stops_the_triggered_axes_together(self) -> None:
        dispatch(
            self.console,
            "POST",
            "/api/cluster/survey",
            {"cranes": {"crane-1": {"x": 150, "y": 0}, "crane-2": {"x": 0, "y": 150}}},
        )
        status, payload = dispatch(self.console, "POST", "/api/limit/check")
        self.assertEqual(status, 200)
        data = payload["data"]
        self.assertEqual(data["exceeded"], ["slew", "trolley"])
        arbitration = data["arbitration"]
        self.assertEqual(arbitration["axes"], ["slew", "trolley"])
        self.assertEqual(arbitration["stops"], {"slew": True, "trolley": True})
        self.assertEqual(arbitration["profile"], "mid-field")
        self.assertTrue(data["limit"]["stopped"]["slew"])
        self.assertTrue(data["limit"]["stopped"]["trolley"])
        group_stops = [
            entry
            for entry in self.console.audit.snapshot()
            if entry.action == "limit-group-stop"
        ]
        self.assertEqual(len(group_stops), 1)
        self.assertIn("profile=mid-field", group_stops[0].detail)


class LimitProfileBootTests(unittest.TestCase):
    def test_active_profile_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = build_app(directory, web_root=str(WEB_ROOT))
            dispatch(
                first.console, "POST", "/api/limit/activate", {"profile": "wall-side"}
            )
            dispatch(
                first.console,
                "POST",
                "/api/limit/profile",
                {
                    "id": "night-shift",
                    "name": "夜间作业",
                    "ranges": {"hoist": {"minimum": -20, "maximum": 20}},
                },
            )
            first.shutdown()

            second = build_app(directory, web_root=str(WEB_ROOT))
            self.assertEqual(active_profile_id(second.console.profiles), "wall-side")
            ids = [p.profile_id for p in profiles_snapshot(second.console.profiles)]
            self.assertIn("night-shift", ids)
            self.assertEqual(switch_count(second.console.profiles), 1)
            second.shutdown()


if __name__ == "__main__":
    unittest.main()

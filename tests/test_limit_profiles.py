"""Tests for work-zone limit profiles, traced switching and batch stops."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cranefleet.audit import Audit
from cranefleet.console import Request
from cranefleet.errors import (
    AxesMoving,
    LimitProfileInvalid,
    LimitProfileUnknown,
)
from cranefleet.hoist import Hoist, lower as hoist_lower
from cranefleet.limit import (
    AXES,
    AXIS_SLEW,
    AXIS_TROLLEY,
    PROFILE_OPEN,
    PROFILE_WALL,
    Limit,
    activate_profile,
    active_envelopes,
    active_profile,
    arbitrate,
    define_profile,
    enforce_envelopes,
    is_switching,
    persist_limit,
    profiles_snapshot,
    restore_limit,
    switch_history,
    trip,
)
from cranefleet.slew import Slew
from cranefleet.store import MemoryStore
from cranefleet.trolley import Trolley
from cranefleet.wiring import build_app

WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def dispatch(console, method: str, path: str, body: dict | None = None):  # noqa: ANN001, ANN201
    request = Request(method=method, path=path, query={}, body=body or {}, body_valid=True)
    return console.dispatch(request)


class ProfileDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limit = Limit()
        self.audit = Audit(50)

    def test_builtin_profiles_cover_every_axis(self) -> None:
        profiles = profiles_snapshot(self.limit)
        self.assertEqual(set(profiles), {PROFILE_OPEN, PROFILE_WALL})
        for ranges in profiles.values():
            self.assertEqual(set(ranges), set(AXES))
        self.assertEqual(active_profile(self.limit), PROFILE_OPEN)

    def test_define_profile_replaces_ranges(self) -> None:
        define_profile(
            self.limit,
            "yard-north",
            {"slew": [-10, 80], "trolley": [0, 30], "hoist": [-5, 40]},
            self.audit,
        )
        ranges = profiles_snapshot(self.limit)["yard-north"]
        self.assertEqual(ranges["slew"], (-10.0, 80.0))
        self.assertEqual(ranges["hoist"], (-5.0, 40.0))
        self.assertEqual(self.audit.latest("limit-profile-define").detail, "yard-north")

    def test_define_profile_rejects_bad_ranges(self) -> None:
        with self.assertRaises(LimitProfileInvalid):
            define_profile(self.limit, "", {"slew": [0, 1], "trolley": [0, 1], "hoist": [0, 1]})
        with self.assertRaises(LimitProfileInvalid):
            define_profile(self.limit, "x", {"slew": [0, 1]})
        with self.assertRaises(LimitProfileInvalid):
            define_profile(
                self.limit, "x", {"slew": [5, 5], "trolley": [0, 1], "hoist": [0, 1]}
            )
        with self.assertRaises(LimitProfileInvalid):
            define_profile(
                self.limit,
                "x",
                {"slew": [0, 1], "trolley": [0, 1], "hoist": [0, 1], "jib": [0, 1]},
            )
        self.assertNotIn("x", profiles_snapshot(self.limit))

    def test_profiles_survive_persistence(self) -> None:
        store = MemoryStore()
        define_profile(
            self.limit,
            "yard-north",
            {"slew": [-10, 80], "trolley": [0, 30], "hoist": [-5, 40]},
        )
        persist_limit(self.limit, store)
        reloaded = Limit()
        restore_limit(reloaded, store)
        self.assertEqual(
            profiles_snapshot(reloaded)["yard-north"]["trolley"], (0.0, 30.0)
        )


class ProfileSwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limit = Limit()
        self.slew = Slew("crane-1")
        self.trolley = Trolley("crane-1")
        self.hoist = Hoist("crane-1")
        self.audit = Audit(50)

    def _activate(self, name: str):
        return activate_profile(
            self.limit, name, self.slew, self.trolley, self.hoist, self.audit
        )

    def test_switch_leaves_an_audited_trail(self) -> None:
        switch = self._activate(PROFILE_WALL)
        self.assertEqual(switch.previous, PROFILE_OPEN)
        self.assertEqual(switch.active, PROFILE_WALL)
        self.assertEqual(switch.sequence, 1)
        self.assertTrue(switch.at)
        self.assertEqual(active_profile(self.limit), PROFILE_WALL)
        entry = self.audit.latest("limit-profile-switch")
        self.assertIsNotNone(entry)
        self.assertIn("open -> wall", entry.detail)
        second = self._activate(PROFILE_OPEN)
        self.assertEqual(second.sequence, 2)
        history = switch_history(self.limit)
        self.assertEqual([item.active for item in history], [PROFILE_WALL, PROFILE_OPEN])

    def test_switch_is_refused_while_an_axis_moves(self) -> None:
        hoist_lower(self.hoist)
        with self.assertRaises(AxesMoving):
            self._activate(PROFILE_WALL)
        self.assertEqual(active_profile(self.limit), PROFILE_OPEN)
        self.assertFalse(is_switching(self.limit))
        self.assertEqual(switch_history(self.limit), [])

    def test_switch_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(LimitProfileUnknown):
            self._activate("nowhere")
        self.assertEqual(active_profile(self.limit), PROFILE_OPEN)

    def test_active_profile_drives_the_envelopes(self) -> None:
        readers = {AXIS_SLEW: lambda: 70.0, AXIS_TROLLEY: lambda: 0.0}
        exceeded = enforce_envelopes(self.limit, active_envelopes(self.limit, readers))
        self.assertEqual(exceeded, [])
        self._activate(PROFILE_WALL)
        exceeded = enforce_envelopes(self.limit, active_envelopes(self.limit, readers))
        self.assertEqual(exceeded, [AXIS_SLEW])

    def test_switch_state_survives_persistence(self) -> None:
        store = MemoryStore()
        self._activate(PROFILE_WALL)
        persist_limit(self.limit, store)
        reloaded = Limit()
        restore_limit(reloaded, store)
        self.assertEqual(active_profile(reloaded), PROFILE_WALL)
        history = switch_history(reloaded)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].active, PROFILE_WALL)
        self.assertEqual(history[0].sequence, 1)


class BatchStopTests(unittest.TestCase):
    """A limit event must stop the whole tripped batch, never half of it."""

    def setUp(self) -> None:
        self.limit = Limit()
        self.slew = Slew("crane-1")
        self.trolley = Trolley("crane-1")
        self.hoist = Hoist("crane-1")

    def test_batch_stops_together_under_one_sequence(self) -> None:
        self.slew.moving = True
        self.trolley.moving = True
        trip(self.limit, AXIS_SLEW)
        trip(self.limit, AXIS_TROLLEY)
        arbitration = arbitrate(self.limit, self.slew, self.trolley, self.hoist)
        self.assertEqual(set(arbitration.stops), {AXIS_SLEW, AXIS_TROLLEY})
        self.assertEqual(arbitration.failed, {})
        self.assertFalse(self.slew.is_moving())
        self.assertFalse(self.trolley.is_moving())
        self.assertTrue(self.limit.is_stopped(AXIS_SLEW))
        self.assertTrue(self.limit.is_stopped(AXIS_TROLLEY))

    def test_one_failing_stop_never_spares_the_rest(self) -> None:
        self.slew.moving = True
        self.trolley.moving = True
        trip(self.limit, AXIS_SLEW)
        trip(self.limit, AXIS_TROLLEY)
        with mock.patch(
            "cranefleet.limit.slew_limit_stop", side_effect=RuntimeError("drive fault")
        ):
            arbitration = arbitrate(self.limit, self.slew, self.trolley, self.hoist)
        # The healthy axis still stopped in the same pass...
        self.assertIn(AXIS_TROLLEY, arbitration.stops)
        self.assertFalse(self.trolley.is_moving())
        self.assertTrue(self.limit.is_stopped(AXIS_TROLLEY))
        # ...while the failed axis is reported, stays tripped and is not
        # flagged stopped, so nothing pretends the batch finished halfway.
        self.assertEqual(set(arbitration.failed), {AXIS_SLEW})
        self.assertTrue(self.limit.is_tripped(AXIS_SLEW))
        self.assertFalse(self.limit.is_stopped(AXIS_SLEW))
        # A later pass retries the failed axis and stops it for good.
        retry = arbitrate(self.limit, self.slew, self.trolley, self.hoist)
        self.assertEqual(set(retry.stops), {AXIS_SLEW})
        self.assertFalse(self.slew.is_moving())
        self.assertTrue(self.limit.is_stopped(AXIS_SLEW))
        self.assertFalse(self.limit.is_tripped(AXIS_SLEW))


class ProfileApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = build_app("memory", web_root=str(WEB_ROOT))
        self.console = self.app.console

    def tearDown(self) -> None:
        self.app.shutdown()

    def test_activate_flow_and_audit_trail(self) -> None:
        status, payload = dispatch(self.console, "POST", "/api/limit/activate", {"name": "wall"})
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["switch"]["active"], "wall")
        self.assertEqual(payload["data"]["limit"]["active"], "wall")
        status, payload = dispatch(self.console, "GET", "/api/limit/profiles")
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["active"], "wall")
        self.assertEqual(len(payload["data"]["switches"]), 1)
        self.assertIn("wall", payload["data"]["profiles"])
        status, payload = dispatch(self.console, "GET", "/api/audit", None)
        switches = [e for e in payload["data"] if e["action"] == "limit-profile-switch"]
        self.assertEqual(len(switches), 1)

    def test_activate_rejected_while_moving(self) -> None:
        dispatch(self.console, "POST", "/api/slew/start")
        status, payload = dispatch(self.console, "POST", "/api/limit/activate", {"name": "wall"})
        self.assertEqual(status, 409)
        self.assertIn("axes moving", payload["error"])
        status, payload = dispatch(self.console, "GET", "/api/limit/profiles")
        self.assertEqual(payload["data"]["active"], "open")

    def test_activate_unknown_or_missing_profile(self) -> None:
        status, _ = dispatch(self.console, "POST", "/api/limit/activate", {"name": "nowhere"})
        self.assertEqual(status, 404)
        status, _ = dispatch(self.console, "POST", "/api/limit/activate", {})
        self.assertEqual(status, 400)

    def test_define_profile_over_api(self) -> None:
        status, payload = dispatch(
            self.console,
            "POST",
            "/api/limit/profile",
            {"name": "yard-north", "ranges": {"slew": [-10, 80], "trolley": [0, 30], "hoist": [-5, 40]}},
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["data"]["profiles"]["yard-north"]["slew"], [-10.0, 80.0])
        status, payload = dispatch(
            self.console,
            "POST",
            "/api/limit/profile",
            {"name": "broken", "ranges": {"slew": [1, 1], "trolley": [0, 1], "hoist": [0, 1]}},
        )
        self.assertEqual(status, 400)

    def test_check_uses_the_active_profile_ranges(self) -> None:
        dispatch(self.console, "POST", "/api/cluster/survey", {"cranes": {"crane-1": {"x": 70, "y": 0}}})
        status, payload = dispatch(self.console, "POST", "/api/limit/check")
        self.assertEqual(payload["data"]["exceeded"], [])
        self.assertEqual(payload["data"]["profile"], "open")
        dispatch(self.console, "POST", "/api/limit/activate", {"name": "wall"})
        status, payload = dispatch(self.console, "POST", "/api/limit/check")
        self.assertEqual(payload["data"]["exceeded"], ["slew"])
        self.assertEqual(payload["data"]["profile"], "wall")
        self.assertIsNotNone(payload["data"]["arbitration"])

    def test_motion_is_refused_during_a_switch(self) -> None:
        self.console.limit.switching = True
        try:
            status, payload = dispatch(self.console, "POST", "/api/slew/start")
            self.assertEqual(status, 409)
            self.assertEqual(payload["error"], "limit profile switch in progress")
            status, payload = dispatch(
                self.console,
                "POST",
                "/api/crane/command",
                {"source": "cab", "action": "swing", "angle": 10},
            )
            self.assertEqual(status, 409)
            # A stop command always goes through, even mid-switch.
            status, _ = dispatch(
                self.console, "POST", "/api/crane/command", {"source": "cab", "action": "stop"}
            )
            self.assertEqual(status, 200)
        finally:
            self.console.limit.switching = False


class ProfileBootTests(unittest.TestCase):
    def test_active_profile_survives_a_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = build_app(directory, web_root=str(WEB_ROOT))
            dispatch(first.console, "POST", "/api/limit/activate", {"name": "wall"})
            first.shutdown()

            second = build_app(directory, web_root=str(WEB_ROOT))
            status, payload = dispatch(second.console, "GET", "/api/limit/profiles")
            self.assertEqual(payload["data"]["active"], "wall")
            self.assertEqual(len(payload["data"]["switches"]), 1)
            self.assertTrue(
                any(
                    entry.action == "limit-profile-switch"
                    for entry in second.console.audit.snapshot()
                )
            )
            second.shutdown()


if __name__ == "__main__":
    unittest.main()

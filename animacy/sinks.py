"""Robot sinks — where retargeted joint frames go.

Every sink takes joint values in the robot's ``ROBOT.md`` units (the same
numbers ``retarget`` produces and the exporters write) and owns the vendor
conversion. Hardware-verified on 2026-08-26 for :class:`ReachyDaemonSink`
(see ``docs/evidence/reachy_sim2real_20260826.md``).
"""
from __future__ import annotations

import io
import math
import time
from typing import Dict, Optional

import pandas as pd

from .profile import Profile


class Sink:
    name = "base"

    def prepare(self) -> None: ...
    def send(self, joints: Dict[str, float]) -> None: ...
    def neutral(self, duration: float = 1.5) -> None: ...
    def close(self) -> None: ...


class ReachyDaemonSink(Sink):
    """Reachy Mini daemon HTTP API (Wireless unit or `reachy-mini-daemon` on a laptop).

    ``POST /api/move/set_target`` with ``FullBodyTarget``; head pose x/y/z in
    metres and roll/pitch/yaw in radians, antennas ``[left, right]`` radians,
    body yaw radians. Canonical +pitch is UP; the daemon's +pitch is nose-down,
    so pitch is negated here — exactly once, here.
    """

    name = "reachy_daemon"

    def __init__(self, url: str = "http://192.168.1.60:8000", pitch_sign: float = -1.0, wake: bool = True):
        import requests

        self.base = url.rstrip("/")
        self.s = requests.Session()
        self.pitch_sign = pitch_sign
        self.wake = wake

    def _get(self, path, timeout=4.0):
        # The daemon occasionally drops an idle keep-alive socket; one retry on a fresh connection.
        for attempt in (0, 1):
            try:
                return self.s.get(self.base + path, timeout=timeout).json()
            except Exception:  # noqa: BLE001
                if attempt:
                    raise
                self.s.close()

    def _post(self, path, body=None, timeout=6.0):
        for attempt in (0, 1):
            try:
                r = self.s.post(self.base + path, json=body, timeout=timeout)
                r.raise_for_status()
                return r.json() if r.content else {}
            except Exception:  # noqa: BLE001
                if attempt:
                    raise
                self.s.close()

    def prepare(self) -> None:
        mode = str(self._get("/api/motors/status").get("mode", "")).lower()
        if mode != "enabled":
            self._post("/api/motors/set_mode/enabled")
            time.sleep(0.5)
        if self.wake:
            self._post("/api/move/play/wake_up", timeout=20)
            for _ in range(60):
                time.sleep(0.25)
                if not self._get("/api/move/running"):
                    break

    def target(self, j: Dict[str, float]) -> dict:
        return {
            "target_head_pose": {
                "x": j.get("head_x", 0.0) / 1000.0, "y": j.get("head_y", 0.0) / 1000.0, "z": j.get("head_z", 0.0) / 1000.0,
                "roll": math.radians(j.get("head_roll", 0.0)),
                "pitch": self.pitch_sign * math.radians(j.get("head_pitch", 0.0)),
                "yaw": math.radians(j.get("head_yaw", 0.0)),
            },
            "target_antennas": [math.radians(j.get("antenna_left", 0.0)), math.radians(j.get("antenna_right", 0.0))],
            "target_body_yaw": math.radians(j.get("body_yaw", 0.0)),
        }

    def send(self, joints: Dict[str, float]) -> None:
        self._post("/api/move/set_target", self.target(joints), timeout=1.5)

    def read_back(self) -> dict:
        return {"head_pose": self._get("/api/state/present_head_pose"),
                "antennas": self._get("/api/state/present_antenna_joint_positions"),
                "body_yaw": self._get("/api/state/present_body_yaw")}

    def neutral(self, duration: float = 1.5) -> None:
        self._post("/api/move/goto", {"head_pose": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0},
                                      "antennas": [0.0, 0.0], "body_yaw": 0.0, "duration": duration,
                                      "interpolation": "minjerk"}, timeout=10)

    def sleep(self) -> None:
        self._post("/api/move/play/goto_sleep", timeout=20)


class AutonomousHalSink(Sink):
    """Autonomous OS HAL (:5001). Two paths:

    * clip: ``upload(table, name)`` then ``play(name)`` — the vendor's own
      playback loop resamples to 30 Hz and stretches over-speed segments.
    * live: ``send(joints)`` → ``POST /servo/move`` with a short duration; the
      HAL safety gate stretches anything over SAFETY.md ``motion.max_speed``.
    UNVERIFIED on hardware (no lamp on hand yet); the request shapes are from
    hal/models.py and hal/routes/servo.py in autonomous-os.
    """

    name = "autonomous_hal"

    def __init__(self, url: str = "http://127.0.0.1:5001", profile: Optional[Profile] = None, live_duration: float = 0.05):
        import requests

        self.base = url.rstrip("/")
        self.s = requests.Session()
        self.profile = profile
        self.live_duration = live_duration

    def upload(self, table: pd.DataFrame, name: str) -> None:
        from .export import to_autonomous_os_csv

        text = to_autonomous_os_csv(table, self.profile)
        r = self.s.post(self.base + "/servo/upload", files={"file": (f"{name}.csv", io.BytesIO(text.encode()), "text/csv")},
                        data={"recording_name": name}, timeout=15)
        r.raise_for_status()

    def play(self, name: str) -> None:
        self.s.post(self.base + "/servo/play", json={"recording": name}, timeout=10).raise_for_status()

    def send(self, joints: Dict[str, float]) -> None:
        self.s.post(self.base + "/servo/move", json={"positions": {f"{k}.pos": v for k, v in joints.items()},
                                                    "duration": self.live_duration}, timeout=1.5)

    def neutral(self, duration: float = 1.5) -> None:
        self.s.post(self.base + "/servo/play", json={"recording": "idle"}, timeout=10)

    def stop(self) -> None:
        self.s.post(self.base + "/servo/stop", timeout=5)


class PrintSink(Sink):
    name = "print"

    def __init__(self, every: int = 30):
        self.every, self.n = every, 0

    def send(self, joints: Dict[str, float]) -> None:
        self.n += 1
        if self.n % self.every == 0:
            print({k: round(v, 1) for k, v in joints.items()})


def make_sink(profile: Profile, override: Optional[str] = None, url: Optional[str] = None) -> Sink:
    kind = override or profile.runtime.kind
    u = url or profile.runtime.url
    if kind in ("reachy_sdk", "reachy_daemon"):
        return ReachyDaemonSink(u or "http://192.168.1.60:8000")
    if kind == "autonomous_os_hal":
        return AutonomousHalSink(u or "http://127.0.0.1:5001", profile)
    return PrintSink()


def stream_table(table: pd.DataFrame, profile: Profile, sink: Sink, rate_hz: Optional[float] = None,
                 slew_deg: float = 6.0, slew_mm: float = 4.0) -> None:
    """Play a retargeted joint table on a sink at the robot rate with a slew clamp."""
    rate = rate_hz or profile.rate_hz
    dt = 1.0 / rate
    cur: Dict[str, float] = {}
    for i in range(len(table)):
        row = table.iloc[i]
        cmd = {}
        for j in profile.joints:
            v = float(row[j.name])
            if j.name in cur:
                cap = slew_mm if j.unit == "mm" else (slew_deg * 3 if "antenna" in j.name else slew_deg)
                v = cur[j.name] + max(-cap, min(cap, v - cur[j.name]))
            cmd[j.name] = v
        cur = cmd
        t0 = time.perf_counter()
        try:
            sink.send(cmd)
        except Exception as e:  # noqa: BLE001
            print("sink error:", e)
        rem = dt - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)

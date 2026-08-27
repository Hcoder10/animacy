"""``animacy.mirror`` — live mirror: video / webcam -> robot in real time.

    python -m animacy.mirror --source data/raw/talk.webm --robot reachy_mini --sink print
    python -m animacy.mirror --source 0 --robot reachy_mini --sink reachy_daemon --url http://192.168.1.60:8000 --preview
    options: --mode default|puppet  --speed 1.0  --duration s  --arm right|left|none
             --neutral-seconds 1.0 (0 = whole-file prepass median, files only)  --pose-every N  --hold 0.5

Data path per frame: ``capture.Trackers`` (FaceLandmarker + PoseLandmarker,
VIDEO mode) -> :class:`ChannelAssembler` (same math as ``animacy.capture``:
neutral zeroing of head/gaze/brows/torso, absolute arm) -> a single-slot
mailbox -> the **sender thread** ticking at ``profile.rate_hz`` on a steady
clock: ``LiveRetargeter.step(channels, dt)`` -> ``sink.send(joints)``.

Two clocks, deliberately decoupled:

* The **tracker** (main thread, so ``--preview`` can draw) runs as fast as the
  source delivers. A file source is paced by its own timestamps x ``--speed``
  (:class:`VideoPacer`): a 25 fps clip plays in real time; frames that are
  already late by more than ~1.5 source periods are skipped without tracking
  (``late_skipped``). A webcam is never paced.
* The **sender** (:class:`Ticker`) fires every ``1/rate_hz`` regardless. If no
  new tracker sample arrived it re-sends the last channels (``dup`` — frame
  duplicated, the retargeter keeps smoothing); if several arrived, only the
  newest is used (``dropped``). Missed ticks are skipped, never burst.

Neutral pose: webcam / ``--neutral-seconds N`` = median of the first N
seconds of face-valid frames (rest is sent until then); ``--neutral-seconds
0`` on a file = a full tracking prepass over the file (cached under
``data/debug/neutral_cache/``). When the face drops out, the last channels
are held for ``--hold`` seconds, then decay to rest via the retargeter.
``speaking`` is not computed live (no audio path here) and is sent as 0.

Pose runs every ``--pose-every`` frames (default 1); if the tracker cannot
keep ~90% of the robot rate over the first 2 s, it switches to every 2nd
frame automatically and holds torso/arm values between (logged).

Status line once per second: tracker fps, face_valid rate, send p50 latency,
dropped / dup / late-skipped / ticks-skipped, last sent yaw & pitch. With the
Reachy daemon sink, ``read_back()`` is polled every ``--readback-every`` s in
the status thread and printed next to what was sent, and a JSON log of those
pairs goes to ``data/debug/mirror_<stamp>.json``.

Ctrl-C (or EOF / ``--duration``): the sender stops, ``sink.neutral()`` runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import capture_math as cm
from .schema import ARM_CHANNELS, CHANNELS, FACE_CHANNELS, TORSO_CHANNELS

FACE_VALUE_KEYS = ["gaze_yaw", "gaze_pitch", "brow_l", "brow_r", "brow_furrow",
                   "eye_open_l", "eye_open_r", "mouth_open", "smile"]


# ------------------------------------------------------------------ clocks & pacing
class Clock:
    """Wall clock; tests substitute a fake with the same two methods."""

    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, s: float) -> None:
        if s > 0:
            time.sleep(s)


class VideoPacer:
    """Pace a file source to wall time: source second ``t`` is due at ``t0 + t/speed``."""

    def __init__(self, clock: Clock, speed: float = 1.0, late_tolerance: float = 0.06):
        self.clock = clock
        self.speed = max(float(speed), 1e-3)
        self.late_tolerance = late_tolerance
        self.t0: Optional[float] = None
        self.late_skipped = 0

    def due_at(self, t_src: float) -> float:
        if self.t0 is None:
            self.t0 = self.clock.now() - t_src / self.speed
        return self.t0 + t_src / self.speed

    def lateness(self, t_src: float) -> float:
        """Seconds the frame at ``t_src`` is already late (negative = early)."""
        return self.clock.now() - self.due_at(t_src)

    def wait_or_skip(self, t_src: float) -> bool:
        """Sleep until the frame is due and return True; return False (skip) if it is
        already more than ``late_tolerance`` late."""
        late = self.lateness(t_src)
        if late > self.late_tolerance:
            self.late_skipped += 1
            return False
        if late < 0:
            self.clock.sleep(-late)
        return True


class Ticker:
    """Steady ``rate_hz`` schedule. ``wait()`` blocks until the next tick and returns the
    elapsed schedule time (a multiple of the period when ticks were missed)."""

    def __init__(self, clock: Clock, rate_hz: float):
        self.clock = clock
        self.period = 1.0 / float(rate_hz)
        self.t0: Optional[float] = None
        self.k = 0
        self.skipped = 0

    def wait(self) -> float:
        now = self.clock.now()
        if self.t0 is None:
            self.t0 = now
            return self.period
        prev_k = self.k
        self.k += 1
        target = self.t0 + self.k * self.period
        if now > target + self.period:
            missed = int((now - target) // self.period)
            self.k += missed
            self.skipped += missed
            target = self.t0 + self.k * self.period
        if now < target:
            self.clock.sleep(target - now)
        return (self.k - prev_k) * self.period


@dataclass
class Sample:
    t_src: float
    channels: Dict[str, float]
    face_ok: bool
    seq: int = 0


class LatestSlot:
    """Single-slot mailbox. The producer overwrites; ``take`` says whether the item is new.
    An item overwritten before it was taken counts as ``dropped``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._item: Optional[Sample] = None
        self._seq = 0
        self._taken = 0
        self.dropped = 0

    def put(self, item: Sample) -> None:
        with self._lock:
            if self._item is not None and self._seq > self._taken:
                self.dropped += 1
            self._seq += 1
            item.seq = self._seq
            self._item = item

    def take(self) -> Tuple[Optional[Sample], bool]:
        with self._lock:
            is_new = self._seq > self._taken
            self._taken = self._seq
            return self._item, is_new


# ------------------------------------------------------------------ channels
def rest_channels() -> Dict[str, float]:
    """All-zero canonical channels (the retargeter maps them to every joint's rest)."""
    out = {c: 0.0 for c in CHANNELS}
    out["eye_open_l"] = out["eye_open_r"] = 0.6
    return out


class ChannelAssembler:
    """Raw tracker sample -> neutral-relative canonical channels, with neutral capture and hold.

    Same math as ``capture.build_frames`` (``relative_head``, ``face_channels_relative``,
    torso minus its neutral, arm absolute) applied per frame.
    """

    def __init__(self, neutral_seconds: float = 1.0, hold_s: float = 0.5, neutral: Optional[Dict] = None):
        self.neutral_seconds = neutral_seconds
        self.hold_s = hold_s
        self.neutral: Optional[Dict] = neutral
        self._buf: List[Dict] = []
        self._t_first_valid: Optional[float] = None
        self._last_face: Optional[Tuple[float, Dict[str, float]]] = None
        self._last_torso: Optional[Tuple[float, Dict[str, float]]] = None
        self._last_arm: Optional[Tuple[float, Dict[str, float]]] = None

    @property
    def calibrated(self) -> bool:
        return self.neutral is not None

    def _finish_neutral(self) -> None:
        ang = np.array([s["head_angles"] for s in self._buf]).reshape(-1, 3)
        tr = np.array([s["head_trans"] for s in self._buf]).reshape(-1, 3)
        n = cm.neutral_pose(ang, tr)
        n["face_raw"] = {k: float(np.median([s["face_raw"][k] for s in self._buf])) for k in cm.RAW_FACE_KEYS}
        tors = [s["torso_vals"] for s in self._buf if s.get("pose_ok")]
        n["torso_deg"] = ([float(np.median([t[c] for t in tors])) for c in TORSO_CHANNELS] if tors else [0.0, 0.0, 0.0])
        n["n_frames"] = len(self._buf)
        n["seconds"] = self.neutral_seconds
        self.neutral = n
        self._buf = []

    def update(self, s: Dict) -> Optional[Dict[str, float]]:
        """Return channels for this sample, or None while the neutral window is still filling."""
        t = float(s["t"])
        if self.neutral is None:
            if s.get("face_ok"):
                if self._t_first_valid is None:
                    self._t_first_valid = t
                self._buf.append(s)
                if t - self._t_first_valid >= self.neutral_seconds and len(self._buf) >= 3:
                    self._finish_neutral()
            if self.neutral is None:
                return None
        ch = rest_channels()
        for c in FACE_CHANNELS + TORSO_CHANNELS + ARM_CHANNELS:
            ch[c] = float("nan")

        if s.get("face_ok"):
            r_body = cm.head_rotmat_from_angles_deg(*s["head_angles"])
            yaw, pitch, roll, x, y, z = cm.relative_head(r_body, s["head_trans"], self.neutral)
            fv = cm.face_channels_relative(s["face_raw"], self.neutral.get("face_raw", {}))
            face = {"head_yaw": yaw, "head_pitch": pitch, "head_roll": roll, "head_x": x, "head_y": y, "head_z": z}
            face.update({k: fv[k] for k in FACE_VALUE_KEYS})
            self._last_face = (t, face)
        if self._last_face is not None and t - self._last_face[0] <= self.hold_s:
            ch.update(self._last_face[1])
            ch["face_valid"] = 1.0 if s.get("face_ok") else 0.0

        if s.get("pose_ok"):
            torso = {c: s["torso_vals"][c] - n for c, n in zip(TORSO_CHANNELS, self.neutral.get("torso_deg", [0, 0, 0]))}
            self._last_torso = (t, torso)
        if self._last_torso is not None and (s.get("pose_skipped") or t - self._last_torso[0] <= self.hold_s):
            ch.update(self._last_torso[1])

        if s.get("arm_ok"):
            self._last_arm = (t, dict(s["arm_vals"]))
        if self._last_arm is not None and (s.get("pose_skipped") or t - self._last_arm[0] <= self.hold_s):
            ch.update(self._last_arm[1])
            ch["arm_valid"] = 1.0 if s.get("arm_ok") else 0.0

        for c in FACE_CHANNELS + TORSO_CHANNELS + ARM_CHANNELS:
            v = ch[c]
            if not (isinstance(v, float) and math.isnan(v)):
                lo, hi = cm_bounds(c)
                ch[c] = min(max(v, lo), hi)
        return ch


def cm_bounds(c: str) -> Tuple[float, float]:
    from .schema import BOUNDS

    return BOUNDS[c]


# ------------------------------------------------------------------ sender
@dataclass
class SenderStats:
    sent: int = 0
    dup: int = 0
    errors: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    last_joints: Dict[str, float] = field(default_factory=dict)
    last_channels: Dict[str, float] = field(default_factory=dict)

    def p50_ms(self, window: int = 60) -> float:
        w = self.latencies_ms[-window:]
        return statistics.median(w) if w else float("nan")


class Sender:
    """Ticks at the robot rate: latest channels -> LiveRetargeter -> sink."""

    def __init__(self, retargeter, sink, slot: LatestSlot, rate_hz: float, clock: Optional[Clock] = None):
        self.rt = retargeter
        self.sink = sink
        self.slot = slot
        self.rate_hz = rate_hz
        self.clock = clock or Clock()
        self.ticker = Ticker(self.clock, rate_hz)
        self.stats = SenderStats()
        self.stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def tick(self) -> Dict[str, float]:
        """One step: wait for the tick, take the latest sample, retarget, send."""
        dt = self.ticker.wait()
        item, is_new = self.slot.take()
        channels = item.channels if item is not None else rest_channels()
        if item is not None and not is_new:
            self.stats.dup += 1
        joints = self.rt.step(channels, dt)
        t0 = self.clock.now()
        try:
            self.sink.send(joints)
        except Exception as exc:  # noqa: BLE001 - keep the loop alive, count it
            self.stats.errors += 1
            if self.stats.errors <= 3:
                print(f"sink error: {type(exc).__name__}: {exc}", flush=True)
        self.stats.latencies_ms.append((self.clock.now() - t0) * 1000.0)
        if len(self.stats.latencies_ms) > 3000:
            del self.stats.latencies_ms[:-1500]
        self.stats.sent += 1
        self.stats.last_joints = joints
        self.stats.last_channels = channels
        return joints

    def run(self) -> None:
        while not self.stop.is_set():
            self.tick()

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, daemon=True, name="animacy-sender")
        self._thread.start()

    def join(self, timeout: float = 2.0) -> None:
        self.stop.set()
        if self._thread is not None:
            self._thread.join(timeout)


# ------------------------------------------------------------------ neutral prepass (files)
def prepass_neutral(source: str, arm: str) -> Dict:
    """Whole-file tracking pass -> neutral pose (cached by path/size/mtime)."""
    from . import capture as cap

    st = os.stat(source)
    key = hashlib.sha1(f"{os.path.abspath(source)}|{st.st_size}|{int(st.st_mtime)}|{arm}".encode()).hexdigest()[:16]
    cache_dir = os.path.join(os.path.dirname(cap.models_dir()), "debug", "neutral_cache")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        print(f"neutral: cached prepass {path}")
        return json.load(open(path, encoding="utf-8"))
    print("neutral: whole-file prepass (tracking every frame once; cached afterwards)", flush=True)
    samples, _, _ = cap.run_source(source, arm, 0.0, False, want_audio=False)
    _, extra = cap.build_frames(samples, 0.0)
    neutral = extra["neutral"]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(neutral, fh)
    return neutral


# ------------------------------------------------------------------ main loop
class Mirror:
    def __init__(self, source: str, profile, mode: str, sink, speed: float = 1.0, preview: bool = False,
                 neutral_seconds: float = 1.0, duration: float = 0.0, arm: str = "right", pose_every: int = 1,
                 hold_s: float = 0.5, status_every: float = 1.0, readback_every: float = 5.0,
                 clock: Optional[Clock] = None, log_path: Optional[str] = None, start: float = 0.0):
        self.source, self.profile, self.mode, self.sink = source, profile, mode, sink
        self.speed, self.preview, self.neutral_seconds, self.duration = speed, preview, neutral_seconds, duration
        self.start = start
        self.arm, self.pose_every, self.hold_s = arm, max(1, int(pose_every)), hold_s
        self.status_every, self.readback_every = status_every, readback_every
        self.clock = clock or Clock()
        self.slot = LatestSlot()
        self.stop = threading.Event()
        self.log_path = log_path
        self.readback_log: List[Dict] = []
        self.track_ms: List[float] = []
        self.tracked = 0
        self.face_hits = 0
        self._win: List[Tuple[float, bool]] = []  # (wall time, face_ok) for the 1 s window

    # ---- status thread
    def _status_loop(self, sender: Sender) -> None:
        last_rb = self.clock.now()
        can_read = hasattr(self.sink, "read_back")
        while not self.stop.is_set():
            self.clock.sleep(self.status_every)
            now = self.clock.now()
            win = [w for w in self._win if now - w[0] <= 1.0]
            fps = len(win)
            fv = (sum(1 for w in win if w[1]) / fps) if fps else 0.0
            j = sender.stats.last_joints
            line = (f"[{now - self.t_wall0:6.1f}s] tracker {fps:4.1f} fps  face_valid {fv:4.0%}  "
                    f"send p50 {sender.stats.p50_ms():5.1f} ms  sent {sender.stats.sent}  dup {sender.stats.dup}  "
                    f"dropped {self.slot.dropped}  late_skipped {self.pacer.late_skipped if self.pacer else 0}  "
                    f"ticks_skipped {sender.ticker.skipped}  errors {sender.stats.errors}  "
                    f"| sent yaw {j.get('head_yaw', 0):+6.1f} pitch {j.get('head_pitch', 0):+6.1f}")
            if can_read and now - last_rb >= self.readback_every:
                last_rb = now
                try:
                    rb = self.sink.read_back()
                    hp = rb.get("head_pose", {}) or {}
                    meas = {k: math.degrees(float(hp[k])) for k in ("yaw", "pitch", "roll") if k in hp}
                    sent = {k: float(j.get(f"head_{k}", 0.0)) for k in ("yaw", "pitch", "roll")}
                    # the daemon's own pitch frame is nose-down; the sink negates once, so compare in the daemon frame
                    sent_daemon = dict(sent)
                    sent_daemon["pitch"] = -sent["pitch"] if getattr(self.sink, "pitch_sign", 1.0) < 0 else sent["pitch"]
                    line += (f"\n           READBACK yaw {meas.get('yaw', float('nan')):+6.1f} pitch(daemon) {meas.get('pitch', float('nan')):+6.1f} "
                             f"roll {meas.get('roll', float('nan')):+6.1f}  vs sent yaw {sent_daemon['yaw']:+6.1f} "
                             f"pitch(daemon) {sent_daemon['pitch']:+6.1f} roll {sent_daemon['roll']:+6.1f}")
                    self.readback_log.append({"t": now - self.t_wall0, "sent_deg": sent, "sent_daemon_frame_deg": sent_daemon,
                                              "measured_daemon_frame_deg": meas, "raw": rb})
                except Exception as exc:  # noqa: BLE001
                    line += f"\n           READBACK failed: {type(exc).__name__}: {exc}"
            print(line, flush=True)

    # ---- tracker (main thread)
    def run(self) -> int:
        import cv2

        from . import capture as cap
        from .retarget import LiveRetargeter

        cap_obj, is_cam = cap.open_source(self.source)
        if not cap_obj.isOpened():
            print(f"cannot open source {self.source!r}")
            return 1
        src_fps = float(cap_obj.get(cv2.CAP_PROP_FPS) or 0.0) or (30.0 if not is_cam else 0.0)
        start_idx = 0
        if not is_cam and self.start > 0:
            start_idx = int(round(self.start * src_fps))
            cap_obj.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        neutral = None
        if not is_cam and self.neutral_seconds <= 0:
            neutral = prepass_neutral(self.source, self.arm)
        elif self.neutral_seconds <= 0:
            print("webcam has no prepass; using --neutral-seconds 1.0")
            self.neutral_seconds = 1.0
        assembler = ChannelAssembler(self.neutral_seconds, self.hold_s, neutral=neutral)
        trackers = cap.Trackers(want_pose=True)
        self.pacer = VideoPacer(self.clock, self.speed) if not is_cam else None

        rt = LiveRetargeter(self.profile, self.mode)
        sender = Sender(rt, self.sink, self.slot, self.profile.rate_hz, self.clock)
        print(f"mirror: {self.source} -> {self.profile.name}/{self.mode} -> {self.sink.name} @ {self.profile.rate_hz:.0f} Hz"
              f"  (src {src_fps:.2f} fps, speed x{self.speed}, pose every {self.pose_every})", flush=True)
        if not assembler.calibrated:
            print(f"neutral: hold still and look at the camera for {self.neutral_seconds:.1f}s (rest is sent meanwhile)", flush=True)
        self.sink.prepare()
        self.t_wall0 = self.clock.now()
        sender.start()
        status = threading.Thread(target=self._status_loop, args=(sender,), daemon=True, name="animacy-status")
        status.start()

        idx, last_pos = start_idx, -1.0
        adapted = False
        try:
            while not self.stop.is_set():
                # cheap catch-up: when far behind on a file, grab (decode-skip) without tracking
                if self.pacer is not None and idx > start_idx and self.pacer.lateness((idx) / src_fps) > 0.25:
                    if not cap_obj.grab():
                        break
                    self.pacer.late_skipped += 1
                    idx += 1
                    continue
                ok, frame = cap_obj.read()
                if not ok:
                    break
                if is_cam:
                    t = self.clock.now() - self.t_wall0
                else:
                    pos = float(cap_obj.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                    t_idx = idx / src_fps
                    t = pos if (pos > last_pos and abs(pos - t_idx) < 0.5) else t_idx
                    last_pos = max(last_pos, pos)
                idx += 1
                if self.duration > 0 and t > self.duration:
                    break
                if self.pacer is not None and not self.pacer.wait_or_skip(t):
                    continue
                want_pose = (self.tracked % self.pose_every) == 0
                t0 = self.clock.now()
                s = trackers.detect(frame, t, arm=self.arm, want_pose=want_pose)
                self.track_ms.append((self.clock.now() - t0) * 1000.0)
                self.tracked += 1
                self.face_hits += int(s["face_ok"])
                now = self.clock.now()
                self._win.append((now, bool(s["face_ok"])))
                if len(self._win) > 200:
                    del self._win[:-120]
                channels = assembler.update(s)
                if channels is None:
                    if assembler.calibrated:
                        pass
                    else:
                        channels = rest_channels()
                self.slot.put(Sample(t, channels, bool(s["face_ok"])))
                # adapt: if we cannot keep up with the robot rate, run pose on every 2nd frame
                if not adapted and now - self.t_wall0 > 2.0 and self.pose_every == 1:
                    adapted = True
                    recent = [w for w in self._win if now - w[0] <= 1.0]
                    if len(recent) < 0.9 * self.profile.rate_hz and (self.pacer is None or self.pacer.late_skipped > 0):
                        self.pose_every = 2
                        print(f"tracker at {len(recent)} fps < 90% of {self.profile.rate_hz:.0f} Hz: pose every 2nd frame from now", flush=True)
                if self.preview:
                    rel = channels if channels else None
                    img = cap.draw_overlay(frame, s, rel=rel, title=f"t={t:6.2f}s  {'CALIBRATING' if not assembler.calibrated else ''}")
                    cv2.imshow("animacy mirror (q to stop)", img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            print("\ninterrupted", flush=True)
        finally:
            self.stop.set()
            sender.join()
            cap_obj.release()
            if self.preview:
                cv2.destroyAllWindows()
            try:
                self.sink.neutral()
            except Exception as exc:  # noqa: BLE001
                print(f"sink.neutral failed: {type(exc).__name__}: {exc}")
            try:
                self.sink.close()
            except Exception:  # noqa: BLE001
                pass
            self._summary(sender, assembler)
        return 0

    def _summary(self, sender: Sender, assembler: ChannelAssembler) -> None:
        wall = self.clock.now() - self.t_wall0
        fps = self.tracked / wall if wall > 0 else 0.0
        print(f"done: {wall:.1f}s wall, tracked {self.tracked} frames ({fps:.1f} fps, track p50 "
              f"{statistics.median(self.track_ms) if self.track_ms else float('nan'):.1f} ms), face_valid "
              f"{(self.face_hits / self.tracked) if self.tracked else 0:.0%}, sent {sender.stats.sent} "
              f"(dup {sender.stats.dup}, dropped {self.slot.dropped}, late_skipped {self.pacer.late_skipped if self.pacer else 0}, "
              f"ticks_skipped {sender.ticker.skipped}, errors {sender.stats.errors}), send p50 {sender.stats.p50_ms(10**6):.1f} ms",
              flush=True)
        if self.log_path and (self.readback_log or sender.stats.sent):
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "w", encoding="utf-8") as fh:
                json.dump({"source": self.source, "robot": self.profile.name, "mode": self.mode, "sink": self.sink.name,
                           "speed": self.speed, "wall_s": wall, "tracked": self.tracked, "tracker_fps": fps,
                           "track_ms_p50": statistics.median(self.track_ms) if self.track_ms else None,
                           "face_valid_frac": (self.face_hits / self.tracked) if self.tracked else 0.0,
                           "sent": sender.stats.sent, "dup": sender.stats.dup, "dropped": self.slot.dropped,
                           "late_skipped": self.pacer.late_skipped if self.pacer else 0, "ticks_skipped": sender.ticker.skipped,
                           "send_p50_ms": sender.stats.p50_ms(10**6), "neutral": assembler.neutral,
                           "readback": self.readback_log}, fh, indent=1, default=float)
            print(f"log: {self.log_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="animacy.mirror", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index or video file")
    p.add_argument("--robot", required=True)
    p.add_argument("--mode", default="default")
    p.add_argument("--sink", default=None, help="reachy_daemon | autonomous_os_hal | print (default: the profile's runtime.kind)")
    p.add_argument("--url", default=None)
    p.add_argument("--speed", type=float, default=1.0, help="file playback speed multiplier")
    p.add_argument("--duration", type=float, default=0.0, help="stop after this many source seconds (0 = EOF / q / Ctrl-C)")
    p.add_argument("--start", type=float, default=0.0, help="seek a file source to this many seconds first")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--arm", default="right", choices=["right", "left", "none"])
    p.add_argument("--neutral-seconds", type=float, default=1.0, help="0 = whole-file prepass median (files only)")
    p.add_argument("--pose-every", type=int, default=1)
    p.add_argument("--hold", type=float, default=0.5, help="seconds to hold the last channels after the face drops out")
    p.add_argument("--readback-every", type=float, default=5.0)
    p.add_argument("--log", default=None, help="JSON log path (default data/debug/mirror_<stamp>.json)")
    return p


def run_from_args(a) -> int:
    """Entry point for ``animacy.cli`` (same attribute names as :func:`build_parser`)."""
    from .profile import find_robot
    from .sinks import make_sink

    profile = find_robot(a.robot)
    errs = profile.check()
    if errs:
        print("profile fails animacy check:", *errs, sep="\n  - ")
        return 1
    sink = make_sink(profile, a.sink, a.url)
    log = a.log or os.path.join("data", "debug", f"mirror_{time.strftime('%Y%m%d_%H%M%S')}.json")
    m = Mirror(str(a.source), profile, a.mode, sink, speed=a.speed, preview=a.preview, neutral_seconds=a.neutral_seconds,
               duration=a.duration, arm=a.arm, pose_every=a.pose_every, hold_s=a.hold, readback_every=a.readback_every,
               log_path=log, start=a.start)
    return m.run()


def main(argv=None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())

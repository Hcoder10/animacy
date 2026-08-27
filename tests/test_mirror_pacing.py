"""Pacing / drop / duplicate logic of ``animacy.mirror`` — no models, no hardware."""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import pytest

from animacy import capture_math as cm
from animacy.mirror import (ChannelAssembler, LatestSlot, Sample, Sender, Ticker, VideoPacer,
                            rest_channels)


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t
        self.slept = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += max(s, 0.0)


class RecordingSink:
    name = "recording"

    def __init__(self, fail_every: int = 0):
        self.sent = []
        self.neutral_calls = 0
        self.fail_every = fail_every

    def prepare(self): ...

    def send(self, joints):
        if self.fail_every and (len(self.sent) + 1) % self.fail_every == 0:
            self.sent.append(None)
            raise ConnectionError("boom")
        self.sent.append(dict(joints))

    def neutral(self, duration: float = 1.5):
        self.neutral_calls += 1

    def close(self): ...


class PassThroughRetargeter:
    """Stands in for LiveRetargeter: joint = channel, so what was sent is inspectable."""

    def __init__(self):
        self.dts = []

    def step(self, channels, dt):
        self.dts.append(dt)
        v = channels.get("head_yaw", 0.0)
        return {"head_yaw": 0.0 if (isinstance(v, float) and math.isnan(v)) else float(v)}


# ---------------------------------------------------------------- Ticker
def test_ticker_holds_a_steady_schedule():
    clk = FakeClock()
    tk = Ticker(clk, rate_hz=10.0)
    assert tk.wait() == pytest.approx(0.1)  # first call sets t0, no sleep
    for k in range(1, 6):
        clk.t += 0.03  # some work
        dt = tk.wait()
        assert dt == pytest.approx(0.1)
        assert clk.now() == pytest.approx(k * 0.1)
    assert tk.skipped == 0


def test_ticker_skips_missed_ticks_instead_of_bursting():
    clk = FakeClock()
    tk = Ticker(clk, rate_hz=10.0)
    tk.wait()
    clk.t += 0.35  # the sink stalled for 3.5 periods
    dt = tk.wait()
    assert dt == pytest.approx(0.3)  # ticks at 0.1 and 0.2 were skipped; we are now on the 0.3 tick
    assert tk.skipped == 2
    assert clk.slept == []  # no sleep: already past the 0.3 tick
    dt = tk.wait()  # next tick is 0.4: sleeps 0.05
    assert dt == pytest.approx(0.1) and clk.now() == pytest.approx(0.4)


# ---------------------------------------------------------------- VideoPacer
def test_video_pacer_plays_25fps_in_real_time():
    clk = FakeClock(100.0)
    pacer = VideoPacer(clk, speed=1.0)
    for i in range(5):
        t_src = i * 0.04
        assert pacer.wait_or_skip(t_src)
        assert clk.now() == pytest.approx(100.0 + t_src)
    assert pacer.late_skipped == 0


def test_video_pacer_speed_and_late_skip():
    clk = FakeClock()
    pacer = VideoPacer(clk, speed=2.0)
    assert pacer.wait_or_skip(0.0)
    assert pacer.wait_or_skip(1.0) and clk.now() == pytest.approx(0.5)  # 2x: 1 s of video in 0.5 s
    clk.t += 0.5  # tracker stalled; the frame at 1.04 was due at 0.52, now 1.0 -> 0.48 s late
    assert not pacer.wait_or_skip(1.04)
    assert pacer.late_skipped == 1
    assert pacer.wait_or_skip(2.2) and clk.now() == pytest.approx(1.1)  # catches up and paces again


# ---------------------------------------------------------------- LatestSlot
def test_latest_slot_counts_drops_and_duplicates():
    slot = LatestSlot()
    item, new = slot.take()
    assert item is None and not new
    for k in range(3):
        slot.put(Sample(t_src=k * 0.01, channels={"head_yaw": float(k)}, face_ok=True))
    item, new = slot.take()
    assert new and item.channels["head_yaw"] == 2.0 and item.seq == 3
    assert slot.dropped == 2
    item, new = slot.take()
    assert not new and item.channels["head_yaw"] == 2.0  # holds the last sample


# ---------------------------------------------------------------- Sender
def test_sender_duplicates_when_source_is_slow_and_drops_when_fast():
    clk = FakeClock()
    slot = LatestSlot()
    sink = RecordingSink()
    rt = PassThroughRetargeter()
    s = Sender(rt, sink, slot, rate_hz=30.0, clock=clk)
    s.tick()  # empty slot -> rest channels
    assert sink.sent[-1]["head_yaw"] == 0.0 and s.stats.dup == 0
    slot.put(Sample(0.0, {"head_yaw": 5.0}, True))
    for _ in range(4):
        s.tick()
    assert s.stats.dup == 3  # one new sample, three re-sends of it
    assert all(j["head_yaw"] == 5.0 for j in sink.sent[1:])
    for k in range(4):  # a fast source: 4 samples between two ticks
        slot.put(Sample(0.1 + k * 0.005, {"head_yaw": 10.0 + k}, True))
    s.tick()
    assert sink.sent[-1]["head_yaw"] == 13.0 and slot.dropped == 3
    assert all(dt == pytest.approx(1 / 30) for dt in rt.dts)
    assert s.stats.sent == 6 and s.stats.errors == 0


def test_sender_survives_sink_errors_and_counts_them():
    clk = FakeClock()
    slot = LatestSlot()
    sink = RecordingSink(fail_every=3)
    s = Sender(PassThroughRetargeter(), sink, slot, rate_hz=30.0, clock=clk)
    for _ in range(9):
        s.tick()
    assert s.stats.sent == 9 and s.stats.errors == 3
    assert len(s.stats.latencies_ms) == 9


def test_sender_thread_real_clock_rates():
    """0.6 s of real time: a 10 Hz producer yields ~2 dups per new sample; a 120 Hz producer drops most."""
    for prod_hz, expect in ((10.0, "dup"), (120.0, "drop")):
        slot = LatestSlot()
        sink = RecordingSink()
        s = Sender(PassThroughRetargeter(), sink, slot, rate_hz=30.0)
        stop = threading.Event()

        def produce():
            k = 0
            while not stop.is_set():
                slot.put(Sample(k / prod_hz, {"head_yaw": float(k)}, True))
                k += 1
                time.sleep(1.0 / prod_hz)

        th = threading.Thread(target=produce, daemon=True)
        s.start()
        th.start()
        time.sleep(0.6)
        s.join()
        stop.set()
        th.join(1.0)
        assert 12 <= s.stats.sent <= 24, s.stats.sent  # ~18 ticks at 30 Hz, tolerant of a loaded laptop
        if expect == "dup":
            assert s.stats.dup >= 0.4 * s.stats.sent and slot.dropped <= 2
        else:
            assert slot.dropped >= 1.5 * s.stats.sent and s.stats.dup <= 2
        assert s.stats.p50_ms() < 5.0


# ---------------------------------------------------------------- ChannelAssembler
def _sample(t, yaw=5.0, pitch=-3.0, roll=2.0, trans=(-400.0, 50.0, 60.0), face_ok=True, pose_ok=True, torso=(20.0, 1.0, -3.0)):
    s = {"t": t, "face_ok": face_ok, "pose_ok": pose_ok, "arm_ok": False}
    if face_ok:
        s["head_angles"] = (yaw, pitch, roll)
        s["head_trans"] = np.array(trans)
        s["face_raw"] = cm.face_raw_from_blendshapes({"jawOpen": 0.2, "browDownLeft": 0.5, "browDownRight": 0.5})
    if pose_ok:
        s["torso_vals"] = dict(zip(["torso_lean_fwd", "torso_lean_side", "torso_yaw"], torso))
    return s


def test_assembler_neutral_window_then_relative_values():
    asm = ChannelAssembler(neutral_seconds=0.5, hold_s=0.2)
    # neutral yaw 5 with pitch/roll 0 so the relative yaw is exactly additive
    outs = [asm.update(_sample(k / 30, pitch=0.0, roll=0.0)) for k in range(16)]  # 0..0.5 s
    assert outs[0] is None and outs[10] is None
    assert asm.calibrated and outs[-1] is not None
    ch = asm.update(_sample(0.6, pitch=0.0, roll=0.0))
    assert ch["head_yaw"] == pytest.approx(0.0, abs=1e-6) and ch["head_x"] == pytest.approx(0.0, abs=1e-6)
    assert ch["torso_lean_fwd"] == pytest.approx(0.0, abs=1e-6) and ch["brow_l"] == 0.0 and ch["face_valid"] == 1.0
    ch = asm.update(_sample(0.63, yaw=15.0, pitch=0.0, roll=0.0, trans=(-390.0, 50.0, 60.0), torso=(25.0, 1.0, -3.0)))
    assert ch["head_yaw"] == pytest.approx(10.0, abs=1e-6) and ch["head_x"] == pytest.approx(10.0)
    assert ch["torso_lean_fwd"] == pytest.approx(5.0)
    assert ch["mouth_open"] == pytest.approx(0.2)  # absolute channel, not zeroed


def test_assembler_holds_then_releases_after_dropout():
    asm = ChannelAssembler(neutral_seconds=0.0, hold_s=0.2,
                           neutral={"head_angles_deg": [0, 0, 0], "head_trans_mm": [0, 0, 0], "face_raw": {}, "torso_deg": [0, 0, 0]})
    ch = asm.update(_sample(0.0, yaw=12.0))
    assert ch["head_yaw"] == pytest.approx(12.0) and ch["face_valid"] == 1.0
    ch = asm.update(_sample(0.1, face_ok=False, pose_ok=False))
    assert ch["head_yaw"] == pytest.approx(12.0) and ch["face_valid"] == 0.0  # held, flagged invalid
    ch = asm.update(_sample(0.5, face_ok=False, pose_ok=False))
    assert math.isnan(ch["head_yaw"]) and math.isnan(ch["torso_yaw"]) and ch["face_valid"] == 0.0
    # pose skipped on purpose (every-2nd-frame mode): torso is held regardless of hold time
    asm.update(_sample(0.6, torso=(7.0, 0.0, 0.0)))
    held = asm.update({**_sample(1.9, pose_ok=False), "pose_skipped": True})
    assert held["torso_lean_fwd"] == pytest.approx(7.0)


def test_rest_channels_cover_schema():
    from animacy.schema import CHANNELS

    r = rest_channels()
    assert set(r) == set(CHANNELS) and r["eye_open_l"] == 0.6 and r["head_yaw"] == 0.0

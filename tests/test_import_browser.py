"""Browser Record-mode importer: synthetic motion.json + generated audio (webm/opus via ffmpeg if
available, else wav — see ``AUDIO_KIND`` in the test output)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import zipfile

import numpy as np
import pytest
import soundfile as sf

from animacy import import_browser as ib
from animacy.schema import AUDIO_SR, CHANNELS, FACE_CHANNELS, HumanClip, empty_frames

RATE = 30.0
N = 150  # 5 s


def make_motion(n=N, gap=(60, 75)) -> dict:
    f = empty_frames(n, RATE)
    t = f["t"].to_numpy()
    f["face_valid"] = 1.0
    f["head_yaw"] = 12 * np.sin(2 * np.pi * 0.5 * t) + np.random.default_rng(0).normal(0, 0.8, n)
    f["head_pitch"] = 5 * np.cos(2 * np.pi * 0.7 * t)
    f["mouth_open"] = np.clip(0.3 + 0.3 * np.sin(2 * np.pi * 3 * t), 0, 1)
    f["torso_yaw"] = 2.0
    f["speaking"] = 1.0  # browser value, must be replaced
    clip = HumanClip.from_frames(f)
    j = clip.to_web_json()
    a, b = gap
    for c in FACE_CHANNELS:  # a tracking gap: nulls with face_valid still 1 (importer must demote)
        for i in range(a, b):
            j["data"][c][i] = None
    return j


def make_audio(path_wav: str, seconds: float, sr: int = AUDIO_SR) -> np.ndarray:
    """Tone bursts at 1-2 s and 3-4 s, silence elsewhere."""
    t = np.arange(int(seconds * sr)) / sr
    x = np.zeros_like(t, dtype=np.float32)
    for a, b in ((1.0, 2.0), (3.0, 4.0)):
        m = (t >= a) & (t < b)
        x[m] = 0.3 * np.sin(2 * np.pi * 220 * t[m]) * (1 + 0.5 * np.sin(2 * np.pi * 5 * t[m]))
    sf.write(path_wav, x, sr)
    return x


def encode_webm(wav: str, webm: str) -> bool:
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    r = subprocess.run([exe, "-v", "error", "-y", "-i", wav, "-c:a", "libopus", "-b:a", "48k", webm],
                       capture_output=True, text=True, timeout=60)
    return r.returncode == 0 and os.path.exists(webm) and os.path.getsize(webm) > 100


def write_segment(d: str, role: str, seconds: float = 5.0, prefix: str = "", motion=None, extra_meta=None) -> str:
    os.makedirs(d, exist_ok=True)
    json.dump(motion or make_motion(), open(os.path.join(d, prefix + "motion.json"), "w"))
    wav = os.path.join(d, prefix + "audio.wav")
    make_audio(wav, seconds)
    webm = os.path.join(d, prefix + "audio.webm")
    kind = "webm" if encode_webm(wav, webm) else "wav"
    if kind == "webm":
        os.remove(wav)
    meta = {"source": "webcam-browser", "role": role, "arm": "right", "license": "CC-BY-4.0",
            "neutral": {"head_angles_deg": [1, 2, 3]}, "versions": {"viewer": "0.1", "mediapipe": "0.10.x"}}
    meta.update(extra_meta or {})
    json.dump(meta, open(os.path.join(d, prefix + "meta.json"), "w"))
    print("AUDIO_KIND", kind)
    return kind


def fake_vad(audio, sr, t_grid):
    """Marks the tone bursts: proves the importer's speaking column comes from the VAD callback."""
    if audio is None:
        return np.zeros(len(t_grid), bool), "fake-none"
    win = int(0.02 * sr)
    idx = np.clip((t_grid * sr).astype(int), 0, len(audio) - 1)
    rms = np.array([np.sqrt(np.mean(audio[max(i - win, 0):i + win] ** 2)) for i in idx])
    return rms > 0.05, "fake-energy"


# ---------------------------------------------------------------- parsing
def test_frames_from_web_json_nulls_channels_and_grid():
    j = make_motion()
    j["data"]["t"][7] = 0.3  # jittered browser timestamp
    del j["data"]["gaze_pitch"]
    j["data"]["bogus"] = [0.0] * N
    df, notes = ib.frames_from_web_json(j)
    assert list(df.columns) == CHANNELS and len(df) == N
    assert np.isnan(df.loc[65, "head_yaw"]) and df.loc[65, "face_valid"] == 1.0  # not yet demoted
    assert np.allclose(df["t"].to_numpy(), np.arange(N) / RATE)
    assert notes["t_max_dev_ms"] == pytest.approx((0.3 - 7 / RATE) * 1000, abs=1e-6)
    assert notes["unknown_channels"] == ["bogus"] and notes["missing_channels"] == ["gaze_pitch"]
    assert np.isnan(df["gaze_pitch"]).all()
    j["data"]["head_yaw"] = j["data"]["head_yaw"][:-1]
    with pytest.raises(ValueError):
        ib.frames_from_web_json(j)


# ---------------------------------------------------------------- audio alignment
def test_align_audio_trims_pads_and_offsets():
    sr = 1000
    audio = np.arange(1, 5001, dtype=np.float32)  # 5.0 s
    out, info = ib.align_audio(audio, n_frames=120, rate_hz=30.0, sr=sr)  # motion 4.0 s
    assert len(out) == 4000 and info["action"] == "trimmed" and info["audio_minus_motion_ms"] == pytest.approx(1000)
    out, info = ib.align_audio(audio[:3000], 120, 30.0, sr)
    assert len(out) == 4000 and info["action"] == "padded" and out[3500] == 0.0 and out[2999] == 3000
    out, info = ib.align_audio(audio, 120, 30.0, sr, offset_s=0.25)  # audio started 250 ms after frame 0
    assert (out[:250] == 0).all() and out[250] == 1 and info["audio_offset_s"] == 0.25
    out, _ = ib.align_audio(audio, 120, 30.0, sr, offset_s=-0.1)  # audio started before frame 0
    assert out[0] == 101


# ---------------------------------------------------------------- discovery
def test_find_segments_both_layouts(tmp_path):
    write_segment(str(tmp_path / "seg_a"), "speaking")
    write_segment(str(tmp_path / "flat"), "listening", prefix="take2_")
    segs = ib.find_segments(str(tmp_path))
    by = {s["name"]: s for s in segs}
    assert set(by) == {"seg_a", "take2"}
    for s in segs:
        assert s["audio"] and os.path.basename(s["audio"]).endswith(("audio.webm", "audio.wav"))
        assert s["meta"] and s["meta"].endswith("meta.json")


# ---------------------------------------------------------------- end to end
def test_import_speaking_segment(tmp_path):
    src = tmp_path / "rec"
    kind = write_segment(str(src), "speaking")
    out = tmp_path / "clip"
    rc, dirs = ib.import_path(str(src), str(out), vad_fn=fake_vad)
    assert rc == 0 and dirs == [str(out)]
    clip = HumanClip.load(str(out))
    assert clip.validate() == []
    assert len(clip) == N and clip.duration == pytest.approx((N - 1) / RATE)
    assert len(clip.audio) == int(N / RATE * AUDIO_SR)
    sp = clip.frames["speaking"].to_numpy()
    t = clip.frames["t"].to_numpy()
    assert sp[(t > 1.1) & (t < 1.9)].all() and sp[(t > 3.1) & (t < 3.9)].all()
    assert not sp[(t < 0.9)].any() and not sp[(t > 2.1) & (t < 2.9)].any()  # browser's all-1 speaking was replaced
    fv = clip.frames["face_valid"].to_numpy()
    assert not fv[60:75].any() and fv[:60].all() and fv[75:].all()  # NaN rows demoted
    assert clip.meta["fixes"]["face_valid_demoted"] == 15
    assert np.isnan(clip.frames.loc[65, "head_yaw"])
    assert clip.meta["role"] == "speaking" and clip.meta["license"] == "CC-BY-4.0" and clip.meta["source"] == "webcam-browser"
    assert clip.meta["neutral"] == {"head_angles_deg": [1, 2, 3]} and clip.meta["tool_versions"]["browser"]["viewer"] == "0.1"
    assert clip.meta["vad"] == "fake-energy" and clip.meta["audio_align"]["action"] == "exact"
    assert (kind == "webm") == ("ffmpeg" in clip.meta["audio_backend"])
    # smoothing happened (noise reduced) but the signal survived
    raw = np.array([v if v is not None else np.nan for v in make_motion()["data"]["head_yaw"]])
    sm = clip.frames["head_yaw"].to_numpy()
    ok = ~np.isnan(sm)
    assert np.abs(np.diff(sm[ok], n=2)).mean() < np.abs(np.diff(raw[ok], n=2)).mean() * 0.7
    assert np.corrcoef(sm[ok], raw[ok])[0, 1] > 0.98


def test_import_listening_forces_speaking_zero_and_real_vad_path(tmp_path):
    src = tmp_path / "rec"
    write_segment(str(src), "listening")
    out = tmp_path / "clip"
    rc, _ = ib.import_path(str(src), str(out))  # real VAD path (silero or energy) must not be consulted
    clip = HumanClip.load(str(out))
    assert rc == 0 and clip.validate() == []
    assert not clip.frames["speaking"].to_numpy().any()
    assert clip.meta["vad"].startswith("forced 0")


def test_import_zip_with_two_segments_and_audio_length_mismatch(tmp_path):
    rec = tmp_path / "rec"
    write_segment(str(rec / "talk_01"), "speaking", seconds=5.4)          # audio 400 ms long -> trimmed
    write_segment(str(rec / "listen_01"), "listening", seconds=4.7)       # audio 300 ms short -> padded
    zpath = tmp_path / "session.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        for dp, _, files in os.walk(rec):
            for f in files:
                full = os.path.join(dp, f)
                z.write(full, os.path.relpath(full, rec))
    out = tmp_path / "clips"
    rc, dirs = ib.import_path(str(zpath), str(out), vad_fn=fake_vad)
    assert rc == 0 and sorted(os.path.basename(d) for d in dirs) == ["listen_01", "talk_01"]
    talk = HumanClip.load(str(out / "talk_01"))
    listen = HumanClip.load(str(out / "listen_01"))
    assert talk.validate() == [] and listen.validate() == []
    assert talk.meta["audio_align"]["action"] == "trimmed" and talk.meta["audio_align"]["audio_minus_motion_ms"] == pytest.approx(400, abs=60)
    assert listen.meta["audio_align"]["action"] == "padded" and listen.meta["audio_align"]["audio_minus_motion_ms"] == pytest.approx(-300, abs=60)
    assert len(talk.audio) == len(listen.audio) == int(N / RATE * AUDIO_SR)
    assert talk.frames["speaking"].mean() > 0.3 and listen.frames["speaking"].sum() == 0


def test_audio_offset_from_meta_shifts_speech(tmp_path):
    src = tmp_path / "rec"
    write_segment(str(src), "speaking", extra_meta={"audio_offset_s": 0.5})
    out = tmp_path / "clip"
    ib.import_path(str(src), str(out), vad_fn=fake_vad)
    clip = HumanClip.load(str(out))
    t = clip.frames["t"].to_numpy()
    sp = clip.frames["speaking"].to_numpy()
    assert sp[(t > 1.6) & (t < 2.4)].all() and not sp[(t > 1.0) & (t < 1.4)].any()  # bursts moved +0.5 s
    assert clip.validate() == []


def test_no_smooth_keeps_raw_values(tmp_path):
    src = tmp_path / "rec"
    m = make_motion()
    write_segment(str(src), "listening", motion=m)
    out = tmp_path / "clip"
    ib.import_path(str(src), str(out), smooth=False)
    clip = HumanClip.load(str(out))
    raw = np.array([v for v in m["data"]["head_yaw"][:60]])
    assert np.allclose(clip.frames["head_yaw"].to_numpy()[:60], raw, atol=1e-3)

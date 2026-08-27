"""Core contracts: schema round-trip, ROBOT.md validation, retarget (offline + live), exporters."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from animacy.export import read_autonomous_os_csv, to_autonomous_os_csv, to_pollen_move, validate_autonomous_os_csv
from animacy.profile import Profile, load_profile, robots_root
from animacy.retarget import LiveRetargeter, retarget_clip, stretch_timeline, to_urdf_values
from animacy.schema import CHANNELS, HumanClip, MAPPABLE, empty_frames

ROBOTS = [d for d in sorted(os.listdir(robots_root())) if not d.startswith("_") and os.path.exists(os.path.join(robots_root(), d, "ROBOT.md"))]


def _synthetic_clip(n=120, rate=30.0):
    f = empty_frames(n, rate)
    f["face_valid"] = 1.0
    t = f["t"].to_numpy()
    f["head_yaw"] = 25 * np.sin(2 * np.pi * 0.5 * t)
    f["head_pitch"] = 10 * np.sin(2 * np.pi * 0.8 * t)
    f["brow_l"] = np.clip(np.sin(2 * np.pi * 0.3 * t), 0, 1)
    f["brow_r"] = f["brow_l"]
    return HumanClip.from_frames(f, source="synthetic")


def test_schema_roundtrip_and_validate():
    c = _synthetic_clip()
    assert c.validate() == []
    d = tempfile.mkdtemp()
    c.save(d)
    c2 = HumanClip.load(d)
    assert len(c2) == len(c) and list(c2.frames.columns) == CHANNELS
    assert c2.meta["source"] == "synthetic"
    j = c2.to_web_json()
    assert j["n"] == len(c) and set(j["data"]) == set(CHANNELS)


def test_validate_catches_nan_on_valid_frames():
    c = _synthetic_clip()
    c.frames.loc[5, "head_yaw"] = np.nan
    assert any("NaN" in p for p in c.validate())


@pytest.mark.parametrize("name", ROBOTS)
def test_profile_parses_and_mappings_reference_real_channels(name):
    p = load_profile(os.path.join(robots_root(), name))
    assert p.name == name
    assert "default" in p.retarget
    for mode, mp in p.retarget.items():
        for jn, m in mp.items():
            assert jn in p.joint_names, (mode, jn)
            for term in m.terms():
                assert term.from_ in MAPPABLE, (mode, jn, term.from_)
    errs = [e for e in p.check() if "urdf" not in e.lower() and "native_clips" not in e]
    assert errs == [], errs


@pytest.mark.parametrize("name", ROBOTS)
def test_retarget_offline_is_speed_legal_and_in_bounds(name):
    p = load_profile(os.path.join(robots_root(), name))
    c = _synthetic_clip()
    t = retarget_clip(c, p)
    assert list(t.columns) == ["t"] + p.joint_names
    dt = np.diff(t["t"].to_numpy())
    assert np.all(dt > 0)
    for j in p.joints:
        v = t[j.name].to_numpy()
        assert v.min() >= j.min - 1e-6 and v.max() <= j.max + 1e-6, j.name
        speed = np.abs(np.diff(v)) / dt
        assert speed.max() <= j.max_speed * (1 + 1e-6), (j.name, speed.max())


def test_stretch_only_widens_impossible_segments():
    spec = {"schema": "animacy.robot.v1", "name": "t", "display_name": "T", "description": {"urdf": "x.urdf"},
            "joints": [{"name": "a", "min": -90, "max": 90, "rest": 0, "max_speed": 100}],
            "retarget": {"default": {"a": {"from": "head_yaw"}}}}
    p = Profile(**spec)
    import pandas as pd
    tbl = pd.DataFrame({"t": [0, 0.1, 0.2, 0.3], "a": [0, 5, 50, 55]})
    out = stretch_timeline(tbl["t"].to_numpy(dtype=float), tbl, p, margin=1.0)
    assert np.isclose(out[1] - out[0], 0.1)          # 50 deg/s: untouched
    assert np.isclose(out[2] - out[1], 0.45)         # 450 deg/s → stretched to 45/100
    assert np.isclose(out[3] - out[2], 0.1)


def test_live_retargeter_clips_velocity_and_converges():
    spec = {"schema": "animacy.robot.v1", "name": "t", "display_name": "T", "description": {"urdf": "x.urdf"},
            "joints": [{"name": "a", "min": -90, "max": 90, "rest": 0, "max_speed": 60}],
            "retarget": {"default": {"a": {"from": "head_yaw", "gain": 1.0, "smooth_hz": 100}}}}
    rt = LiveRetargeter(Profile(**spec))
    prev = 0.0
    for _ in range(90):
        y = rt.step({"head_yaw": 40.0}, 1 / 30)["a"]
        assert y - prev <= 60 / 30 + 1e-9
        prev = y
    assert abs(prev - 40.0) < 0.5


def test_check_refuses_profile_range_wider_than_urdf(tmp_path):
    urdf = tmp_path / "r.urdf"
    urdf.write_text('<robot name="r"><link name="a"/><link name="b"/>'
                    '<joint name="j" type="revolute"><parent link="a"/><child link="b"/>'
                    '<axis xyz="0 0 1"/><limit lower="-0.5" upper="0.5" effort="1" velocity="1"/></joint></robot>')
    base = {"schema": "animacy.robot.v1", "name": "r", "display_name": "R", "description": {"urdf": "r.urdf"},
            "retarget": {"default": {}}}
    ok = Profile(**base, joints=[{"name": "j", "min": -28, "max": 28, "rest": 0, "max_speed": 100}])
    ok.path = str(tmp_path / "ROBOT.md")
    assert ok.check() == []
    bad = Profile(**base, joints=[{"name": "j", "min": -90, "max": 90, "rest": 0, "max_speed": 100}])
    bad.path = str(tmp_path / "ROBOT.md")
    assert any("exceeds the URDF" in e for e in bad.check())


def test_to_urdf_values_applies_sign_offset_units():
    spec = {"schema": "animacy.robot.v1", "name": "t", "display_name": "T", "description": {"urdf": "x.urdf"},
            "joints": [{"name": "p", "min": -90, "max": 90, "rest": 0, "max_speed": 100, "urdf_sign": -1, "urdf_offset": 10},
                       {"name": "z", "unit": "mm", "min": -50, "max": 50, "rest": 0, "max_speed": 100}],
            "retarget": {"default": {}}}
    import pandas as pd
    p = Profile(**spec)
    v = to_urdf_values(pd.DataFrame({"t": [0], "p": [20.0], "z": [25.0]}), p)
    assert np.isclose(v["p"][0], np.deg2rad(-30.0)) and np.isclose(v["z"][0], 0.025)


def test_autonomous_csv_roundtrip_and_validator():
    p = load_profile(os.path.join(robots_root(), "lamp"))
    t = retarget_clip(_synthetic_clip(), p)
    text = to_autonomous_os_csv(t, p)
    assert validate_autonomous_os_csv(text, p.joint_names, {j.name: j.max_speed for j in p.joints}) == []
    assert text.splitlines()[0] == "timestamp," + ",".join(f"{j}.pos" for j in p.joint_names)
    bad = text.replace("base_yaw.pos", "base_yaw")
    assert validate_autonomous_os_csv(bad, p.joint_names)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "x.csv")
    open(path, "w").write(text)
    back = read_autonomous_os_csv(path)
    assert list(back.columns) == ["t"] + p.joint_names and len(back) == len(t)


def test_vendor_lamp_clips_pass_the_upload_validator():
    p = load_profile(os.path.join(robots_root(), "lamp"))
    d = os.path.join(p.dir, "clips", "native")
    files = [f for f in os.listdir(d) if f.endswith(".csv")]
    assert len(files) >= 30
    for f in files:
        text = open(os.path.join(d, f), encoding="utf-8").read()
        assert validate_autonomous_os_csv(text, p.joint_names) == [], f


def test_pollen_move_export_shape():
    p = load_profile(os.path.join(robots_root(), "reachy_mini"))
    t = retarget_clip(_synthetic_clip(), p)
    mv = to_pollen_move(t, p)
    assert len(mv["time"]) == len(t) == len(mv["set_target_data"])
    fr = mv["set_target_data"][10]
    assert np.asarray(fr["head"]).shape == (4, 4) and len(fr["antennas"]) == 2

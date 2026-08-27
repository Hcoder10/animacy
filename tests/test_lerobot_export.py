"""LeRobot v3.0 export: writer contract on synthetic clips (main venv), plus a real
``LeRobotDataset`` load in the separate lerobot venv when it exists."""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from animacy.lerobot_export import (
    ACTION,
    AUDIO_NAMES,
    CODEBASE_VERSION,
    DEFAULT_FEATURES,
    OBS_AUDIO,
    OBS_ENV_STATE,
    OBS_HUMAN,
    OBS_SPEAKING,
    OBS_STATE,
    build_features,
    clean_title,
    default_lerobot_python,
    episodes_from_clip,
    export,
    task_string,
    validate_with_lerobot,
)
from animacy.model.data import MODEL_CHANNELS, make_synthetic_clips
from animacy.profile import load_profile, robots_root
from animacy.schema import HumanClip, empty_frames

ROBOTS = [d for d in sorted(os.listdir(robots_root())) if not d.startswith("_") and os.path.exists(os.path.join(robots_root(), d, "ROBOT.md"))]
FPS = 30


@pytest.fixture(scope="module")
def synth_clips():
    d = tempfile.mkdtemp(prefix="animacy_lr_clips_")
    # 4 clips x 12 s: even ones have a face gap (-> several runs), #3 is a listener
    make_synthetic_clips(d, n_clips=4, seconds=12.0, seed=1, n_subjects=2)
    return d


@pytest.fixture(scope="module")
def lamp_export(synth_clips):
    out = os.path.join(tempfile.mkdtemp(prefix="animacy_lr_out_"), "animacy_lamp")
    summary = export("lamp", synth_clips, out, fps=FPS, max_seconds=5.0, verbose=False)
    return out, summary


def _read_data(out):
    files = sorted(os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(out, "data")) for f in fs if f.endswith(".parquet"))
    assert files, "no data parquet written"
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True), files


def test_clean_title_and_task_string():
    assert clean_title("2014-09-13 President Obama's Weekly Address.webm") == "president obama's weekly address"
    assert clean_title("Video Blog- CBP Super Bowl LII Countdown to Kickoff - Day 2.webm") == "video blog cbp super bowl lii countdown to kickoff day 2"
    assert len(clean_title("x " * 100)) <= 64
    assert task_string({"title": "Some Talk.mp4"}, "clip", 0.9) == "speaking: some talk"
    assert task_string({"title": "Some Talk.mp4"}, "clip", 0.1) == "listening: some talk"
    assert task_string({"role": "Listening", "prompt": "nod along"}, "clip", 0.9) == "listening: nod along"
    assert task_string({}, "my_clip_01", 0.9) == "speaking: my clip 01"


def test_tree_and_info(lamp_export):
    out, summary = lamp_export
    for rel in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet",
                "meta/animacy.json", "data/chunk-000/file-000.parquet"):
        assert os.path.exists(os.path.join(out, rel)), rel
    info = json.load(open(os.path.join(out, "meta/info.json"), encoding="utf-8"))
    assert info["codebase_version"] == CODEBASE_VERSION
    assert info["fps"] == FPS and info["robot_type"] == "animacy/lamp" and info["video_path"] is None
    assert info["splits"] == {"train": f"0:{info['total_episodes']}"}
    assert list(info["features"])[-5:] == list(DEFAULT_FEATURES)
    prof = load_profile(os.path.join(robots_root(), "lamp"))
    assert info["features"][OBS_STATE]["names"] == prof.joint_names
    assert info["features"][ACTION]["shape"] == [len(prof.joints)]
    assert info["features"][OBS_HUMAN]["names"] == MODEL_CHANNELS
    assert info["features"][OBS_AUDIO]["names"] == AUDIO_NAMES and info["features"][OBS_AUDIO]["shape"] == [66]
    assert info["features"][OBS_ENV_STATE]["shape"] == [67]
    assert summary["total_episodes"] == info["total_episodes"] and summary["total_frames"] == info["total_frames"]


def test_data_rows_match_contract(lamp_export):
    out, _ = lamp_export
    info = json.load(open(os.path.join(out, "meta/info.json"), encoding="utf-8"))
    df, _ = _read_data(out)
    assert len(df) == info["total_frames"]
    assert df["index"].tolist() == list(range(len(df)))
    eps = pd.read_parquet(os.path.join(out, "meta/episodes/chunk-000/file-000.parquet"))
    assert len(eps) == info["total_episodes"]
    prof = load_profile(os.path.join(robots_root(), "lamp"))
    for _, ep in eps.iterrows():
        a, b = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        assert b - a == int(ep["length"]) and 3 * FPS <= b - a <= 5 * FPS
        seg = df.iloc[a:b]
        assert seg["episode_index"].nunique() == 1 and int(seg["episode_index"].iloc[0]) == int(ep["episode_index"])
        assert seg["frame_index"].tolist() == list(range(b - a))
        np.testing.assert_allclose(seg["timestamp"].to_numpy(), np.arange(b - a) / FPS, atol=1e-5)
        st = np.stack(seg[OBS_STATE].to_numpy())
        ac = np.stack(seg[ACTION].to_numpy())
        assert st.dtype == np.float32 and st.shape == (b - a, len(prof.joints))
        np.testing.assert_array_equal(ac[:-1], st[1:])          # action = next state
        for i, j in enumerate(prof.joints):
            assert st[:, i].min() >= j.min - 1e-4 and st[:, i].max() <= j.max + 1e-4, j.name
        env = np.stack(seg[OBS_ENV_STATE].to_numpy())
        au = np.stack(seg[OBS_AUDIO].to_numpy())
        np.testing.assert_array_equal(env[:, :66], au)
        np.testing.assert_array_equal(env[:, 66], seg[OBS_SPEAKING].to_numpy())
        assert set(np.unique(seg[OBS_SPEAKING].to_numpy())) <= {0.0, 1.0}
        assert np.isfinite(np.stack(seg[OBS_HUMAN].to_numpy())).all()
        assert ep["tasks"][0].split(":")[0] in ("speaking", "listening")
    tasks = pd.read_parquet(os.path.join(out, "meta/tasks.parquet"))
    tasks.index.name = "task"
    assert tasks["task_index"].tolist() == list(range(len(tasks))) and len(tasks) == info["total_tasks"]
    assert set(df["task_index"].unique()) <= set(tasks["task_index"])
    for t in eps["tasks"]:
        assert t[0] in tasks.index


def test_stats_shapes(lamp_export):
    out, _ = lamp_export
    info = json.load(open(os.path.join(out, "meta/info.json"), encoding="utf-8"))
    stats = json.load(open(os.path.join(out, "meta/stats.json"), encoding="utf-8"))
    assert set(stats) == set(info["features"])
    df, _ = _read_data(out)
    for k, ft in info["features"].items():
        d = int(np.prod(ft["shape"]))
        for key in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
            assert len(stats[k][key]) == d, (k, key)
        assert stats[k]["count"] == [info["total_frames"]]
        assert all(lo <= hi for lo, hi in zip(stats[k]["min"], stats[k]["max"]))
    st = np.stack(df[OBS_STATE].to_numpy()).astype(np.float64)
    np.testing.assert_allclose(stats[OBS_STATE]["mean"], st.mean(axis=0), atol=1e-5)
    np.testing.assert_allclose(stats[OBS_STATE]["std"], st.std(axis=0), atol=1e-5)


def test_face_gap_splits_runs_and_short_runs_are_dropped():
    prof = load_profile(os.path.join(robots_root(), "lamp"))
    f = empty_frames(15 * FPS)
    t = f["t"].to_numpy()
    f["head_yaw"] = 20 * np.sin(2 * np.pi * 0.4 * t)
    f["face_valid"] = 1.0
    f.loc[6 * FPS:6 * FPS + 10, "face_valid"] = 0.0        # 11-frame gap -> two runs of ~6 s and ~9 s
    f.loc[14 * FPS:, "face_valid"] = 0.0                    # trailing invalid tail
    f.loc[0:2 * FPS - 1, "face_valid"] = 0.0                # 2 s head: too short on its own
    f.loc[2 * FPS, "face_valid"] = 0.0
    clip = HumanClip.from_frames(f, source="synthetic", title="Gap Test")
    eps = episodes_from_clip(clip, "gap", prof, fps=FPS, min_seconds=3.0, max_seconds=20.0)
    runs = sorted({e.run for e in eps})
    assert len(runs) == 2, runs
    assert all(len(e) >= 3 * FPS for e in eps)
    assert all(e.task == "listening: gap test" for e in eps)    # no speaking flag anywhere
    # no audio -> zero features
    assert all(not e.audio.any() for e in eps)
    # source frame indices stay inside their run and are monotone
    for e in eps:
        assert e.src_frames.min() >= e.run[0] and e.src_frames.max() < e.run[1]
        assert np.all(np.diff(e.src_frames) >= 0)


def test_long_run_is_split_evenly():
    prof = load_profile(os.path.join(robots_root(), "lamp"))
    f = empty_frames(int(45.5 * FPS))
    f["face_valid"] = 1.0
    f["speaking"] = 1.0
    clip = HumanClip.from_frames(f, source="synthetic")
    eps = episodes_from_clip(clip, "long", prof, fps=FPS, min_seconds=3.0, max_seconds=20.0)
    assert len(eps) == 3
    assert all(FPS * 15 <= len(e) <= FPS * 20 for e in eps)
    assert sum(len(e) for e in eps) == int(45.5 * FPS)
    # the boundary frame's action is the true next state, only the run's last frame holds
    np.testing.assert_array_equal(eps[0].action[-1], eps[1].state[0])
    np.testing.assert_array_equal(eps[-1].action[-1], eps[-1].state[-1])


def test_max_stretch_drops_impossible_runs():
    prof = load_profile(os.path.join(robots_root(), "lamp"))
    f = empty_frames(6 * FPS)
    f["face_valid"] = 1.0
    f["head_yaw"] = np.where(np.arange(6 * FPS) % 2 == 0, -80.0, 80.0)   # 160 deg flips every frame: way over 250 deg/s
    clip = HumanClip.from_frames(f, source="synthetic")
    dropped: list = []
    assert episodes_from_clip(clip, "wild", prof, fps=FPS, max_stretch=1.1, dropped=dropped) == []
    assert dropped and "time-stretched" in dropped[0]
    kept = episodes_from_clip(clip, "wild", prof, fps=FPS, max_stretch=None)
    assert kept and kept[0].stretch > 1.1


@pytest.mark.parametrize("name", ROBOTS)
def test_every_robot_exports(name, synth_clips):
    out = os.path.join(tempfile.mkdtemp(prefix="animacy_lr_robot_"), f"animacy_{name}")
    s = export(name, synth_clips, out, fps=FPS, verbose=False, env_state="none")
    prof = load_profile(os.path.join(robots_root(), name))
    assert s["total_episodes"] > 0 and s["robot_type"] == f"animacy/{name}"
    assert OBS_ENV_STATE not in s["features"]
    assert s["features"][OBS_STATE]["names"] == prof.joint_names


def test_refuses_to_clobber_non_dataset_dir(synth_clips):
    d = tempfile.mkdtemp(prefix="animacy_lr_clobber_")
    open(os.path.join(d, "keep.txt"), "w").close()
    with pytest.raises(FileExistsError):
        export("lamp", synth_clips, d, verbose=False)
    with pytest.raises(FileExistsError):
        export("lamp", synth_clips, d, verbose=False, force=True)   # not a lerobot tree -> still refused
    assert os.path.exists(os.path.join(d, "keep.txt"))


def test_build_features_env_state_choices():
    prof = load_profile(os.path.join(robots_root(), "lamp"))
    assert build_features(prof, "human")[OBS_ENV_STATE]["shape"] == (len(MODEL_CHANNELS) + 1,)
    with pytest.raises(ValueError):
        build_features(prof, "video")


@pytest.mark.skipif(default_lerobot_python() is None, reason="no lerobot venv (.venv-lerobot / ANIMACY_LEROBOT_PYTHON)")
def test_loads_with_real_lerobot(lamp_export):
    out, summary = lamp_export
    ok, log, vsum = validate_with_lerobot(out, repo_id="animacy/test_lamp")
    assert ok, log[-3000:]
    assert vsum["codebase_version"] == CODEBASE_VERSION
    assert vsum["total_frames"] == summary["total_frames"] and vsum["total_episodes"] == summary["total_episodes"]
    assert vsum["policy_features"][OBS_STATE] == "STATE" and vsum["policy_features"][ACTION] == "ACTION"
    assert vsum["policy_features"][OBS_ENV_STATE] == "ENV"
    assert vsum["batch_ok"]

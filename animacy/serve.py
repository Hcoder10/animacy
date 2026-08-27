"""The interaction runtime: the robot talks and moves in sync, from one waveform.

    animacy say "Hey! Nice to meet you." --robot reachy_mini [--source model|retrieval|envelope]

Pipeline: text → TTS waveform (``animacy.tts``) → audio features
(``animacy.features``) → a *motion source* that returns a canonical
:class:`HumanClip` on the same 30 Hz grid → ``retarget_clip`` through the
robot's ``ROBOT.md`` → streamed to the robot sink while the audio plays on the
local speaker. Motion and speech are derived from the same samples, so sync is
structural: nothing is timed by hand.

Motion sources:
  model      the trained audio→motion model (``animacy.model.infer``)
  retrieval  motion matching against the corpus (``animacy.model.retrieval``)
  envelope   an explicit, labelled heuristic (speech energy → small nods, brow
             raises on stressed syllables, a lean-in at onsets). It exists to
             test the plumbing and as the floor the learned sources must beat;
             it is never reported as learned behaviour.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Optional

import numpy as np
import pandas as pd

from .features import RATE_HZ, audio_features
from .profile import Profile, find_robot
from .retarget import retarget_clip
from .schema import HumanClip, empty_frames
from .sinks import make_sink, stream_table


# ---------------------------------------------------------------- motion sources
def envelope_motion(wav: np.ndarray, sr: int = 16000, seed: int = 0) -> HumanClip:
    """Speech-envelope heuristic (see module docstring). Deterministic per seed."""
    rng = np.random.default_rng(seed)
    n = int(np.ceil(len(wav) / sr * RATE_HZ))
    feats = audio_features(wav, sr, n_ticks=n)
    energy = feats[:, 64]                       # normalised log energy
    denergy = feats[:, 65]
    speaking = (energy > -0.3).astype(np.float32)
    # smooth the envelope; onsets = positive energy jumps
    k = np.hanning(9) / np.hanning(9).sum()
    env = np.convolve(energy, k, mode="same")
    onset = np.clip(np.convolve(denergy, k, mode="same"), 0, None)
    f = empty_frames(n)
    f["face_valid"] = 1.0
    f["speaking"] = speaking
    t = np.arange(n) / RATE_HZ
    # slow drift of gaze (a talker never holds perfectly still)
    f["head_yaw"] = 6 * np.sin(2 * np.pi * 0.11 * t + rng.uniform(0, 6)) + 3 * np.sin(2 * np.pi * 0.23 * t + rng.uniform(0, 6))
    f["head_roll"] = 3 * np.sin(2 * np.pi * 0.07 * t + rng.uniform(0, 6))
    # nods ride the envelope: louder → head dips a little; onsets → brief downbeat
    f["head_pitch"] = -4 * np.clip(env, -1, 2) - 6 * np.convolve(onset, np.hanning(7) / 3.5, mode="same")
    # brows lift on stressed (loud) stretches
    brow = np.clip(0.35 * (env - 0.2), 0, 1)
    f["brow_l"] = brow
    f["brow_r"] = brow
    # lean in at utterance onsets, settle back
    lean = np.zeros(n)
    on = np.where((speaking[1:] > speaking[:-1]))[0]
    for i in on:
        lean[i:i + 45] += 40 * np.exp(-np.arange(min(45, n - i)) / 15.0)
    f["head_x"] = lean
    f["torso_lean_fwd"] = lean / 8.0
    f["mouth_open"] = np.clip(0.5 * (env + 0.5), 0, 1) * speaking
    return HumanClip.from_frames(f, source="envelope-heuristic", rate_hz=RATE_HZ)


def _speaking_from_audio(wav: np.ndarray, n: int, sr: int = 16000) -> np.ndarray:
    """Talk mode: the robot is the speaker wherever its own TTS has energy."""
    feats = audio_features(wav, sr, n_ticks=n)
    return (feats[:, 64] > -0.3).astype(np.int64)


_MODEL_CACHE: dict = {}


def model_motion(wav: np.ndarray, sr: int = 16000, checkpoint: str = "checkpoints/v1", seed: int = 0,
                 listen: bool = False, temperature: float = 0.8, bigram_weight: float = 0.5) -> HumanClip:
    """The learned audio→motion model (``animacy.model.infer``)."""
    import os

    from .model.infer import MotionModel, generate

    key = os.path.abspath(checkpoint)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MotionModel.load(checkpoint)
    n = int(np.ceil(len(wav) / sr * RATE_HZ))
    feats = audio_features(wav, sr, n_ticks=n)
    speaking = np.zeros(n, np.int64) if listen else _speaking_from_audio(wav, n, sr)
    return generate(_MODEL_CACHE[key], feats, speaking, causal=listen, temperature=temperature,
                    bigram_weight=bigram_weight, seed=seed)


def retrieval_motion(wav: np.ndarray, sr: int = 16000, checkpoint: str = "checkpoints/v1", seed: int = 0,
                     listen: bool = False) -> HumanClip:
    """Motion matching against the corpus (``animacy.model.retrieval``)."""
    import os

    from .model.infer import motion_to_clip
    from .model.retrieval import RetrievalIndex

    key = ("retrieval", os.path.abspath(checkpoint))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = RetrievalIndex.load(os.path.join(checkpoint, "retrieval.json"))
    n = int(np.ceil(len(wav) / sr * RATE_HZ))
    feats = audio_features(wav, sr, n_ticks=n)
    speaking = np.zeros(n, np.int64) if listen else _speaking_from_audio(wav, n, sr)
    motion = _MODEL_CACHE[key].query(feats, speaking)
    return motion_to_clip(motion, speaking, RATE_HZ, source="retrieval", mode="listen" if listen else "talk")


SOURCES = {"envelope": envelope_motion, "model": model_motion, "retrieval": retrieval_motion}


# ---------------------------------------------------------------- the loop
def say(text: str, profile: Profile, source: str = "envelope", sink_kind: Optional[str] = None,
        url: Optional[str] = None, tts_engine: str = "auto", play_audio: bool = True,
        dry_run: bool = False, seed: int = 0, checkpoint: str = "checkpoints/v1") -> pd.DataFrame:
    from .tts import synth

    t0 = time.perf_counter()
    wav, sr = synth(text, engine=tts_engine)
    t_tts = time.perf_counter() - t0
    fn = SOURCES[source]
    t1 = time.perf_counter()
    clip = fn(wav, sr, seed=seed) if source == "envelope" else fn(wav, sr, checkpoint=checkpoint, seed=seed)
    t_motion = time.perf_counter() - t1
    probs = clip.validate()
    if probs:
        raise RuntimeError(f"motion source {source!r} produced an invalid clip: {probs}")
    table = retarget_clip(clip, profile)
    print(f"[say] {len(wav) / sr:.2f}s speech | tts {t_tts:.2f}s | motion({source}) {t_motion:.2f}s | "
          f"{len(table)} frames on {profile.name}", flush=True)
    if dry_run:
        return table
    sink = make_sink(profile, sink_kind, url)
    sink.prepare()
    player = None
    if play_audio:
        try:
            from .tts import play_async

            player = play_async(wav)
        except Exception as e:  # noqa: BLE001
            print("[say] audio playback unavailable:", e)
    stream_table(table, profile, sink)
    if player is not None:
        player.wait()
    sink.neutral(1.0)
    return table


def main(args) -> int:
    profile = find_robot(args.robot)
    if args.source not in SOURCES:
        print(f"unknown source {args.source!r}; choose from {list(SOURCES)}")
        return 2
    say(args.text, profile, source=args.source, sink_kind=args.sink, url=args.url, tts_engine=args.tts,
        play_audio=not args.no_audio, dry_run=args.dry_run, seed=args.seed, checkpoint=args.checkpoint)
    return 0


if __name__ == "__main__":  # pragma: no cover
    from .cli import main as cli_main

    sys.exit(cli_main(["say", *sys.argv[1:]]))

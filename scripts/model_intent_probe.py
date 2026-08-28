"""Which gesture prototype fires for the tuning lines: TTS each line, run the retrieval
source exactly as ``animacy say`` does (intent from the text, prototype bonus, amplitude
tier, energy floor, settle, pitch floor) and print tag / tier / mean prototype score of the
chosen windows / energy before and after the floor.

    python scripts/model_intent_probe.py --ckpt checkpoints/v2a [--proto-weight 0.25] [--lines "..." "..."]

Default lines = the five grader tuning lines read from ``animacy.grade.movements`` (never
stored here). Sealed held-out lines are, by construction, not available to this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from animacy.features import RATE_HZ, audio_features  # noqa: E402
from animacy.model.infer import MotionModel, motion_energy, retrieve  # noqa: E402
from animacy.model.intent import analyse  # noqa: E402
from animacy.model.retrieval import RetrievalIndex  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v2a")
    ap.add_argument("--proto-weight", type=float, default=None, help="None = bundle default")
    ap.add_argument("--lines", nargs="*", default=None)
    ap.add_argument("--tts", default="auto")
    a = ap.parse_args()
    lines = a.lines
    if not lines:
        from animacy.grade.movements import MOVEMENTS

        lines = [m.text for m in MOVEMENTS]
    from animacy.tts import synth

    model = MotionModel.load(a.ckpt, "cpu")
    index = RetrievalIndex.load(os.path.join(a.ckpt, "retrieval.json"), motion_fp16=True)
    pp = model.info.get("postprocess", {})
    print(f"index {len(index)} windows, proto fields {'yes' if index.proto is not None else 'NO'}; bundle postprocess {pp}")
    rows = []
    for text in lines:
        wav, sr = synth(text, engine=a.tts)
        n = int(np.ceil(len(wav) / sr * RATE_HZ))
        feats = audio_features(wav, sr, n_ticks=n)
        speaking = (feats[:, 64] > -0.3).astype(np.int64)
        it = analyse(text)
        raw = retrieve(index, feats, speaking, model, intent=it, proto_weight=0.0, energy_floor=0, settle_s=0, pitch_floor=None, amplitude=1.0)
        clip = retrieve(index, feats, speaking, model, intent=it, proto_weight=a.proto_weight)
        e_raw = motion_energy(raw.frames[model.vq.channels].to_numpy(), model.vq.stats)
        e_out = clip.meta["energy"]
        # prototype scores of the chosen windows for every tag (which gesture the selection resembles)
        _, ids = index.query(feats, speaking, target_arousal=it.arousal, intent_tag=it.tag,
                             proto_weight=clip.meta["proto_weight"], return_ids=True)
        per_tag = {t: round(float(np.mean(index.proto[t][ids])), 2) for t in index.proto} if index.proto is not None else {}
        best = max(per_tag, key=per_tag.get) if per_tag else None
        gp = clip.meta.get("gesture_placement") or {}
        row = {"text": text, "tag": it.tag, "arousal": round(it.arousal, 2), "amplitude_tier": it.amplitude,
               "proto_weight": clip.meta["proto_weight"], "proto_mean_for_tag": clip.meta.get("proto_mean"),
               "chosen_windows_proto_by_tag": per_tag, "dominant_prototype_of_chosen": best,
               "energy_raw": round(e_raw, 3), "energy_out": round(e_out, 3), "energy_floor": clip.meta.get("energy_floor"),
               "seconds": round(n / RATE_HZ, 2),
               "accents": gp.get("accents"), "placements": gp.get("placements"), "gesture_amplitude": gp.get("gesture_amplitude")}
        rows.append(row)
        acc = gp.get("accents") or {}
        pl = [(p["kind"], p["gesture_id"], round(p["peak_at"] / RATE_HZ, 2)) for p in gp.get("placements", [])]
        print(f"{it.tag:11s} tier {it.amplitude:.2f} proto[{it.tag}] of chosen {row['proto_mean_for_tag']}: chosen resemble {per_tag} -> {best}; "
              f"energy {e_raw:.3f} -> {e_out:.3f} (floor {row['energy_floor']})  {text!r}")
        print(f"            accents at {[round(x / RATE_HZ, 2) for x in acc.get('accents', [])]} s (speech {acc.get('speech_s')} s, onset "
              f"{round(acc.get('onset', 0) / RATE_HZ, 2)} s) -> placed {pl} (kind, gesture window id, peak time s); gesture amplitude {gp.get('gesture_amplitude')}")
    out = os.path.join(a.ckpt, "intent_probe.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1, default=float)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

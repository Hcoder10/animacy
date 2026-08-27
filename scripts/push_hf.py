"""Push captured clips to the Hugging Face Hub as an animacy dataset.

    python scripts/push_hf.py --repo squaredcuber/animacy-human-motion [--clips data/clips] [--private] [--exclude a,b]

Layout on the Hub:
    clips/<name>/motion.parquet | audio.wav | meta.json
    index.json      one row per clip: name, seconds, rate_hz, source, license, source_url, subject/role, validity fractions
    README.md       dataset card generated from the clips' meta (sources + licenses listed verbatim)

Only clips whose meta carries a license (or source == webcam/self) are pushed;
anything else is refused, mirroring scripts/fetch_sources.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from animacy.schema import HumanClip  # noqa: E402


def clip_row(path: str) -> dict:
    c = HumanClip.load(path, audio=False)
    f = c.frames
    m = c.meta
    return {
        "name": os.path.basename(path),
        "seconds": round(c.duration, 1),
        "rate_hz": c.rate_hz,
        "frames": len(f),
        "source": m.get("source"),
        "source_url": m.get("source_url") or m.get("page_url"),
        "title": m.get("title"),
        "license": m.get("license"),
        "license_evidence": m.get("license_evidence"),
        "subject": m.get("subject"),
        "role": m.get("role"),
        "face_valid": round(float(f["face_valid"].mean()), 3),
        "arm_valid": round(float(f["arm_valid"].mean()), 3),
        "speaking": round(float(f["speaking"].fillna(0).mean()), 3),
    }


def card(rows: list[dict], repo: str) -> str:
    total = sum(r["seconds"] for r in rows) / 60
    lines = [
        "---",
        "license: cc-by-4.0",
        "task_categories: [other]",
        "tags: [robotics, motion, expressive-robots, animacy, reachy-mini, autonomous-lamp, vla]",
        "pretty_name: animacy human motion",
        "---",
        f"# {repo}",
        "",
        "Canonical human conversational motion (`animacy.human.v1`, 30 Hz) captured with",
        "[animacy](https://github.com/Hcoder10/animacy) — the open interaction layer for",
        "expressive robots. Each clip is `motion.parquet` (28 channels: head 6-DoF, gaze,",
        "brows, eyes, mouth, torso, a puppet arm chain, speaking/validity flags),",
        "`audio.wav` (16 kHz mono, same clock) and `meta.json`.",
        "",
        "Retarget any clip to a robot with `animacy retarget --robot <name>`; robots are",
        "one `ROBOT.md` each (Autonomous Lamp and Reachy Mini included).",
        "",
        f"**{len(rows)} clips, {total:.1f} minutes.**",
        "",
        "| clip | seconds | source | license | face valid | speaking |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        src = f"[{r['title'] or r['source']}]({r['source_url']})" if r.get("source_url") else (r["source"] or "")
        lines.append(f"| {r['name']} | {r['seconds']} | {src} | {r['license']} | {r['face_valid']} | {r['speaking']} |")
    lines += [
        "",
        "## Provenance",
        "",
        "Video sources are public-domain or CC-BY works; the license evidence field",
        "recorded at fetch time is in each `meta.json` (`license_evidence`). Webcam",
        "clips are the maintainer's own recordings, released CC-BY-4.0. No",
        "no-derivatives or non-commercial material is included by construction",
        "(`scripts/fetch_sources.py` refuses it).",
        "",
        "## Schema",
        "",
        "See `docs/CANONICAL.md` in the repo for every channel, unit and sign.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--clips", default=os.path.join(ROOT, "data", "clips"))
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    excl = set(x for x in a.exclude.split(",") if x)
    rows = []
    for n in sorted(os.listdir(a.clips)):
        p = os.path.join(a.clips, n)
        if n in excl or not os.path.exists(os.path.join(p, "motion.parquet")):
            continue
        r = clip_row(p)
        if not r["license"] and r["source"] not in ("webcam", "self"):
            print("refusing (no license metadata):", n)
            continue
        if not r["license"] and r["source"] in ("webcam", "self"):
            r["license"] = "CC-BY-4.0"
        rows.append(r)
    print(f"{len(rows)} clips, {sum(r['seconds'] for r in rows) / 60:.1f} min")
    os.makedirs(os.path.join(ROOT, "data", "hf_stage"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "hf_stage", "index.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    with open(os.path.join(ROOT, "data", "hf_stage", "README.md"), "w", encoding="utf-8") as fh:
        fh.write(card(rows, a.repo))
    if a.dry_run:
        print(card(rows, a.repo))
        return 0
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "data", "hf_stage", "README.md"), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=os.path.join(ROOT, "data", "hf_stage", "index.json"), path_in_repo="index.json", repo_id=a.repo, repo_type="dataset")
    for r in rows:
        api.upload_folder(folder_path=os.path.join(a.clips, r["name"]), path_in_repo=f"clips/{r['name']}", repo_id=a.repo,
                          repo_type="dataset", allow_patterns=["motion.parquet", "audio.wav", "meta.json"])
        print("pushed", r["name"])
    print(f"https://huggingface.co/datasets/{a.repo}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

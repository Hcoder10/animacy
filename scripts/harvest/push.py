"""Sharded pushes of kept clips to the Hub, then delete them locally (the volume has ~18 GB free).

    python scripts/harvest/push.py --repo squaredcuber/animacy-human-motion-large [--min-hours 50] [--loop] [--force]

A shard is one commit: ``clips/sNNNN/<name>/{motion.parquet,audio.opus,meta.json}`` for every
unpushed kept clip (oldest first, <= --max-clips per commit) + ``manifests/sNNNN.json``. After the
commit the clips are marked pushed and their local directories removed (``--keep-local`` to keep).
Then ``index.json`` (rebuilt from the clips table, i.e. everything ever pushed) and ``README.md``
(dataset card: license policy, per-source / per-language hours) are uploaded.

Triggers in --loop mode: >= --min-hours of unpushed clip time, or unpushed clips occupying more than
--max-local-gb on disk (push early to free space), or --force.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import time
from typing import Dict, List

import common as C
import index as IDX

CARD_HEAD = """---
license: other
license_name: mixed-cc-by-and-public-domain
license_link: https://creativecommons.org/licenses/by/4.0/
task_categories: [other]
tags: [robotics, motion, expressive-robots, animacy, reachy-mini, autonomous-lamp, vla, talking-head, speech]
pretty_name: animacy human motion (large harvest)
size_categories: [{size_cat}]
---
# {repo}

Canonical human conversational motion (`animacy.human.v1`, 30 Hz) captured at scale with
[animacy](https://github.com/Hcoder10/animacy) — the open interaction layer for expressive robots.
Every clip is a directory `clips/<shard>/<name>/` holding

* `motion.parquet` — 28 channels: head 6-DoF, gaze, brows, eyes, mouth, torso, a puppet arm chain,
  `speaking` / validity flags (schema: `docs/CANONICAL.md` in the repo);
* `audio.opus` — the clip's audio, mono, Opus {kbps} kb/s from the 16 kHz capture track, on the same
  clock as `motion.t` (`ffmpeg -i audio.opus -ar 16000 -ac 1 audio.wav` restores the 16 kHz wav that
  `animacy.schema.HumanClip.load` reads);
* `meta.json` — source url, **license and the machine-readable license evidence**, capture settings,
  neutral pose, validity stats, and a `harvest` block (source family, channel, speaker key, language
  guess, chunk offset, prescreen result).

`index.json` is one row per clip (all shards). `manifests/<shard>.json` is the same for one commit.

**{n_clips} clips, {hours:.1f} hours ({valid_hours:.1f} face-valid hours), {n_speakers} speaker keys, {n_lang} languages.**

## License policy (enforced in code, not remembered per file)

Only material whose *machine-readable* license metadata says public domain / CC0 / CC-BY is
captured; anything ND, NC, SA, or with missing license metadata is refused
(`scripts/fetch_sources.py::classify_license`, applied by `scripts/harvest/fetch.py`). Evidence per source:

| source | evidence recorded in `meta.json.license_evidence` |
|---|---|
| YouTube (CC search) | info-json `license` == "Creative Commons Attribution license (reuse allowed)" (CC-BY 3.0), channel id, upload date |
| U.S. Government channels | allowlisted official agency channel + channel id match; basis 17 U.S.C. § 105 (works of the U.S. Government). VOA clips carry a caveat about third-party newswire material |
| Wikimedia Commons | `videoinfo.extmetadata` LicenseShortName / License / LicenseUrl / Copyrighted=False |
| archive.org | item metadata `licenseurl` / `rights` |

Attribution: every clip's `meta.json` has `source_url`, `artist`/channel, and the license label;
CC-BY clips must be credited to that channel/author when redistributed. This dataset is a derived
work (motion parameters + downsampled audio), not the source video.

## Quality gate

Chunks of <= 600 s; prescreen (>= {pre_frac:.0%} of {pre_n} sampled frames show exactly one face);
kept if `face_valid >= {min_fv:.0%}` and `face_valid * duration >= {min_vs:.0f} s`. No channel contributes
more than {cap:.0%} of the kept seconds (fetch-time cap). Speaker keys over-merge (a channel = one
speaker), so a per-speaker cap at training time is conservative.

## Counts
"""

CARD_TAIL = """
## Provenance and reproducibility

Harvested by `scripts/harvest/` (crawl -> fetch -> workers -> push) on a CPU-only workstation; the
`harvest` block in each `meta.json` names the worker and chunk. Full pipeline description:
`docs/HARVEST.md`. Nothing here was captured from a source whose license metadata was missing.
"""


def unpushed(con) -> List[dict]:
    return [dict(r) for r in con.execute("SELECT * FROM clips WHERE state='kept' ORDER BY captured_at")]


def card(idx: Dict, repo: str) -> str:
    t = idx["totals"]
    h = t["kept_hours"]
    size_cat = "n<1K" if t["kept"] < 1000 else "1K<n<10K" if t["kept"] < 10000 else "10K<n<100K" if t["kept"] < 100000 else "100K<n<1M"
    s = CARD_HEAD.format(repo=repo, kbps=C.OPUS_KBPS, n_clips=t["kept"], hours=h, valid_hours=t["valid_hours"], n_speakers=t["speakers"],
                         n_lang=len(t["hours_by_language"]), pre_frac=C.PRESCREEN_MIN_FRAC, pre_n=C.PRESCREEN_FRAMES,
                         min_fv=C.MIN_FACE_VALID, min_vs=C.MIN_VALID_S, cap=C.CHANNEL_CAP_SHARE, size_cat=size_cat)
    lines = ["", "| source | hours |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in t["hours_by_source"].items()]
    lines += ["", "| license | hours |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in t["hours_by_license"].items()]
    lines += ["", "| language (guess) | hours |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in list(t["hours_by_language"].items())[:40]]
    lines += ["", "| series | hours |", "|---|---|"]
    lines += [f"| {k} | {v:.1f} |" for k, v in t["hours_by_series"].items()]
    return s + "\n".join(lines) + "\n" + CARD_TAIL


def ensure_repo(api, repo: str) -> None:
    from huggingface_hub import hf_hub_download

    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    try:
        p = hf_hub_download(repo, ".gitattributes", repo_type="dataset")
        ga = open(p, encoding="utf-8").read()
    except Exception:
        ga = ""
    if "*.opus" not in ga:
        ga = ga.rstrip("\n") + "\n*.opus filter=lfs diff=lfs merge=lfs -text\n*.parquet filter=lfs diff=lfs merge=lfs -text\n"
        api.upload_file(path_or_fileobj=ga.encode(), path_in_repo=".gitattributes", repo_id=repo, repo_type="dataset",
                        commit_message="lfs: opus + parquet")


def push_shard(con, api, repo: str, clips: List[dict], keep_local: bool) -> str:
    from huggingface_hub import CommitOperationAdd

    with C.tx(con):
        n = int(C.kv_get(con, "next_shard", "1"))
        C.kv_set(con, "next_shard", str(n + 1))
    shard = f"s{n:04d}"
    ops = []
    rows = []
    for c in clips:
        d = c["path"]
        if not os.path.isdir(d):
            con.execute("UPDATE clips SET state='lost' WHERE name=?", (c["name"],))
            continue
        for f in ("motion.parquet", "audio.opus", "audio.wav", "meta.json"):
            fp = os.path.join(d, f)
            if os.path.exists(fp):
                ops.append(CommitOperationAdd(path_in_repo=f"clips/{shard}/{c['name']}/{f}", path_or_fileobj=fp))
        row = json.loads(c["row"])
        row["shard"] = shard
        rows.append(row)
    if not rows:
        return ""
    man = os.path.join(C.STAGE, f"{shard}.json")
    with open(man, "w", encoding="utf-8") as fh:
        json.dump({"shard": shard, "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "clips": rows}, fh, ensure_ascii=False)
    ops.append(CommitOperationAdd(path_in_repo=f"manifests/{shard}.json", path_or_fileobj=man))
    hours = sum(r["duration_s"] for r in rows) / 3600
    C.log(f"pushing {shard}: {len(rows)} clips, {hours:.1f} h, {len(ops)} files")
    api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                      commit_message=f"{shard}: {len(rows)} clips, {hours:.1f} h")
    now = time.time()
    with C.tx(con):
        for r in rows:
            con.execute("UPDATE clips SET state='pushed', shard=?, pushed_at=? WHERE name=?", (shard, now, r["name"]))
    C.event(con, "push", f"{shard}: {len(rows)} clips {hours:.1f} h")
    if not keep_local:
        for c in clips:
            shutil.rmtree(c["path"], ignore_errors=True)
    return shard


def push_index(con, api, repo: str) -> None:
    idx = IDX.build(con)
    idx["clips"] = [r for r in idx["clips"] if r.get("push_state") == "pushed"]
    idx["totals"]["kept"] = len(idx["clips"])
    ip = os.path.join(C.STAGE, "index.json")
    with open(ip, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, ensure_ascii=False)
    rp = os.path.join(C.STAGE, "README.md")
    with open(rp, "w", encoding="utf-8") as fh:
        fh.write(card(idx, repo))
    from huggingface_hub import CommitOperationAdd
    api.create_commit(repo_id=repo, repo_type="dataset", commit_message=f"index: {idx['totals']['kept']} clips, {idx['totals']['kept_hours']} h",
                      operations=[CommitOperationAdd(path_in_repo="index.json", path_or_fileobj=ip),
                                  CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=rp)])


def local_gb(clips: List[dict]) -> float:
    return sum((c.get("bytes") or 0) for c in clips) / 1e9


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=os.environ.get("HARVEST_HF_REPO", "squaredcuber/animacy-human-motion-large"))
    p.add_argument("--min-hours", type=float, default=50.0)
    p.add_argument("--max-clips", type=int, default=600, help="clips per commit")
    p.add_argument("--max-local-gb", type=float, default=2.5)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep-local", action="store_true")
    p.add_argument("--check-every", type=float, default=600.0)
    a = p.parse_args(argv)
    C.ensure_dirs()
    from huggingface_hub import HfApi

    api = HfApi()
    con = C.db()
    ensure_repo(api, a.repo)
    while True:
        clips = unpushed(con)
        hours = sum(c["duration_s"] for c in clips) / 3600
        gb = local_gb(clips)
        due = a.force or hours >= a.min_hours or gb >= a.max_local_gb
        if due and clips:
            try:
                while clips:
                    batch, clips = clips[: a.max_clips], clips[a.max_clips:]
                    shard = push_shard(con, api, a.repo, batch, a.keep_local)
                    if shard:
                        C.log(f"pushed {shard}")
                push_index(con, api, a.repo)
                IDX.main(["--out", os.path.join(C.ROOT, "_index.json")])
                C.log(f"index pushed -> https://huggingface.co/datasets/{a.repo}")
            except Exception as exc:
                C.log(f"push failed: {type(exc).__name__}: {exc}; retrying in 5 min")
                C.event(con, "push_error", f"{type(exc).__name__}: {exc}"[:300])
                time.sleep(300)
                if not a.loop:
                    return 1
                continue
            a.force = False
        elif not a.loop:
            C.log(f"nothing due: {len(clips)} unpushed clips, {hours:.1f} h, {gb:.2f} GB (min {a.min_hours} h / {a.max_local_gb} GB)")
            return 0
        if not a.loop:
            return 0
        time.sleep(a.check_every)


if __name__ == "__main__":
    raise SystemExit(main())

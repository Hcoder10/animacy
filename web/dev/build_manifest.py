"""Write web/manifest.json: what the static viewer can load without probing.

    python web/dev/build_manifest.py

The page is a static site (GitHub Pages has no directory listing and every
missing-file probe is a red 404 in the console), so this script records, for
each robot in web/robots/*.json:
  * whether its `description.urdf` exists under robots/<name>/
  * its native clips (robots/<name>/clips/native/*.csv|*.json, with the
    descriptions from index.json when present)
and the captured canonical clips in web/clips/*.json.

Re-run it after adding clips or landing a URDF (screenshot.py and the tests
run it automatically).
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(WEB, ".."))


def build() -> dict:
    robots = {}
    native = {}
    for pj in sorted(glob.glob(os.path.join(WEB, "robots", "*.json"))):
        prof = json.load(open(pj, encoding="utf-8"))
        name = prof["name"]
        rdir = os.path.join(ROOT, "robots", name)
        urdf_rel = prof["description"]["urdf"]
        urdf_path = os.path.join(rdir, urdf_rel)
        robots[name] = {"urdf": urdf_rel, "exists": os.path.isfile(urdf_path)}
        clips = []
        ndir = os.path.join(rdir, (prof.get("native_clips") or {}).get("dir", "clips/native"))
        desc = {}
        idx = os.path.join(ndir, "index.json")
        if os.path.isfile(idx):
            try:
                data = json.load(open(idx, encoding="utf-8"))
                for e in (data.get("clips") if isinstance(data, dict) else data) or []:
                    if isinstance(e, dict) and e.get("name"):
                        desc[e["name"]] = e.get("description", "")
            except Exception as e:  # noqa: BLE001
                print(f"warning: could not read {idx}: {e}", file=sys.stderr)
        for f in sorted(glob.glob(os.path.join(ndir, "*.csv")) + glob.glob(os.path.join(ndir, "*.json"))):
            base = os.path.basename(f)
            if base == "index.json":
                continue
            stem = os.path.splitext(base)[0]
            clips.append({"file": base, "name": stem, "description": desc.get(stem, "")})
        native[name] = clips
    captured = []
    for f in sorted(glob.glob(os.path.join(WEB, "clips", "*.json"))):
        base = os.path.basename(f)
        if base == "index.json":
            continue
        captured.append({"file": base, "name": os.path.splitext(base)[0]})
    # motion models: web/models/<name>.onnx (+ optional <name>.json metadata). Absent = "coming soon".
    models = []
    for f in sorted(glob.glob(os.path.join(WEB, "models", "*.onnx"))):
        base = os.path.basename(f)
        stem = os.path.splitext(base)[0]
        entry = {"file": base, "name": stem, "bytes": os.path.getsize(f)}
        meta = os.path.join(WEB, "models", stem + ".json")
        if os.path.isfile(meta):
            try:
                entry["meta"] = json.load(open(meta, encoding="utf-8"))
                entry["meta_file"] = os.path.basename(meta)
            except Exception as e:  # noqa: BLE001
                print(f"warning: could not read {meta}: {e}", file=sys.stderr)
        models.append(entry)
    return {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "robots": robots, "native": native, "clips": captured, "models": models}


def main() -> int:
    m = build()
    out = os.path.join(WEB, "manifest.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=1)
    for n, r in m["robots"].items():
        print(f"{n}: urdf {r['urdf']} {'OK' if r['exists'] else 'MISSING (viewer uses the dev stand-in)'}; {len(m['native'][n])} native clips")
    print(f"{len(m['clips'])} captured clip(s) in web/clips; {len(m['models'])} model(s) in web/models; wrote {os.path.relpath(out, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""One-off: YouTube CC clips were labelled CC-BY-4.0 (classify_license's default when the string has no
version); YouTube's CC option is CC BY 3.0. Rewrites items, clips rows, local meta.json files and the
manifest.

    python scripts/harvest/fix_cc_label.py
"""
from __future__ import annotations

import json
import os

import common as C


def main() -> int:
    con = C.db()
    n_items = con.execute("UPDATE items SET license='CC-BY-3.0' WHERE backend='ytdlp' AND license='CC-BY-4.0'").rowcount
    n_rows = n_meta = 0
    for r in con.execute("SELECT name, path, row FROM clips").fetchall():
        row = json.loads(r["row"])
        ev = row.get("license_evidence") or {}
        if row.get("license") == "CC-BY-4.0" and "creative commons attribution" in str(ev.get("license", "")).lower():
            row["license"] = "CC-BY-3.0"
            con.execute("UPDATE clips SET row=? WHERE name=?", (json.dumps(row, ensure_ascii=False), r["name"]))
            n_rows += 1
            mp = os.path.join(r["path"], "meta.json")
            if os.path.exists(mp):
                m = json.load(open(mp, encoding="utf-8"))
                if m.get("license") == "CC-BY-4.0":
                    m["license"] = "CC-BY-3.0"
                    with open(mp, "w", encoding="utf-8") as fh:
                        json.dump(m, fh, indent=1, ensure_ascii=False)
                    n_meta += 1
    rows = [json.loads(r["row"]) for r in con.execute("SELECT row FROM clips ORDER BY captured_at")]
    with open(C.MANIFEST, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    C.log(f"items {n_items}, clip rows {n_rows}, meta.json {n_meta}; manifest rewritten ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""One screen of harvest status.

    python scripts/harvest/status.py [--json] [--hours 24]

Hours queued / fetched / captured / kept / pushed, per-source yield (what fraction of captured seconds
survived the drop rule, how many chunks the prescreen rejected), per-language and per-series kept
hours, throughput over the last 1 / 6 / 24 h, ETA to the target, rate-limit incidents, disk, workers.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time
from typing import Dict

import common as C


def collect(con, window_h: float) -> Dict:
    now = time.time()
    st = {r["state"]: {"n": r["n"], "hours": round((r["h"] or 0), 1)} for r in
          con.execute("SELECT state, COUNT(*) n, SUM(duration_s)/3600 h FROM items GROUP BY state")}
    kept = con.execute("SELECT COUNT(*) n, COALESCE(SUM(duration_s),0) s, COALESCE(SUM(valid_s),0) v, COALESCE(SUM(bytes),0) b FROM clips").fetchone()
    pushed = con.execute("SELECT COUNT(*) n, COALESCE(SUM(duration_s),0) s FROM clips WHERE state='pushed'").fetchone()
    local = con.execute("SELECT COUNT(*) n, COALESCE(SUM(duration_s),0) s, COALESCE(SUM(bytes),0) b FROM clips WHERE state='kept'").fetchone()

    def kept_h_since(h: float) -> float:
        r = con.execute("SELECT COALESCE(SUM(duration_s),0) s FROM clips WHERE captured_at > ?", (now - h * 3600,)).fetchone()
        return r["s"] / 3600

    def fetched_h_since(h: float) -> float:
        r = con.execute("SELECT COALESCE(SUM(duration_s),0) s FROM items WHERE fetched_at > ?", (now - h * 3600,)).fetchone()
        return r["s"] / 3600

    def captured_h_since(h: float) -> float:
        r = con.execute("SELECT COALESCE(SUM(captured_s),0) s FROM items WHERE captured_at > ?", (now - h * 3600,)).fetchone()
        return r["s"] / 3600

    first = con.execute("SELECT MIN(captured_at) t FROM clips").fetchone()["t"]
    elapsed_h = (now - first) / 3600 if first else 0.0
    thr = {f"{h}h": {"kept_h": round(kept_h_since(h), 2), "captured_h": round(captured_h_since(h), 2), "fetched_h": round(fetched_h_since(h), 2)}
           for h in (1, 6, 24)}
    rate = thr["6h"]["kept_h"] / min(6.0, max(elapsed_h, 1e-6)) if elapsed_h > 0 else 0.0
    if elapsed_h < 1.0 and elapsed_h > 0:
        rate = thr["1h"]["kept_h"] / max(elapsed_h, 1e-6)
    remaining = max(C.TARGET_HOURS - kept["s"] / 3600, 0.0)
    eta_h = remaining / rate if rate > 0 else float("inf")

    # per-source yield from finished items
    ysrc: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in con.execute("SELECT source_kind, state, COUNT(*) n, COALESCE(SUM(duration_s),0) d, COALESCE(SUM(captured_s),0) c, "
                         "COALESCE(SUM(kept_s),0) k, COALESCE(SUM(n_chunks),0) nc, COALESCE(SUM(n_kept),0) nk "
                         "FROM items GROUP BY source_kind, state"):
        y = ysrc[r["source_kind"] or "?"]
        y[f"n_{r['state']}"] += r["n"]
        if r["state"] in ("captured", "dropped"):
            y["done_h"] += r["d"] / 3600
            y["captured_h"] += r["c"] / 3600
            y["kept_h"] += r["k"] / 3600
            y["chunks"] += r["nc"]
            y["chunks_kept"] += r["nk"]
    # prescreen rejections + drop reasons from the per-item chunk records of the last window
    pre_rej = drop_rej = cap_ok = 0
    for r in con.execute("SELECT prescreen FROM items WHERE captured_at > ? AND prescreen IS NOT NULL", (now - window_h * 3600,)):
        try:
            for ch in json.loads(r["prescreen"]):
                res = str(ch.get("result", ""))
                if res == "prescreen":
                    pre_rej += 1
                elif res.startswith("dropped"):
                    drop_rej += 1
                elif res == "kept":
                    cap_ok += 1
        except Exception:
            pass
    per_lang: Dict[str, float] = collections.defaultdict(float)
    per_series: Dict[str, float] = collections.defaultdict(float)
    per_chan: Dict[str, float] = collections.defaultdict(float)
    for r in con.execute("SELECT c.duration_s d, i.language l, i.series s, i.channel_key k, i.channel_name cn FROM clips c JOIN items i ON i.id=c.item_id"):
        per_lang[r["l"] or "?"] += r["d"] / 3600
        per_series[r["s"] or "?"] += r["d"] / 3600
        per_chan[(r["cn"] or r["k"] or "?")[:40]] += r["d"] / 3600
    inc = con.execute("SELECT COUNT(*) n, MAX(ts) t FROM events WHERE kind='ratelimit' AND ts > ?", (now - 86400,)).fetchone()
    last_inc = con.execute("SELECT detail FROM events WHERE kind='ratelimit' ORDER BY ts DESC LIMIT 1").fetchone()
    hb = float(C.kv_get(con, "workers_heartbeat", "0"))
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"), "target_h": C.TARGET_HOURS,
        "kept": {"clips": kept["n"], "hours": round(kept["s"] / 3600, 2), "valid_hours": round(kept["v"] / 3600, 2)},
        "pushed": {"clips": pushed["n"], "hours": round(pushed["s"] / 3600, 2)},
        "local_unpushed": {"clips": local["n"], "hours": round(local["s"] / 3600, 2), "gb": round(local["b"] / 1e9, 2)},
        "items_by_state": st, "throughput": thr, "elapsed_h": round(elapsed_h, 2),
        "kept_h_per_wall_h": round(rate, 2), "eta_h_to_target": round(eta_h, 1) if eta_h != float("inf") else None,
        "yield_by_source": {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in ysrc.items()},
        f"chunks_last_{int(window_h)}h": {"kept": cap_ok, "dropped_after_capture": drop_rej, "prescreen_rejected": pre_rej},
        "hours_by_language": {k: round(v, 2) for k, v in sorted(per_lang.items(), key=lambda kv: -kv[1])[:20]},
        "hours_by_series": {k: round(v, 2) for k, v in sorted(per_series.items(), key=lambda kv: -kv[1])},
        "top_channels": {k: round(v, 2) for k, v in sorted(per_chan.items(), key=lambda kv: -kv[1])[:10]},
        "ratelimit_24h": {"n": inc["n"], "last": last_inc["detail"][:160] if last_inc else None},
        "disk_free_gb": round(C.disk_free_gb(), 1), "workers": {"n": C.kv_get(con, "workers_n", "?"), "heartbeat_age_s": round(now - hb) if hb else None},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true")
    p.add_argument("--hours", type=float, default=24.0)
    a = p.parse_args(argv)
    con = C.db()
    s = collect(con, a.hours)
    if a.json:
        print(json.dumps(s, indent=1, ensure_ascii=False))
        return 0
    k, pu, lo = s["kept"], s["pushed"], s["local_unpushed"]
    print(f"animacy harvest  {s['time']}   target {s['target_h']:.0f} h")
    print(f"  kept {k['hours']:.1f} h ({k['valid_hours']:.1f} face-valid h, {k['clips']} clips)   pushed {pu['hours']:.1f} h / {pu['clips']} clips   "
          f"local unpushed {lo['hours']:.1f} h {lo['gb']:.2f} GB   disk free {s['disk_free_gb']} GB")
    print("  queue: " + "  ".join(f"{st} {v['n']} ({v['hours']} h)" for st, v in sorted(s["items_by_state"].items())))
    t = s["throughput"]
    print(f"  throughput  1h: kept {t['1h']['kept_h']} h / captured {t['1h']['captured_h']} h / fetched {t['1h']['fetched_h']} h    "
          f"6h: kept {t['6h']['kept_h']} h    24h: kept {t['24h']['kept_h']} h    elapsed {s['elapsed_h']} h")
    eta = s["eta_h_to_target"]
    print(f"  rate {s['kept_h_per_wall_h']} kept h per wall h  ->  ETA to target " + (f"{eta:.0f} h ({eta / 24:.1f} days)" if eta else "n/a"))
    ch = s[f"chunks_last_{int(a.hours)}h"]
    print(f"  chunks last {a.hours:.0f} h: kept {ch['kept']}, dropped after capture {ch['dropped_after_capture']}, prescreen rejected {ch['prescreen_rejected']}")
    print("  yield by source (finished items):")
    for src, y in s["yield_by_source"].items():
        done = y.get("done_h", 0)
        print(f"    {src:<9} fetched-done {done:6.1f} h  captured {y.get('captured_h', 0):6.1f} h  kept {y.get('kept_h', 0):6.1f} h "
              f"({(y.get('kept_h', 0) / done * 100) if done else 0:4.0f}% of source time)  chunks kept {int(y.get('chunks_kept', 0))}/{int(y.get('chunks', 0))}  "
              f"refused {int(y.get('n_refused', 0))} failed {int(y.get('n_failed', 0))} queued {int(y.get('n_queued', 0))}")
    print("  by language (h): " + ", ".join(f"{k} {v}" for k, v in s["hours_by_language"].items()))
    print("  by series (h):   " + ", ".join(f"{k} {v}" for k, v in s["hours_by_series"].items()))
    print("  top channels (h): " + ", ".join(f"{k} {v}" for k, v in s["top_channels"].items()))
    r = s["ratelimit_24h"]
    print(f"  rate-limit incidents 24h: {r['n']}" + (f"   last: {r['last']}" if r["last"] else ""))
    w = s["workers"]
    print(f"  workers: {w['n']}  heartbeat {w['heartbeat_age_s']} s ago")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

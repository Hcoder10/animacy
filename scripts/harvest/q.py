"""Ad-hoc SQL against the harvest queue (read-only unless --write).

    python scripts/harvest/q.py "SELECT state, COUNT(*) FROM items GROUP BY state"
    python scripts/harvest/q.py "SELECT id, error FROM items WHERE state='failed'" --limit 20
"""
from __future__ import annotations

import argparse

import common as C


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sql")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    con = C.db()
    cur = con.execute(a.sql)
    if a.write:
        con.commit() if con.in_transaction else None
        print("ok", cur.rowcount)
        return 0
    rows = cur.fetchmany(a.limit)
    if rows:
        print(" | ".join(rows[0].keys()))
    for r in rows:
        print(" | ".join(str(v)[:160].replace("\n", " ") for v in r).encode("ascii", "replace").decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

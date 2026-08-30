"""
Audit the published track using ONLY what is published.

    python Live/5_Publish/verify_track.py

NO PANEL, NO ENGINE, NO PIPELINE, AND NO DEPENDENCIES.  Standard library only,
reading nothing but `docs/` and `Live/instrument_mapping.csv` -- both of which
are in the repository.  That is the entire point: this is the check an outside
reader can run on a clean checkout, which is what makes the published figures
something other than an assertion.

HOW THIS DIFFERS FROM `verify_publish` IN Update.py.  That one asks whether the
site agrees with the PIPELINE, and needs the parquet files to answer -- which
are not in the repository and never will be.  This one asks whether the site
agrees with ITSELF: does the arithmetic close, does the equity chain join up,
does the headline match the series behind it.  A stale publish passes this and
fails that; a corrupted publish fails both.  Neither replaces the other, and
only this one can run anywhere.

WHAT A FAILURE HERE MEANS.  Not "the strategy is wrong" -- this proves nothing
about the model.  It means the published record contradicts itself, which is
the one defect that would make every other number on the site unreadable.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
DATA = DOCS / "data"
MAPPING = ROOT / "Live" / "instrument_mapping.csv"
PUBLISH = ROOT / "Live" / "5_Publish" / "publish.py"
PAGES = ["index.html", "journal.html", "pnl.html", "mapping.html",
         "qa.html"]
FORBIDDEN = ["norgate"]

# Tolerances are the PUBLISHED ROUNDING, not a fudge.  Money is written to the
# cent, so a figure derived from 63 of them can sit a few cents from a figure
# rounded once; `history.json` rounds independently of the per-session books.
TOL_CENTS = 0.01
TOL_SUM = 1.0           # 63 instruments, each rounded to the cent
results: list[tuple[bool, str, str]] = []


def ok(label: str, cond: bool, detail: str = "") -> None:
    results.append((bool(cond), label, detail))


def load(rel: str):
    try:
        return json.loads((DATA / rel).read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    if not DATA.is_dir():
        ok("docs/data exists", False, f"not found at {DATA}")
        return report()

    # ---- 1. everything the site promises is present and parses -------------
    tops = ["latest.json", "history.json", "index.json", "pnl_index.json",
            "mapping.json", "qa.json"]
    got = {n: load(n) for n in tops}
    bad = [n for n in tops if got[n] is None]
    ok("every top-level file present and parses", not bad,
       f"{len(tops) - len(bad)}/{len(tops)}" + (f"   bad: {bad}" if bad else ""))
    if bad:
        return report()

    latest, hist = got["latest.json"], got["history.json"]
    idx = got["index.json"]["days"]
    pidx = got["pnl_index.json"]["days"]
    mapping = got["mapping.json"]["instruments"]

    missing = [x["date"] for x in idx
               if load(f"days/{x['date']}.json") is None]
    ok("every journal session file exists and parses", not missing,
       f"{len(idx) - len(missing)}/{len(idx)}"
       + (f"   bad: {missing[:4]}" if missing else ""))

    books, missing = {}, []
    for x in pidx:
        j = load(f"pnl/{x['date']}.json")
        if j is None:
            missing.append(x["date"])
        else:
            books[x["date"]] = j
    ok("every attribution session file exists and parses", not missing,
       f"{len(pidx) - len(missing)}/{len(pidx)}"
       + (f"   bad: {missing[:4]}" if missing else ""))
    if missing:
        return report()
    dates = [x["date"] for x in pidx]

    # ---- 2. each sheet adds up to its own book -----------------------------
    off = [(d, abs(sum(i["gross_pnl_USD"] for i in books[d]["instruments"])
                   - books[d]["book"]["gross_pnl_USD"])) for d in dates]
    worst = max(off, key=lambda t: t[1])
    ok("every attribution sheet sums to its own book",
       worst[1] <= TOL_SUM, f"worst {worst[1]:.3f} on {worst[0]}")

    # ---- 3. the balance sheet closes, every session ------------------------
    def walk(d):
        b = books[d]["book"]
        return abs(b["opening_equity_USD"] + b["gross_pnl_USD"]
                   - b["commission_USD"] + b["interest_USD"]
                   - b["closing_equity_USD"])
    worst = max(((d, walk(d)) for d in dates), key=lambda t: t[1])
    ok("opening + gross - commission + interest == closing",
       worst[1] <= TOL_CENTS, f"worst {worst[1]:.2e} on {worst[0]}")

    # ---- 4. THE CHAIN.  One session's close is the next one's open ---------
    #
    # The check a fabricated track fails.  Individually plausible sessions can
    # each close perfectly and still not join up; this insists the equity curve
    # is one object rather than 171 unrelated statements.
    breaks = [(b, abs(books[a]["book"]["closing_equity_USD"]
                      - books[b]["book"]["opening_equity_USD"]))
              for a, b in zip(dates, dates[1:])]
    worst = max(breaks, key=lambda t: t[1]) if breaks else ("-", 0.0)
    ok("the equity chain joins: close[t] == open[t+1]",
       worst[1] <= TOL_CENTS,
       f"{len(breaks)} joins, worst {worst[1]:.2e} on {worst[0]}")

    # ---- 5. the curve agrees with the sessions behind it -------------------
    H = {r["date"]: r for r in hist["daily"]}
    ok("history covers exactly the published sessions",
       set(H) == set(dates), f"{len(H)} rows vs {len(dates)} sessions")
    if set(H) == set(dates):
        worst_all = 0.0
        worst_at = ""
        for d in dates:
            b = books[d]["book"]
            for hk, bk in (("equity_USD", "closing_equity_USD"),
                           ("gross_pnl_USD", "gross_pnl_USD"),
                           ("commission_USD", "commission_USD"),
                           ("interest_USD", "interest_USD")):
                e = abs(H[d][hk] - b[bk])
                if e > worst_all:
                    worst_all, worst_at = e, f"{d} {hk}"
        ok("history.json agrees with every session's book",
           worst_all <= TOL_CENTS, f"worst {worst_all:.2e} on {worst_at}")

    # ---- 6. the headline is the series it came from ------------------------
    m = latest["meta"]
    last = hist["daily"][-1]
    diffs = []
    if abs(m["equity_end"] - last["equity_USD"]) > TOL_CENTS:
        diffs.append(f"equity {m['equity_end']} vs {last['equity_USD']}")
    if m["sessions"] != len(hist["daily"]):
        diffs.append(f"sessions {m['sessions']} vs {len(hist['daily'])}")
    if m["as_of"] != last["date"]:
        diffs.append(f"as_of {m['as_of']} vs {last['date']}")
    ok("latest.json is the last row of history.json", not diffs,
       f"{m['as_of']}  {m['sessions']} sessions  {m['equity_end']:,.2f}"
       if not diffs else "  ".join(diffs))

    # ---- 7. the contract specs are well formed and obey the cost model -----
    #
    # `cost_rt` is one tick plus a fee floor, so it can never be at or below a
    # single tick.  That is the published cost methodology stated as a test.
    bad = [r["instrument"] for r in mapping
           if not (r["pointsize"] > 0 and r["tick_size"] > 0
                   and r["cost_rt_local"] > r["tick_size"] * r["pointsize"])]
    ok("specs positive, and cost_rt exceeds one tick",
       not bad, f"{len(mapping)} contracts" + (f"   bad: {bad[:4]}" if bad else ""))

    csv_n = sum(1 for _ in MAPPING.read_text(encoding="utf-8").splitlines()) - 1
    ok("mapping.json covers the whole instrument file",
       csv_n == len(mapping), f"{len(mapping)} published vs {csv_n} in the csv")

    # ---- 8. the pages agree on which assets they are loading --------------
    stamps = set()
    for name in PAGES:
        f = DOCS / name
        if not f.is_file():
            stamps.add(f"MISSING:{name}")
            continue
        for mt in re.finditer(r'(?:app\.js|site\.css)\?v=([A-Za-z0-9]+)',
                              f.read_text(encoding="utf-8")):
            stamps.add(mt.group(1))
    ok("all pages load the same asset version", len(stamps) == 1,
       f"{sorted(stamps)}")

    # ---- 9. the pages and the payload agree on the key names --------------
    #
    # THE ONE FAILURE NEITHER OTHER SUITE CATCHES.  Rename a key in
    # `publish.py` and not in `app.js` and everything still passes: the JSON is
    # valid, the whitelist is satisfied, the arithmetic reconciles -- and the
    # page renders "undefined" because it is asking for a name nobody publishes
    # any more.  Structure and arithmetic are both fine; only the CONTRACT
    # between the two files is broken.
    #
    # The vocabulary is taken from what publish.py PROMISES, not from what
    # today's data happens to contain: `carried_sessions` is a declared column
    # that vanishes from the payload whenever no order is outstanding, and
    # reading only the JSON would call that a break.  Parsed with `ast` rather
    # than imported, because importing it would drag in polars and this suite
    # is dependency-free on purpose.
    vocab = set()
    def _keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                vocab.add(k); _keys(v)
        elif isinstance(o, list):
            for v in o[:3]:
                _keys(v)
    for name in tops:
        _keys(got[name])
    _keys(load(f"days/{dates[-1]}.json"))
    _keys(load(f"pnl/{dates[-1]}.json"))
    declared = 0
    try:
        tree = ast.parse(PUBLISH.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                nm = getattr(t, "id", "")
                if nm.endswith("_COLS") or nm.endswith("_KEYS"):
                    try:
                        vals = ast.literal_eval(node.value)
                    except Exception:
                        continue
                    if isinstance(vals, list):
                        vocab |= {v for v in vals if isinstance(v, str)}
                        declared += 1
    except Exception:
        pass

    src = "".join((DOCS / f).read_text(encoding="utf-8")
                  for f in ["app.js"] + PAGES)
    # The JSON-reading idiom in these pages is a property off a single-letter
    # object (`r.instrument`, `m.equity_end`, `b.gross_pnl_USD`).
    refs = set(re.findall(r"[^A-Za-z0-9_]([a-z])[.]([A-Za-z_][A-Za-z0-9_]*)", src))
    names = {n for _, n in refs}
    looks_published = {n for n in names if "_" in n or n in vocab}
    orphan = sorted(looks_published - vocab)
    ok("every key the pages read is one publish.py declares", not orphan,
       f"{len(looks_published)} keys read, {len(vocab)} declared across "
       f"{declared} lists" + (f"   ORPHANS: {orphan}" if orphan else ""))

    # ---- 9b. the Q&A payload agrees with the track it describes ------------
    #
    # The largest payload on the site and, until this check, the only one
    # nothing verified. Presence is not enough: its benchmark curve is an
    # independent reconstruction of the same equity, so if it disagrees with
    # `latest.json` one of the two is describing a different run.
    qa = got.get("qa.json") or {}
    bc = qa.get("bench_curves") or []
    diffs = []
    if len(bc) != m["sessions"]:
        diffs.append(f"{len(bc)} curve rows vs {m['sessions']} sessions")
    if bc and abs(bc[-1]["book"] - m["equity_end"]) > TOL_CENTS:
        diffs.append(f"ends {bc[-1]['book']} vs {m['equity_end']}")
    npos = len(qa.get("positions") or [])
    if npos != m.get("n_positions", npos):
        diffs.append(f"{npos} positions vs {m.get('n_positions')}")
    ok("qa.json agrees with the published headline", not diffs,
       f"{len(bc)} sessions, {npos} positions" if not diffs
       else "  ".join(diffs))

    # ---- 10. the carve-out holds ------------------------------------------
    hits = []
    for f in DOCS.rglob("*"):
        if f.is_file():
            try:
                low = f.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            if any(w in low for w in FORBIDDEN):
                hits.append(f.relative_to(DOCS).as_posix())
    ok("the data provider is named in no published file", not hits,
       f"{sum(1 for f in DOCS.rglob('*') if f.is_file()):,} files scanned"
       + (f"   HITS: {hits[:3]}" if hits else ""))

    return report()


def report() -> int:
    bar = "=" * 72
    print(bar)
    print("  VERIFY  the published track, from the published files alone")
    print(bar)
    for good, label, detail in results:
        print(f"  [{'OK  ' if good else 'FAIL'}] {label:<52}{detail}")
    bad = sum(1 for g, _, _ in results if not g)
    print("  " + "-" * 68)
    print(f"  {len(results) - bad}/{len(results)} passed"
          + (f"   {bad} FAILED" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

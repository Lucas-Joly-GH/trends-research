from __future__ import annotations

import argparse
import datetime as _dt
from datetime import date as _date
from pathlib import Path

import numpy as np
import polars as pl

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _private() -> Path:
    for p in Path(__file__).resolve().parents:
        cand = p / "LJOLY_Memoire_INSEEC_Msc2"
        if cand.is_dir():
            return cand
    raise SystemExit("[ABORT] cannot find LJOLY_Memoire_INSEEC_Msc2 above "
                     f"{Path(__file__).resolve()}")


PRIVATE = _private()
CONTRACTS = PRIVATE / "Data" / "Paper-trading" / "Contracts"
NOTICE = PRIVATE / "Data" / "Contract-Metadata" / "contracts.csv"
WORKING = Path(__file__).resolve().parent / "worksheet.csv"
CYCLES = Path(__file__).resolve().parents[1] / "contract_cycles.csv"
MONTH_ORDER = "FGHJKMNQUVXZ"

BAD_ROWS = {
    ("HO", "2007-09-03", "HO-2008G"),
}

BAD_SESSIONS = {
    ("LFT9", d) for d in (
        "2025-05-06", "2025-05-07", "2025-05-08", "2025-05-12", "2025-05-13",
        "2025-05-14", "2025-05-15", "2025-05-16", "2025-05-19", "2025-05-20",
        "2025-05-21", "2025-05-22", "2025-05-23", "2025-05-27", "2025-05-28",
        "2025-05-29", "2025-05-30",
    )
}

BV3_SESSIONS = 3
AUTO_ROLL_CD = 5
BEST_VOL_MIN_CD = 5
FORCED_ROLL_MIN_CD = 5
ROLL_CONFIRM_SESSIONS = 2
INCEPTION_VOLUME = 1000

BOOL_COLS = ["is_passed", "Best_Vol", "Best_Oi", "B_V_3", "Auto_Best_V",
             "Forced_Best_V", "+2_Forced_Best_V", "Test_Best_V", "f_r_h_Best_V"]

GATE = {"notice": ("first_notice", "till_notice_cd"),
        "last_trade": ("last_trade", "till_last_trade_cd")}


def sort_key(sym: str) -> int:
    tail = sym.split("-")[1]
    return int(tail[:4]) * 12 + MONTH_ORDER.index(tail[4])


def ratchet(cand: str | None, prev: str | None, till: dict) -> str | None:
    if cand is None or prev is None or cand == prev:
        return cand
    a, b = till.get(prev), till.get(cand)
    return prev if (a is not None and b is not None and b < a) else cand


def gate(inst: str) -> str:
    if not CYCLES.exists():
        return "last_trade"
    t = pl.read_csv(CYCLES, infer_schema_length=0)
    row = t.filter(pl.col("instrument") == inst)
    if not row.height:
        return "last_trade"
    return ("notice" if row.get_column("has_notice").to_list()[0] == "true"
            else "last_trade")


def gate_map(inst: str, which: str) -> dict:
    if not NOTICE.exists():
        return {}
    col = "first_notice" if which == "notice" else "last_trade"
    t = (pl.read_csv(NOTICE, infer_schema_length=0)
         .filter(pl.col("instrument") == inst)
         .select(["symbol", col]))
    return {r["symbol"]: r[col] for r in t.iter_rows(named=True) if r[col]}


def load(inst: str) -> pl.DataFrame:
    rows = []
    for f in sorted((CONTRACTS / inst).glob(f"{inst}-*.csv")):
        t = (pl.read_csv(f, infer_schema_length=0)
             .select(["Date", "Open", "Close", "Volume", "Open Interest"])
             .with_columns(
                 pl.col("Date").cast(pl.Utf8),
                 pl.col("Open").cast(pl.Float64, strict=False),
                 pl.col("Close").cast(pl.Float64, strict=False),
                 pl.col("Volume").cast(pl.Float64, strict=False).fill_null(0.0),
                 pl.col("Open Interest").cast(pl.Float64, strict=False).fill_null(0.0),
                 pl.lit(f.stem).alias("symbol")))
        rows.append(t)
    if not rows:
        raise SystemExit(f"[ABORT] no contracts under {CONTRACTS / inst}")
    d = pl.concat(rows).rename({"Open": "open", "Close": "close",
                                "Volume": "volume",
                                "Open Interest": "open_interest"})
    d = d.with_columns(
        pl.col("Date").str.strptime(pl.Date, "%Y%m%d").alias("date")).drop("Date")
    drop = [_date.fromisoformat(dt) for (i, dt) in BAD_SESSIONS if i == inst]
    if drop:
        d = d.filter(~pl.col("date").is_in(drop))
    for (i, dt, sym) in BAD_ROWS:
        if i != inst:
            continue
        d = d.filter(~((pl.col("date") == _date.fromisoformat(dt))
                       & (pl.col("symbol") == sym)))
    return d


def dead_months(inst: str) -> set:
    if not CYCLES.exists():
        return set()
    t = pl.read_csv(CYCLES, infer_schema_length=0)
    if "Dead_contracts" not in t.columns:
        return set()
    row = t.filter(pl.col("instrument") == inst)
    return set(row.get_column("Dead_contracts").to_list()[0] or "") if row.height else set()


def inception(d: pl.DataFrame, dead: set) -> object:
    sel = (d.filter(~pl.col("symbol").str.slice(-1).is_in(list(dead)))
           if dead else d)
    by_day = (sel.group_by("date").agg(pl.col("volume").max().alias("mx"))
               .filter(pl.col("mx") >= INCEPTION_VOLUME).sort("date"))
    return by_day.get_column("date").to_list()[0] if by_day.height else None


STALE_SESSIONS = 3
QUORUM_FRAC = 0.15
QUORUM_MIN = 5
EDGE_WINDOW = 40


def _tail_dates(path: Path, window: int = EDGE_WINDOW) -> list[str]:
    try:
        with path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 64 * window))
            tail = fh.read().splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(tail):
        head = line.decode("utf-8", "ignore").split(",", 1)[0]
        if len(head) == 8 and head.isdigit():
            out.append(head)
            if len(out) >= window:
                break
    return out


EDGE_LOOKBACK_DAYS = 15


def _contract_last_dates(instruments: list[str] | None = None) -> dict:
    out = {}
    for d in sorted(CONTRACTS.iterdir()):
        if not d.is_dir():
            continue
        if instruments is not None and d.name not in instruments:
            continue
        got = {}
        for f in d.glob(f"{d.name}-[0-9][0-9][0-9][0-9][A-Z].csv"):
            v = _tail_dates(f, window=1)
            if v:
                got[f] = v[0]
        if got:
            out[d.name] = got
    return out


def panel_sessions(instruments: list[str] | None = None) -> dict:
    per = _contract_last_dates(instruments)
    newest = max((d for g in per.values() for d in g.values()), default="")
    if not newest:
        return {}
    cut = (_dt.date(int(newest[:4]), int(newest[4:6]), int(newest[6:]))
           - _dt.timedelta(days=EDGE_LOOKBACK_DAYS)).strftime("%Y%m%d")
    out = {}
    for inst, files in per.items():
        seen: set = set()
        for f, last in files.items():
            if last < cut:
                continue
            seen.update(d for d in _tail_dates(f) if d >= cut)
        if seen:
            out[inst] = seen
    return out


def panel_edge(instruments: list[str] | None = None) -> dict:
    return {i: max(g.values())
            for i, g in _contract_last_dates(instruments).items()}


_AS_OF: tuple[str, dict] | None = None


def panel_as_of(instruments: list[str] | None = None, *,
                refresh: bool = False) -> tuple[str, dict]:
    global _AS_OF
    if _AS_OF is not None and not refresh and instruments is None:
        return _AS_OF

    edge = panel_edge(instruments)
    sess = panel_sessions(instruments)
    if not edge:
        return "", {}
    counts: dict[str, int] = {}
    for s in sess.values():
        for d in s:
            counts[d] = counts.get(d, 0) + 1
    need = max(QUORUM_MIN, int(round(len(edge) * QUORUM_FRAC)))
    quorate = [d for d, n in counts.items() if n >= need]
    as_of = max(quorate) if quorate else min(edge.values())
    got = as_of, edge
    if instruments is None:
        _AS_OF = got
    return got


def report_edge(as_of: str, edge: dict, *, stale: int = STALE_SESSIONS) -> list:
    ahead = sorted(i for i, d in edge.items() if d > as_of)
    behind = sorted((d, i) for i, d in edge.items() if d < as_of)
    print(f"  as_of {as_of}   {len(edge)} instruments")
    if ahead:
        print(f"    {len(ahead)} ahead, held back to as_of: "
              f"{', '.join(f'{i} ({edge[i]})' for i in ahead)}")
    lagging = []
    if behind:
        a = _dt.date(int(as_of[:4]), int(as_of[4:6]), int(as_of[6:]))
        for d, i in behind:
            b = _dt.date(int(d[:4]), int(d[4:6]), int(d[6:]))
            gap = (a - b).days
            if gap > stale * 2:
                lagging.append((i, d, gap))
        closed = len(behind) - len(lagging)
        if closed:
            print(f"    {closed} behind as_of -- market shut that session")
        for i, d, gap in lagging:
            print(f"      [STALE] {i} last traded {d}, {gap} calendar days back")
    return lagging


def worksheet(inst: str, start: str, end: str,
              as_of: str | None = "auto") -> pl.DataFrame:
    which = gate(inst)
    date_col, till_col = GATE[which]
    gates = gate_map(inst, which)

    d = load(inst)
    if as_of == "auto":
        as_of, _ = panel_as_of()
    if as_of:
        cap = _dt.date(int(as_of[:4]), int(as_of[4:6]), int(as_of[6:]))             if as_of.isdigit() else _dt.date.fromisoformat(as_of)
        d = d.filter(pl.col("date") <= cap)
        if not d.height:
            raise SystemExit(f"[ABORT] {inst}: no bars at or before {as_of}")
    dead = dead_months(inst)
    if dead:
        d = d.filter(~pl.col("symbol").str.slice(-1).is_in(list(dead)))
    start_at = inception(d, dead)
    if start_at is not None:
        d = d.filter(pl.col("date") >= start_at)
    lo, hi = np.datetime64(start, "D"), np.datetime64(end, "D")

    warm = (d.filter(pl.col("date") <= pl.lit(hi).cast(pl.Date))
             .sort(["date", "symbol"]))
    streak: dict[str, int] = {}
    prev_hold: str | None = None
    prev_forced: str | None = None
    cf_hold: str | None = None
    cf_pending: str | None = None
    cf_pending_n = 0
    rank = {sym: i for i, sym in enumerate(
        sorted(d.get_column("symbol").unique().to_list(), key=sort_key))}
    out = []

    for day, g in warm.group_by("date", maintain_order=True):
        day = day[0]
        recs = []
        for r in g.iter_rows(named=True):
            fn = gates.get(r["symbol"], "")
            till = (int((np.datetime64(fn, "D") - np.datetime64(day, "D")).astype(int))
                    if fn else None)
            passed = (till is not None and till <= 0)
            recs.append({**r, "gate_date": fn, "till": till,
                         "is_passed": passed})

        def in_window(r):
            return r["till"] is None or r["till"] > BEST_VOL_MIN_CD

        best_v_pool = [r for r in recs
                       if not r["is_passed"] and r["volume"] > 0 and in_window(r)]
        best_v = (max(best_v_pool, key=lambda r: r["volume"])["symbol"]
                  if best_v_pool else None)

        best_o_pool = [r for r in recs
                       if not r["is_passed"] and r["open_interest"] > 0
                       and in_window(r)]
        best_o = (max(best_o_pool, key=lambda r: r["open_interest"])["symbol"]
                  if best_o_pool else None)

        for r in recs:
            sym = r["symbol"]
            streak[sym] = streak.get(sym, 0) + 1 if sym == best_v else 0
        best_3 = {s for s, n in streak.items() if n >= BV3_SESSIONS}

        cal = sorted(recs, key=lambda r: rank[r["symbol"]])
        auto = None
        if cal:
            first = cal[0]
            if first["till"] is not None and first["till"] <= AUTO_ROLL_CD:
                auto = cal[1]["symbol"] if len(cal) > 1 else None
            else:
                auto = first["symbol"]

        plus1 = None
        if cal:
            i = 2 if (cal[0]["till"] is not None
                      and cal[0]["till"] <= AUTO_ROLL_CD) else 1
            plus1 = cal[i]["symbol"] if len(cal) > i else None

        auto_best_v = auto is not None and best_v is not None and auto == best_v

        held = [r for r in recs
                if r["till"] is not None
                and r["till"] > FORCED_ROLL_MIN_CD
                and r["symbol"] != auto]
        forced = (min(held, key=lambda r: (r["till"], rank[r["symbol"]]))["symbol"]
                  if held and not auto_best_v else None)
        forced_best_v = (forced is not None and best_v is not None
                         and forced == best_v)

        held2 = [r for r in held
                 if r["symbol"] != forced and r["till"] > FORCED_ROLL_MIN_CD]
        forced2 = (min(held2, key=lambda r: (r["till"], rank[r["symbol"]]))["symbol"]
                   if held2 and not auto_best_v else None)
        forced2_best_v = (forced2 is not None and best_v is not None
                          and forced2 == best_v)

        till = {r["symbol"]: r["till"] for r in recs}

        forced_hold = ratchet(
            (auto if auto_best_v
             else forced if forced_best_v
             else best_o if best_o is not None
             else best_v),
            prev_forced, till)
        if forced_hold is not None:
            prev_forced = forced_hold
        frh_best_v = (forced_hold is not None and best_v is not None
                      and forced_hold == best_v)

        cf_cand = ratchet(
            (auto if auto_best_v
             else forced if forced_best_v
             else best_o if best_o is not None
             else best_v),
            cf_hold, till)
        if cf_hold is None or cf_hold not in till:
            cf_forced_out = True
        else:
            v = till[cf_hold]
            cf_forced_out = v is not None and v <= FORCED_ROLL_MIN_CD
        if cf_cand is None or cf_cand == cf_hold:
            cf_pending, cf_pending_n = None, 0
        elif cf_cand == cf_pending:
            cf_pending_n += 1
        else:
            cf_pending, cf_pending_n = cf_cand, 1
        if cf_cand is not None and (cf_forced_out
                                    or cf_pending_n >= ROLL_CONFIRM_SESSIONS):
            cf_hold = cf_cand
            cf_pending, cf_pending_n = None, 0

        test_hold = ratchet(
            (auto if auto_best_v
             else forced if forced_best_v
             else forced2),
            prev_hold, till)
        if test_hold is not None:
            prev_hold = test_hold
        test_best_v = (test_hold is not None and best_v is not None
                       and test_hold == best_v)

        if np.datetime64(day, "D") < lo:
            continue

        for r in recs:
            blank = r["is_passed"]
            out.append({
                "date": str(day), "symbol": r["symbol"],
                "open": r["open"], "close": r["close"],
                "volume": int(r["volume"]),
                "open_interest": int(r["open_interest"]),
                date_col: r["gate_date"],
                till_col: r["till"],
                "is_passed": "true" if r["is_passed"] else "false",
                "Best_Vol": "" if blank else str(r["symbol"] == best_v).lower(),
                "Best_Oi": "" if blank else str(r["symbol"] == best_o).lower(),
                "B_V_3": "" if blank else str(r["symbol"] in best_3).lower(),
                "auto_roll": auto or "",
                "Auto_Best_V": str(auto_best_v).lower(),
                "Forced_roll_V": forced or "",
                "Forced_Best_V": ("" if forced is None
                                  else str(forced_best_v).lower()),
                "+2_Forced_Roll_V": forced2 or "",
                "+2_Forced_Best_V": ("" if forced2 is None
                                     else str(forced2_best_v).lower()),
                "auto_roll_hold": auto or "",
                "+1_auto_roll_hold": plus1 or "",
                "forced_roll_hold": forced_hold or "",
                "f_r_h_Best_V": str(frh_best_v).lower(),
                "confirm_forced_roll_hold": cf_hold or "",
                "Test_Hold": test_hold or "",
                "Test_Best_V": str(test_best_v).lower(),
            })

    return pl.DataFrame(out).sort(["date", "symbol"])


def means(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    for c in df.columns:
        if c in BOOL_COLS:
            exprs.append(pl.when(pl.col(c) == "true").then(1.0)
                           .when(pl.col(c) == "false").then(0.0)
                           .otherwise(None).mean().alias(c))
        elif df.schema[c] != pl.Utf8:
            exprs.append(pl.col(c).cast(pl.Float64).mean().alias(c))
    return df.select(exprs)


def session_means(df: pl.DataFrame) -> pl.DataFrame:
    one = df.unique(subset=["date"], keep="first").sort("date")
    return one.select([
        pl.when(pl.col("Auto_Best_V") == "true").then(1.0).otherwise(0.0)
          .mean().alias("Auto_Best_V"),
        pl.when(pl.col("f_r_h_Best_V") == "true").then(1.0).otherwise(0.0)
          .mean().alias("f_r_h_Best_V"),
        pl.col("date").n_unique().alias("sessions")])


def check(got: pl.DataFrame, ref_path: Path) -> int:
    ref = pl.read_csv(ref_path, infer_schema_length=0)
    got = got.with_columns([pl.col(c).cast(pl.Utf8) for c in got.columns])
    ref = ref.with_columns([pl.col(c).fill_null("").cast(pl.Utf8) for c in ref.columns])
    got = got.with_columns([pl.col(c).fill_null("") for c in got.columns])
    if got.height != ref.height:
        print(f"[FAIL] {got.height} rows against {ref.height}")
        return 1
    bad = 0
    for c in ref.columns:
        if c not in got.columns:
            print(f"[FAIL] missing column {c}")
            bad += 1
            continue
        try:
            a = ref.get_column(c).cast(pl.Float64, strict=True)
            b = got.get_column(c).cast(pl.Float64, strict=True)
            n = int(((a - b).abs() > 1e-9).sum())
        except Exception:
            n = int((ref.get_column(c) != got.get_column(c)).sum())
        if n:
            print(f"[FAIL] {c}: {n} differing rows")
            bad += 1
    print("[OK] identical" if not bad else f"{bad} column(s) differ")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="ZC")
    ap.add_argument("--start", default="1989-05-08")
    ap.add_argument("--end", default="1989-05-23")
    ap.add_argument("--out", default=None)
    ap.add_argument("--means", action="store_true",
                    help="print pl.mean() of every column instead of the head")
    ap.add_argument("--check", default=None,
                    help="compare against a reference worksheet instead of writing")
    ap.add_argument("--as-of", default="auto", dest="as_of",
                    help="square the panel off at this date (YYYYMMDD or ISO). "
                         "Default 'auto': the newest date EVERY instrument has. "
                         "Pass 'none' to read the panel ragged.")
    ap.add_argument("--edge", action="store_true",
                    help="print each instrument's newest bar and stop")
    args = ap.parse_args()

    if args.edge:
        as_of, edge = panel_as_of()
        print("PANEL EDGE")
        report_edge(as_of, edge)
        return 0

    as_of = None if args.as_of == "none" else args.as_of
    if as_of == "auto":
        as_of, edge = panel_as_of()
        print("PANEL EDGE")
        report_edge(as_of, edge)
        print("")

    df = worksheet(args.instrument, args.start, args.end, as_of=as_of)
    if args.check:
        return check(df, Path(args.check))

    out = Path(args.out) if args.out else WORKING
    df.write_csv(out)
    with pl.Config(tbl_rows=30, tbl_cols=28, tbl_width_chars=320):
        print(means(df) if args.means else df.head(14))
        if args.means:
            print(session_means(df))
    print(f"\n{df.height} rows -> {out}")
    print(f"gate: {gate(args.instrument)}   "
          f"sessions {df.get_column('date').n_unique()}"
          + (f"   as_of {as_of}" if as_of else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

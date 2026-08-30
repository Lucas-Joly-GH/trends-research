from __future__ import annotations

import argparse
import csv
import datetime as _dt
import platform
import re
import sys
import time
from pathlib import Path

_NODE = platform.node()
if not _NODE.isascii():
    platform.node = lambda _n=_NODE.encode("ascii", "ignore").decode(): _n

import polars as pl

CASH_SETTLED = {
    "BRN", "BTC", "ETH", "EMD", "ES", "FDAX9", "FESX9", "GF", "HE", "HSI",
    "LEU9", "LFT9", "NIY", "NQ", "RTY", "SJB", "SO3", "SR3", "SXF", "VX",
    "YAP4", "YXT4",
}

DEAD_CONTRACTS = {
    "CT": "V",
    "EUA": "FGHJKMNQUVX",
    "GC": "FHKNUVX",
    "HE": "K",
    "HG": "FGJMQVX",
    "LE": "KNUX",
    "LEU9": "FGJKNQVX",
    "PL": "GHKMQUXZ",
    "SB": "FU",
    "SI": "FGJMQVX",
    "YAP4": "FGJKNQVX",
    "ZC": "FX",
}
ACTIVE_CONTRACTS = {
    "6A": "HMUZ",
    "6B": "HMUZ",
    "6C": "HMUZ",
    "6E": "HMUZ",
    "6J": "HMUZ",
    "6M": "HMUZ",
    "6N": "HMUZ",
    "6S": "HMUZ",
    "BRN": "FGHJKMNQUVXZ",
    "BTC": "FGHJKMNQUVXZ",
    "CC": "HKNUZ",
    "CGB": "HMUZ",
    "CL": "FGHJKMNQUVXZ",
    "CT": "HKNZ",
    "DX": "HMUZ",
    "EMD": "HMUZ",
    "ES": "HMUZ",
    "ETH": "FGHJKMNQUVXZ",
    "EUA": "Z",
    "FDAX9": "HMUZ",
    "FESX9": "HMUZ",
    "FGBL9": "HMUZ",
    "FGBM9": "HMUZ",
    "FGBS9": "HMUZ",
    "GAS": "FGHJKMNQUVXZ",
    "GC": "GJMQZ",
    "GF": "FHJKQUVX",
    "HE": "GJMNQVZ",
    "HG": "HKNUZ",
    "HO": "FGHJKMNQUVXZ",
    "HSI": "FGHJKMNQUVXZ",
    "KC": "HKNUZ",
    "LE": "GJMQVZ",
    "LEU9": "HMUZ",
    "LFT9": "HMUZ",
    "LLG": "HMUZ",
    "NG": "FGHJKMNQUVXZ",
    "NIY": "HMUZ",
    "NQ": "HMUZ",
    "PA": "HMUZ",
    "PL": "FJNV",
    "RB": "FGHJKMNQUVXZ",
    "RS": "FHKMNQUX",
    "RTY": "HMUZ",
    "SB": "HKNV",
    "SI": "HKNUZ",
    "SJB": "HMUZ",
    "SO3": "HMUZ",
    "SR3": "HMUZ",
    "SXF": "HMUZ",
    "UB": "HMUZ",
    "VX": "FGHJKMNQUVXZ",
    "YAP4": "HMUZ",
    "YXT4": "HMUZ",
    "ZB": "HMUZ",
    "ZC": "HKNUZ",
    "ZF": "HMUZ",
    "ZL": "FHKNQUVZ",
    "ZM": "FHKNQUVZ",
    "ZN": "HMUZ",
    "ZS": "FHKNQUX",
    "ZT": "HMUZ",
    "ZW": "HKNUZ",
}

MONTH_ORDER = "FGHJKMNQUVXZ"
ROLL_RULE_MIN = 0.95
FORCED_ROLL = {"6B", "6C", "6J", "BRN", "CC", "CL", "CT", "EMD", "ES", "GAS",
               "GF", "HO", "KC", "LE", "NQ", "RB", "RTY", "SJB", "SXF",
               "YAP4", "YXT4", "ZC", "ZL", "ZM", "ZS", "ZW"}

RS_FORCED_ROLL = {"RS"}

HN_AUTO_ROLL = {"CGB", "GC", "HG", "LLG", "PA", "PL",
                "SI", "UB", "ZB", "ZF", "ZN", "ZT"}

LT_FORCED_ROLL = {"6S", "EUA", "SB"}

CS_AUTO_ROLL = {"BTC", "ETH", "FDAX9", "FESX9", "HSI", "LFT9", "NIY"}

CS_FORCED_ROLL = {"HE", "VX"}

STRIP_AUTO_ROLL = {"LEU9"}

STIR_PLUS1 = {"SO3", "SR3"}


LT_AUTO_ROLL = {"6A", "6E", "6M", "6N", "DX", "FGBL9", "FGBM9", "FGBS9", "NG"}

SYM = re.compile(r"^(?P<root>.+)-(?P<year>\d{4})(?P<code>[FGHJKMNQUVXZ])$")
MAPPING_NAME = "instrument_mapping.csv"


def find_mapping() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / MAPPING_NAME
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"{MAPPING_NAME} not found in any parent of {here} -- pass --mapping")


def wait_for_ndu(nd, tries: int = 4) -> bool:
    for attempt in range(tries):
        if nd.status():
            return True
        if attempt < tries - 1:
            time.sleep(2 * (attempt + 1))
    return False


def first_notice(nd, symbol: str) -> str | None:
    try:
        return nd.first_notice_date(symbol, datetimeformat="iso") or None
    except SystemExit as e:
        raise RuntimeError(
            f"norgatedata called sys.exit({e.code!r}) while fetching {symbol}: "
            f"10 attempts failed, so NDU stopped answering mid-build"
        ) from e


def build(instruments: list[str], year: int) -> tuple[pl.DataFrame, dict]:
    import norgatedata as nd

    rows = []
    notices: dict[str, dict[str, str | None]] = {}
    t0 = time.time()
    print(f"\nSTAGE 2  cycle table -- reading {len(instruments)} listings "
          f"from the vendor", flush=True)
    for n, inst in enumerate(instruments, 1):
        print(f"  [{n:>3}/{len(instruments)}] {inst:<8}"
              f"{time.time() - t0:>6.0f}s", flush=True)
        try:
            contracts = nd.futures_market_session_contracts(inst)
        except Exception:
            continue
        codes = set()
        n_contracts = 0
        for c in contracts:
            m = SYM.match(str(c))
            if not m:
                continue
            if int(m.group("year")) != year:
                continue
            codes.add(m.group("code"))
            n_contracts += 1
        if not codes:
            continue
        present = "".join(k for k in MONTH_ORDER if k in codes)
        if n_contracts != len(present):
            print(f"  [warn] {inst}: {n_contracts} contracts for "
                  f"{len(present)} distinct months in {year}")
        notices[inst] = {c: first_notice(nd, f"{inst}-{year}{c}") for c in present}
        n_notice = sum(v is not None for v in notices[inst].values())
        if 0 < n_notice < len(present):
            print(f"  [warn] {inst}: notice dates on {n_notice} of {len(present)} "
                  f"{year} contracts -- has_notice reduces that to false")
        row = {"instrument": inst, "codes": present, "per_year": len(present),
               "Dead_contracts": DEAD_CONTRACTS.get(inst, ""),
               "Active_contracts": ACTIVE_CONTRACTS.get(inst, ""),
               "is_deliverable": inst not in CASH_SETTLED,
                 "has_notice": n_notice == len(present)}
        rows.append(row)

    cols = ["instrument", "codes", "Dead_contracts", "Active_contracts",
            "per_year", "is_deliverable", "has_notice"]
    return pl.DataFrame(rows).select(cols).sort("instrument"), notices


_F32_SIG = 7
GAPS = "gaps"
BAR_COLS = ["Open", "High", "Low", "Close", "Volume", "Open Interest"]


def _widen(a, sig: int = _F32_SIG):
    import numpy as np
    x = np.asarray(a, dtype=np.float64)
    out = x.copy()
    m = np.isfinite(x) & (x != 0.0)
    if m.any():
        mag = np.floor(np.log10(np.abs(x[m])))
        f = 10.0 ** (sig - 1 - mag)
        out[m] = np.round(x[m] * f) / f
    return out


def fetch_bars(nd, symbol: str, tries: int = 3):
    import numpy as np
    rec = None
    for i in range(tries):
        try:
            rec = nd.price_timeseries(
                symbol, timeseriesformat="numpy-recarray",
                datetimeformat="datetime64ns",
                padding_setting=nd.PaddingType.NONE)
            break
        except SystemExit as e:
            raise RuntimeError(
                f"norgatedata called sys.exit({e.code!r}) fetching {symbol}: "
                f"10 attempts failed, so NDU stopped answering mid-run") from e
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2.0)
    if rec is None or len(rec) == 0:
        return None
    d = np.asarray(rec["Date"]).astype("datetime64[D]")
    ymd = [s.replace("-", "") for s in np.datetime_as_string(d, unit="D")]
    cols = {c: _widen(rec[c]) for c in BAR_COLS if c in rec.dtype.names}
    return ymd, cols


def append_bars(path: Path, fetched, *, dry: bool = False) -> dict:
    ymd, cols = fetched
    rep = {"added": 0, "gaps": 0, "conflicts": [], "created": False}
    cur = None
    have = {}
    if path.exists():
        cur = pl.read_csv(path, infer_schema_length=0)
        have = {r["Date"]: r for r in cur.iter_rows(named=True)}
    else:
        rep["created"] = True

    last = max(have) if have else ""
    new_rows = []
    for i, d in enumerate(ymd):
        if d in have:
            for c in BAR_COLS:
                if c not in cols:
                    continue
                prev = have[d].get(c)
                if prev in (None, ""):
                    continue
                if abs(float(prev) - float(cols[c][i])) > 1e-6:
                    rep["conflicts"].append(
                        (d, c, float(prev), float(cols[c][i])))
        else:
            new_rows.append({"Date": d,
                             **{c: float(cols[c][i]) for c in BAR_COLS
                                if c in cols}})
            if d < last:
                rep["gaps"] += 1
    rep["added"] = len(new_rows)
    if new_rows and not dry:
        add = pl.DataFrame(new_rows)
        if cur is not None:
            cur = cur.with_columns([pl.col(c).cast(pl.Float64, strict=False)
                                    for c in cur.columns if c != "Date"])
            out = pl.concat([cur, add.select(cur.columns)], how="vertical")
        else:
            out = add
        out.sort("Date").write_csv(path)
    return rep


def refresh_panel(nd, fc, instruments: list[str], *, full: bool = False,
                  dry: bool = False) -> dict:
    meta = {}
    if fc.NOTICE.exists():
        t = pl.read_csv(fc.NOTICE, infer_schema_length=0)
        meta = {r["symbol"]: (r["last_trade"] or "")
                for r in t.iter_rows(named=True)}

    tot = {"fetched": 0, "skipped": 0, "added": 0, "gaps": 0, "created": 0,
           "empty": 0, "conflicts": 0, "failed": 0}
    detail = {}
    hdr = (f"{'inst':<8}{'listed':>8}{'skip':>7}{'fetch':>7}{'new bars':>10}"
           f"{'new files':>11}{'conflict':>10}{'sec':>7}")
    print(f"\nSTAGE 1  panel refresh -> {fc.CONTRACTS}"
          f"{'   [DRY RUN]' if dry else ''}\n\n{hdr}\n" + "-" * len(hdr))

    for inst in instruments:
        t0 = time.time()
        d = fc.CONTRACTS / inst
        d.mkdir(parents=True, exist_ok=True)
        try:
            syms = [str(c) for c in nd.futures_market_session_contracts(inst)]
        except Exception as exc:
            print(f"{inst:<8}  LISTING FAILED  {type(exc).__name__}: {exc}")
            tot["failed"] += 1
            continue
        r = {k: 0 for k in ("skipped", "fetched", "added", "gaps",
                            "created", "empty", "conflicts")}
        for sym in sorted(syms):
            path = d / f"{sym}.csv"
            lt = meta.get(sym) or ""
            if not full and path.exists() and lt:
                cur = pl.read_csv(path, infer_schema_length=0)
                if cur.height:
                    last = cur.get_column("Date").to_list()[-1]
                    if last >= lt.replace("-", ""):
                        r["skipped"] += 1
                        continue
            try:
                got = fetch_bars(nd, sym)
            except Exception as exc:
                print(f"{inst:<8}  {sym} FETCH FAILED "
                      f"{type(exc).__name__}: {exc}")
                tot["failed"] += 1
                continue
            r["fetched"] += 1
            if got is None:
                r["empty"] += 1
                continue
            rep = append_bars(path, got, dry=dry)
            r["added"] += rep["added"]
            r["gaps"] += rep["gaps"]
            r["created"] += int(rep["created"])
            r["conflicts"] += len(rep["conflicts"])
            if rep["conflicts"]:
                detail.setdefault(inst, []).extend(
                    (sym,) + tuple(c) for c in rep["conflicts"][:3])
        for k in r:
            tot[k] += r[k]
        flag = "  <<<" if r["conflicts"] else ""
        print(f"{inst:<8}{len(syms):>8}{r['skipped']:>7}{r['fetched']:>7}"
              f"{r['added']:>10}{r['created']:>11}{r['conflicts']:>10}"
              f"{time.time() - t0:>7.0f}{flag}")
        sys.stdout.flush()

    print("-" * len(hdr))
    print("")
    print(f"{tot['added']:,} new bars, {tot['created']} new contract files, "
          f"{tot['fetched']:,} fetched, {tot['skipped']:,} skipped as settled")
    if tot["gaps"]:
        n_gap = tot[GAPS]
        print(f"  {n_gap:,} of those filled a HOLE -- a session the panel "
              f"was missing BEHIND its own last bar")
    if tot["empty"]:
        print(f"  {tot['empty']} contracts the vendor returned nothing for")
    if tot["failed"]:
        print(f"  {tot['failed']} failure(s)")
    if tot["conflicts"]:
        print(f"\n  {tot['conflicts']} CONFLICT(S) -- the vendor disagrees with "
              f"bars already on disk.  NOT applied.  Sample:")
        for inst, rows in list(detail.items())[:5]:
            for sym, dt, col, old, new in rows[:3]:
                print(f"     {inst:<6}{sym:<12}{dt}  {col:<14}"
                      f"disk {old:>14,.4f}   vendor {new:>14,.4f}")
    return tot


def refresh_metadata(nd, fc, instruments: list[str], *, dry: bool = False,
                     full: bool = False) -> dict:
    have: dict[str, dict] = {}
    if fc.NOTICE.exists():
        t = pl.read_csv(fc.NOTICE, infer_schema_length=0)
        have = {r["symbol"]: r for r in t.iter_rows(named=True)}

    today = _dt.date.today().isoformat()
    in_run = set(instruments)
    rows = [r for r in have.values() if r.get("instrument") not in in_run]
    carried_other = len(rows)

    queried = carried = added = changed = 0
    print(f"\nSTAGE 1b  contract metadata -> {fc.NOTICE}"
          f"{'   [DRY RUN]' if dry else ''}")
    for inst in instruments:
        try:
            syms = [str(c) for c in nd.futures_market_session_contracts(inst)]
        except Exception as exc:
            print(f"  [warn] {inst}: listing failed, its rows kept as they were"
                  f" -- {type(exc).__name__}: {exc}")
            rows.extend(r for r in have.values() if r.get("instrument") == inst)
            continue
        for sym in sorted(syms):
            m = SYM.match(sym)
            if not m:
                continue
            old = have.get(sym)
            lt_old = (old or {}).get("last_trade") or ""
            if old is not None and not full and lt_old and lt_old < today:
                rows.append(old)
                carried += 1
                continue

            fn = first_notice(nd, sym) or ""
            try:
                lt = nd.last_quoted_date(sym, datetimeformat="iso") or ""
            except SystemExit as e:
                raise RuntimeError(
                    f"norgatedata called sys.exit({e.code!r}) on {sym}") from e
            queried += 1
            fn, lt = (str(fn)[:10] if fn else ""), (str(lt)[:10] if lt else "")
            if old is None:
                added += 1
            elif (old.get("first_notice") or "") != fn \
                    or (old.get("last_trade") or "") != lt:
                changed += 1
                print(f"     {sym:<14}{'first_notice':<14}"
                      f"{(old.get('first_notice') or '-'):>12} -> {fn or '-'}"
                      f"   {'last_trade':<11}"
                      f"{(old.get('last_trade') or '-'):>12} -> {lt or '-'}")
            rows.append({"symbol": sym, "instrument": inst,
                         "year": int(m.group("year")),
                         "month_code": m.group("code"),
                         "month": MONTH_ORDER.index(m.group("code")) + 1,
                         "first_notice": fn, "last_trade": lt})

    print(f"  {len(rows):,} contracts on file   {queried:,} queried   "
          f"{carried:,} settled and carried   {carried_other:,} outside this run")
    print(f"  {added} newly listed   {changed} with a changed date")
    if not dry and rows:
        (pl.DataFrame(rows)
           .select(["symbol", "instrument", "year", "month_code", "month",
                    "first_notice", "last_trade"])
           .with_columns(pl.col("year").cast(pl.Int64),
                         pl.col("month").cast(pl.Int64))
           .sort("symbol")
           .write_csv(fc.NOTICE))
    return {"rows": len(rows), "queried": queried, "carried": carried,
            "added": added, "changed": changed}

def _roll_rules() -> dict[str, str]:
    out: dict[str, str] = {}
    for names, rule in ((HN_AUTO_ROLL, "auto_roll"),
                        (LT_AUTO_ROLL, "auto_roll"),
                        (CS_AUTO_ROLL, "auto_roll"),
                        (STRIP_AUTO_ROLL, "auto_roll"),
                        (FORCED_ROLL, "forced_roll"),
                        (STIR_PLUS1, "+1_auto_roll"),
                        (RS_FORCED_ROLL, "RS_forced_roll"),
                        (LT_FORCED_ROLL, "LT_forced_roll"),
                        (CS_FORCED_ROLL, "CS_forced_roll")):
        for i in names:
            assert i not in out, f"{i} is in two rule sets: {out[i]} and {rule}"
            out[i] = rule
    return out


FC = Path(__file__).resolve().parent / "Front_Contract" / "front_contract.py"
BOOK = Path(__file__).resolve().parents[1] / "2_Engine" / "trading_book.py"


def _load(path: Path, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rule_scores(df: pl.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    fc = _load(FC, "fc")
    todo = [(i, fc) for i in df.get_column("instrument").to_list()]

    tb = _load(BOOK, "tb_cache")
    as_of, _edge = fc.panel_as_of()

    t0 = time.time()
    print(f"\nSTAGE 2b  roll scores -- {len(todo)} worksheets "
          f"(cached: shared with stage 2)", flush=True)

    auto: dict[str, float] = {}
    forced: dict[str, float] = {}
    n_hit = 0
    for n, (inst, mod) in enumerate(todo, 1):
        print(f"  [{n:>3}/{len(todo)}] {inst:<8}{time.time() - t0:>6.0f}s",
              flush=True)
        try:
            w, hit = tb.cached_worksheet(mod, inst, "1900-01-01", "2100-01-01",
                                         as_of)
            n_hit += int(hit)
        except Exception as exc:
            print(f"  [warn] {inst}: scores not computed -- "
                  f"{type(exc).__name__}: {exc}")
            continue
        if not w.height:
            print(f"  [warn] {inst}: empty worksheet, scores left null")
            continue
        sm = mod.session_means(w).row(0, named=True)
        auto[inst] = round(sm["Auto_Best_V"], 4)
        forced[inst] = round(sm["f_r_h_Best_V"], 4)
    print(f"  worksheet cache: {n_hit} hit, {len(todo) - n_hit} built  "
          f"({time.time() - t0:.0f}s)", flush=True)
    return auto, forced


def verify(rows: list[dict], notices: dict, year: int) -> list[str]:
    bad = []
    for r in rows:
        if r["is_deliverable"]:
            continue
        for code, v in notices.get(r["instrument"], {}).items():
            if v:
                bad.append(f"{r['instrument']} marked cash-settled but "
                           f"{r['instrument']}-{year}{code} has a notice date {str(v)[:10]}")
                break
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default=None,
                    help=f"universe CSV; defaults to the {MAPPING_NAME} above this file")
    ap.add_argument("--year", type=int, default=2025,
                    help="count the cycle from this expiry year alone")
    ap.add_argument("--all-sessions", action="store_true",
                    help="every vendor session symbol, not just our universe")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent
                                         / "contract_cycles.csv"))
    ap.add_argument("--no-scores", action="store_true",
                    help="skip the Mean_Auto_Best_V pass (it rebuilds every worksheet)")
    ap.add_argument("--no-refresh", action="store_true",
                    help="skip stage 1 and read the panel as it stands on disk")
    ap.add_argument("--refresh-full", action="store_true",
                    help="stage 1 refetches settled contracts too, not just live ones")
    ap.add_argument("--refresh-only", action="store_true",
                    help="run stage 1 and stop, leaving contract_cycles.csv alone")
    ap.add_argument("--dry-run", action="store_true",
                    help="stage 1 reports what it would append and writes nothing")
    args = ap.parse_args()

    import norgatedata as nd
    from norgatedata import norgatehelper as H
    for name in ("User-Agent", "X-Requested-With"):
        v = H.session.headers.get(name)
        if v is not None and not str(v).isascii():
            H.session.headers[name] = str(v).encode("ascii", "ignore").decode()
    if not wait_for_ndu(nd):
        print("[ABORT] NDU is not answering after 4 attempts")
        return 2

    if args.all_sessions:
        instruments = sorted(nd.futures_market_session_symbols())
    else:
        mapping = Path(args.mapping) if args.mapping else find_mapping()
        with open(mapping, encoding="utf-8") as f:
            instruments = [r["norgate_code"] for r in csv.DictReader(f)]

    fc = _load(FC, "fc")
    if not args.no_refresh:
        refresh_metadata(nd, fc, instruments, dry=args.dry_run,
                         full=args.refresh_full)
        refresh_panel(nd, fc, instruments,
                      full=args.refresh_full, dry=args.dry_run)
    elif args.refresh_only:
        print("[ABORT] --refresh-only with --no-refresh does nothing")
        return 2

    print("")
    print("PANEL EDGE")
    as_of, edge = fc.panel_as_of(refresh=True)
    lagging = fc.report_edge(as_of, edge)
    if lagging:
        print(f"    [WARN] {len(lagging)} instrument(s) look stale rather than "
              f"merely ragged -- a feed that has stopped reads the same as one "
              f"that is early")
    if args.refresh_only:
        print("")
        print("--refresh-only: stopping before the cycle table")
        return 0

    df, notices = build(instruments, args.year)

    problems = verify(df.to_dicts(), notices, args.year)
    for msg in problems:
        print(f"  [FAIL] {msg}")
    if problems:
        print(f"\n{len(problems)} settlement classification(s) contradict the vendor")
        return 1

    out = Path(args.out)
    _prev = out.read_bytes() if out.is_file() else None
    df.write_csv(out)

    if not args.no_scores:
        try:
            scores, forced = rule_scores(df)
        except BaseException:
            if _prev is not None:
                out.write_bytes(_prev)
                print(f"\n  [RESTORED] scoring failed -- {out.name} put back to "
                      f"its previous COMPLETE contents rather than left "
                      f"truncated. Re-run to refresh it.")
            raise
        df = df.with_columns(
            pl.col("instrument").replace_strict(scores, default=None)
              .cast(pl.Float64).alias("Mean_Auto_Best_V"),
            pl.col("instrument").replace_strict(forced, default=None)
              .cast(pl.Float64).alias("Mean_Forced_Best_V"))
        df = df.with_columns(
            pl.col("instrument").replace_strict(_roll_rules(), default=None)
              .alias("Roll_Rule"))
        df = df.with_columns(
            pl.when(pl.col("Roll_Rule").is_not_null()
                    & ~pl.col("Roll_Rule").is_in(["auto_roll", "forced_roll"]))
              .then(pl.lit(True))
              .otherwise(None)
              .alias("Unique_Roll"))
        head = ["instrument", "last_date", "codes", "Dead_contracts",
                "Active_contracts", "per_year", "is_deliverable", "has_notice",
                "Mean_Auto_Best_V", "Mean_Forced_Best_V", "Roll_Rule",
                "Unique_Roll"]
        df = df.with_columns(
            pl.col("instrument").replace_strict(edge, default=None)
              .alias("last_date"))
        df = df.select(head + [c for c in df.columns if c not in head])
        df.write_csv(out)
        if scores:
            avg = sum(scores.values()) / len(scores)
            print("")
            favg = sum(forced.values()) / len(forced) if forced else 0.0
            print(f"Mean_Auto_Best_V written for {len(scores)} instruments"
                  f"  (mean {avg:.4f})")
            print(f"Mean_Forced_Best_V written for {len(forced)} instruments"
                  f"  (mean {favg:.4f})")
            for rule in sorted(set(_roll_rules().values())):
                n = df.filter(pl.col("Roll_Rule") == rule).height
                print(f"Roll_Rule {rule:12s} {n:3d} of {len(scores)}")
            n = df.filter(pl.col("Mean_Auto_Best_V").is_not_null()
                          & pl.col("Roll_Rule").is_null()).height
            print(f"Roll_Rule {'undecided':12s} {n:3d} of {len(scores)}")
        else:
            print("")
            print("Mean_Auto_Best_V: nothing scored")

    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print(df)
    print(f"\n{df.height} instruments -> {out}")
    n_yes = int(df["has_notice"].sum())
    print(f"\nnotice date on every {args.year} contract: {n_yes} instruments; "
          f"{df.height - n_yes} without "
          f"({df.filter(pl.col('is_deliverable') & ~pl.col('has_notice')).height} "
          f"of those deliverable)")
    counts = df.group_by("per_year").len().sort("per_year")
    print("\ncontracts per year, how many instruments:")
    for r in counts.iter_rows(named=True):
        print(f"  {r['per_year']:>2} per year   {r['len']:>3} instruments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

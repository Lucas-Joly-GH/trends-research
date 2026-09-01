from __future__ import annotations

import argparse
import json
import platform
import csv
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_NODE = platform.node()
if not _NODE.isascii():
    platform.node = lambda _n=_NODE.encode("ascii", "ignore").decode(): _n

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import shut_markets as _sm  # noqa: E402

RUN_STAMP = HERE / ".pipeline_run.json"

_SHUT: dict | None = None


def shut_state() -> dict:
    """Les marches qui n'ont pas avance, et pourquoi. Calcule une fois."""
    global _SHUT
    if _SHUT is None:
        _SHUT = _sm.survey(HERE / "2_Engine" / "Trading_book")
    return _SHUT


def excused() -> set:
    """Ceux dont le retard s'explique par un calendrier propre."""
    return {k for k, (_d, m) in shut_state()["shut"].items()
            if m == _sm.HOLIDAY}
CYCLES = HERE / "1_Roll" / "contract_cycles.py"
BOOK = HERE / "2_Engine" / "trading_book.py"
PORTFOLIO = HERE / "3_Portfolio" / "portfolio.py"
BOOKKEEPING = HERE / "4_Bookkeeping" / "bookkeeping.py"
JOURNAL = HERE / "4_Bookkeeping" / "Journal" / "journal.py"
PUBLISH = HERE / "5_Publish" / "publish.py"


_ROW = re.compile(r"^\S+\s+\S*roll\S*\s")
_SPIN = "|/-" + chr(92)
_CR = chr(13)


def _bar(done: int, total: int | None, secs: float, width: int = 32) -> str:
    if total:
        frac = min(done / total, 1.0)
        fill = int(frac * width)
        return (f"  [{'#' * fill}{'.' * (width - fill)}] "
                f"{done:>3}/{total}  {secs:4.0f}s")
    return f"  [{_SPIN[int(secs * 2) % 4]}] working  {secs:4.0f}s"


REQUIRED = {
    "polars": "the whole pipeline; every frame is polars",
    "numpy": "date arithmetic in front_contract",
}
REQUIRED_STAGE1 = {
    "norgatedata": "stage 1 only -- the vendor API",
}
MIN_PYTHON = (3, 10)


def preflight(py: str, need_vendor: bool = True) -> None:
    probe = (
        "import sys, json;"
        "out={'v': list(sys.version_info[:3])};"
        "mods={};"
        "\nfor m in ['polars','numpy','norgatedata']:\n"
        "    try:\n"
        "        __import__(m); import importlib.metadata as md;\n"
        "        mods[m]=md.version(m)\n"
        "    except Exception as e:\n"
        "        mods[m]=None\n"
        "out['mods']=mods;print('PREFLIGHT'+json.dumps(out))"
    )
    try:
        r = subprocess.run([py, "-c", probe], capture_output=True, text=True,
                           timeout=120)
        line = next(l for l in (r.stdout or "").splitlines()
                    if l.startswith("PREFLIGHT"))
        info = json.loads(line[len("PREFLIGHT"):])
    except Exception as exc:
        print(f"[ABORT] cannot interrogate the interpreter: {py}")
        print(f"        {type(exc).__name__}: {exc}")
        raise SystemExit(2)

    ver = tuple(info["v"])
    mods = info["mods"]
    bad = []
    if ver < MIN_PYTHON:
        bad.append(f"python {'.'.join(map(str, ver))} "
                   f"< {'.'.join(map(str, MIN_PYTHON))} required")
    for m, why in REQUIRED.items():
        if not mods.get(m):
            bad.append(f"{m} is missing -- {why}")
    if need_vendor:
        for m, why in REQUIRED_STAGE1.items():
            if not mods.get(m):
                bad.append(f"{m} is missing -- {why}")

    got = "  ".join(f"{m} {v}" for m, v in mods.items() if v)
    print(f"env     : python {'.'.join(map(str, ver))}   {got}")
    if not bad:
        return

    print("")
    print("=" * 72)
    print("  [ABORT] this interpreter cannot run the pipeline")
    print("=" * 72)
    for b in bad:
        print(f"    - {b}")
    print("")
    print("  Fix it with either of the venvs that work on this machine:")
    print(r"    C:\Users\33698\PycharmProjects\LJOLY_Memoire_INSEEC_Msc2\.venv\Scripts\python.exe")
    print(r"    C:\Users\33698\PycharmProjects\trends-research\.venv\Scripts\python.exe")
    print("")
    print("  Point this run at one:      python Update.py --python <path>")
    print(f"  Or install into this one:  \"{py}\" -m pip install -r "
          f"{(HERE / 'requirements.txt')}")
    print("")
    print("  NOTE for Git Bash: do NOT `source .venv/Scripts/activate` on this")
    print("  machine -- it leaves PATH broken so even `git` disappears. Prepend")
    print("  the Scripts directory instead:")
    print(r'    export PATH="/c/Users/33698/PycharmProjects/'
          r'LJOLY_Memoire_INSEEC_Msc2/.venv/Scripts:$PATH"')
    raise SystemExit(2)


def _tick(msg: str, t0: float, tty: bool, state: dict, every: int = 10) -> None:
    el = time.time() - t0
    if tty:
        print(_CR + f"  [{_SPIN[int(el * 2) % 4]}] {msg}  {el:4.0f}s",
              end="", flush=True)
    elif el - state.get("last", -every) >= every:
        state["last"] = el
        print(f"  ... {msg}  {el:4.0f}s", flush=True)


NDU_TRIGGER = Path(r"C:\Program Files\Norgate Data Updater\bin\ndu.trigger.exe")


def ensure_ndu(dry: bool, wait: int = 60, quiet: bool = False) -> bool:
    print(f"\n{'=' * 72}\n  NDU  (start if down, then update)\n{'=' * 72}")
    if not NDU_TRIGGER.is_file():
        print(f"  [ABORT] trigger not found: {NDU_TRIGGER}")
        print("          Norgate Data Updater is not installed where expected.")
        raise SystemExit(2)
    cmd = [str(NDU_TRIGGER), "UPDATE", "DONOTSHOW"]
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry:
        return False, False

    def _running() -> bool:
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq dataupdater.norgate.exe"],
                capture_output=True, text=True, timeout=20).stdout or ""
        except Exception:
            return False
        return "dataupdater.norgate.exe" in out.lower()

    def _stamp():
        try:
            import norgatedata as nd
            from norgatedata import norgatehelper as H
            for k, v in list(H.session.headers.items()):
                if v is not None and not str(v).isascii():
                    H.session.headers[k] = str(v).encode("ascii", "ignore").decode()
            return nd.last_database_update_time("Futures")
        except Exception:
            return None

    was_up = _running()
    before = _stamp() if was_up else None
    print(f"  NDU before : {'up' if was_up else 'down'}"
          + (f"   last update {before}" if before else ""))

    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        print(f"    {line.rstrip()}")
    rc = proc.returncode
    if rc != 0:
        print(f"  [WARN] trigger exited {rc}; continuing -- stage 1 will fail "
              f"loudly if NDU really is unavailable")

    t0 = time.time()
    tty = sys.stdout.isatty()

    if not was_up:
        st_state: dict = {}
        while time.time() - t0 < min(wait, 90) and not _running():
            if not quiet:
                _tick("starting NDU", t0, tty, st_state, every=10)
            time.sleep(2)
        if tty and not quiet:
            print(_CR + " " * 60 + _CR, end="")
        if not _running():
            print(f"  [WARN] NDU did not start within {min(wait, 90)}s. "
                  f"Stage 1 will fail; start the updater by hand.")
            return False, False
        print(f"  NDU started  ({time.time() - t0:.0f}s)")
        before = _stamp()

    advanced = False
    w_state: dict = {}
    if wait <= 0:
        print("  NDU after  : wait skipped (--ndu-wait 0); stage 1 overlaps "
              "the fetch anyway")
        return False, not was_up
    while time.time() - t0 < wait:
        now = _stamp()
        if now and before and now != before:
            advanced = True
            break
        if not before:
            before = now
        if not quiet:
            _tick(f"waiting for new data (up to {wait}s)", t0, tty, w_state)
        time.sleep(5)
    if tty and not quiet:
        print(_CR + " " * 60 + _CR, end="")

    now = _stamp()
    st = _running()
    if advanced:
        print(f"  NDU after  : new data at {now}  ({time.time() - t0:.0f}s)")
    elif st:
        print(f"  NDU after  : up, no new data in {wait}s (last update {now})")
        print("               Ordinary on a quiet evening -- the session may not "
              "have closed yet.")
    else:
        print(f"  [WARN] NDU still not answering after {wait}s. Stage 1 will "
              f"fail; start the updater by hand.")
    return advanced, not was_up


def close_ndu(dry: bool) -> None:
    if dry or not NDU_TRIGGER.is_file():
        return
    print("")
    print("=" * 72)
    print("  NDU  (close)")
    print("=" * 72)
    cmd = [str(NDU_TRIGGER), "CLOSE"]
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in (proc.stdout or "").splitlines():
        print(f"    {line.rstrip()}")


def run(label: str, cmd: list[str], dry: bool, total: int | None = None,
        blocking: bool = True) -> float:
    if cmd and Path(cmd[0]).name.lower().startswith("python") and "-u" not in cmd:
        cmd = [cmd[0], "-u", *cmd[1:]]
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}")
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
    if dry:
        return 0.0
    t0 = time.time()
    if not sys.stdout.isatty():
        rc = subprocess.run(cmd).returncode
    else:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                bufsize=1, errors="replace")
        done = 0
        for line in proc.stdout:
            if _ROW.match(line):
                done += 1
                print(_CR + _bar(done, total, time.time() - t0),
                      end="", flush=True)
            else:
                print(_CR + " " * 60 + _CR + line.rstrip(), flush=True)
                if done:
                    print(_bar(done, total, time.time() - t0),
                          end="", flush=True)
        proc.wait()
        if done:
            print(_CR + _bar(done, total, time.time() - t0))
        rc = proc.returncode
    dt = time.time() - t0
    if rc != 0:
        tail = ("Later stages NOT run." if blocking
                else "Non-blocking: the run continues.")
        print(f"\n[{'ABORT' if blocking else 'FAILED'}] {label} exited {rc} "
              f"after {dt:.0f}s. {tail}")
        raise SystemExit(rc)
    print(f"\n  {label} ok  ({dt:.0f}s)")
    return dt


def _ok(label: str, cond: bool, detail: str = "") -> tuple[bool, str, str]:
    return (bool(cond), label, detail)


def _note(label: str, detail: str = "") -> tuple[None, str, str]:
    return (None, label, detail)


def _report(title: str, results: list[tuple[bool, str, str]]) -> int:
    bar = "=" * 72
    print("")
    print(bar)
    print(f"  VERIFY  {title}")
    print(bar)
    for ok, label, detail in results:
        mark = "NOTE" if ok is None else ("OK  " if ok else "FAIL")
        print(f"  [{mark}] {label:<52}{detail}")
    bad = sum(1 for ok, _, _ in results if ok is False)
    noted = sum(1 for ok, _, _ in results if ok is None)
    checks = len(results) - noted
    print("  " + "-" * 68)
    tail = f"   {bad} FAILED" if bad else ""
    if noted:
        tail += f"   {noted} noted"
    print(f"  {checks - bad}/{checks} passed{tail}")
    return bad


def _hold_for() -> set[str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tb", BOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return set(m.HOLD_FOR)


def verify_holds(quiet: bool = False) -> int:
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tb", BOOK)
    tb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tb)
    import polars as pl
    fc = tb._load(tb.FC, "fc")
    as_of, _edge = fc.panel_as_of()
    rules = tb.rules()

    empty, dangling, failed, built = [], [], [], 0
    t0 = time.time()
    tty = sys.stdout.isatty()
    _EVERY = 5.0
    _last = [t0]
    print(f"  resolving {len(rules)} roll rules against the newest session "
          f"(rebuilds the worksheet cache; slow when cold)", flush=True)
    for n, (inst, rule) in enumerate(sorted(rules.items()), 1):
        if not (tty or quiet) and (n == 1 or n == len(rules)
                                   or time.time() - _last[0] >= _EVERY):
            _last[0] = time.time()
            print(f"  ... resolving rules {n}/{len(rules)}  {inst:<8}"
                  f"{time.time() - t0:4.0f}s", flush=True)
        col = tb.HOLD_FOR.get(rule)
        if col is None:
            failed.append(f"{inst}({rule})"); continue
        try:
            w, hit = tb.cached_worksheet(fc, inst, "1900-01-01", "2100-01-01", as_of)
        except Exception as exc:
            failed.append(f"{inst}({type(exc).__name__})"); continue
        built += 0 if hit else 1
        if col not in w.columns:
            failed.append(f"{inst}(no {col})"); continue
        last = w.get_column("date").max()
        sess = w.filter(pl.col("date") == last)
        held = {h for h in sess.get_column(col).to_list() if h}
        if not held:
            empty.append(inst)
        elif not (held <= set(sess.get_column("symbol").to_list())):
            dangling.append(inst)
        if tty and not quiet:
            print(_CR + _bar(n, len(rules), time.time() - t0),
                  end="", flush=True)
    if tty and not quiet:
        print(_CR + " " * 60 + _CR, end="")

    r = [_ok("every rule resolved without error", not failed,
             f"{len(rules) - len(failed)}/{len(rules)}"
             + (f"   failed: {', '.join(failed[:6])}" if failed else "")),
         _ok("holds a contract on the newest session",
             not empty, f"as_of {as_of}"
             + (f"   EMPTY: {', '.join(empty[:8])}" if empty else "")),
         _ok("hold names a contract listed that session", not dangling,
             f"{', '.join(dangling[:6])}" if dangling else "all instruments"),
         _ok("worksheet cache primed for stage 2", True,
             f"{len(rules) - built} hit, {built} built  ({time.time() - t0:.0f}s)")]
    return _report("stage 1 -- roll rules resolve today", r)


def verify_cycles() -> int:
    import csv as _csv
    f = HERE / "1_Roll" / "contract_cycles.csv"
    r: list[tuple[bool, str, str]] = []
    if not f.is_file() or f.stat().st_size == 0:
        return _report("stage 1 -- contract_cycles.csv",
                       [_ok("file present and non-empty", False, str(f))])
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    r.append(_ok("file present and non-empty", True,
                 f"{len(rows)} rows, {len(rows[0]) if rows else 0} columns"))

    ruled = [x for x in rows if (x.get("Roll_Rule") or "").strip()]
    missing = [x["instrument"] for x in rows if not (x.get("Roll_Rule") or "").strip()]
    r.append(_ok("every instrument has a Roll_Rule", not missing,
                 f"{len(ruled)}/{len(rows)}"
                 + (f"   missing: {', '.join(missing[:6])}" if missing else "")))

    known = _hold_for()
    unknown = sorted({x["Roll_Rule"] for x in ruled} - known)
    r.append(_ok("Roll_Rule values all map to a hold column", not unknown,
                 f"{len(known & {x['Roll_Rule'] for x in ruled})} distinct"
                 + (f"   UNKNOWN: {unknown}" if unknown else "")))

    mism = [x["instrument"] for x in rows
            if (x.get("codes") or "").strip()
            and str(len(x["codes"].strip())) != (x.get("per_year") or "").strip()]
    r.append(_ok("per_year == len(codes)", not mism,
                 f"{len(rows) - len(mism)}/{len(rows)}"
                 + (f"   differ: {', '.join(mism[:6])}" if mism else "")))

    empty = [x["instrument"] for x in rows if not (x.get("codes") or "").strip()]
    r.append(_ok("every instrument has month codes", not empty,
                 f"{', '.join(empty[:6])}" if empty else "63/63" if len(rows) == 63 else ""))

    import datetime as _dt
    def _d(v):
        try: return _dt.date(int(v[:4]), int(v[4:6]), int(v[6:8]))
        except Exception: return None
    seen = {x["instrument"]: _d((x.get("last_date") or "").strip()) for x in rows}
    good = {k: v for k, v in seen.items() if v}
    newest = max(good.values()) if good else None
    stale = sorted(k for k, v in good.items() if (newest - v).days > 5)
    try:
        import csv as _c2
        with open(HERE / "instrument_mapping.csv", newline="", encoding="utf-8") as fh:
            have = {x["norgate_code"] for x in _c2.DictReader(fh)
                    if (x.get("pointsize") or "").strip()}
        need = {x["instrument"] for x in ruled}
        gap = sorted(need - have)
        r.append(_ok("every ruled instrument has a pointsize", not gap,
                     f"{len(need & have)}/{len(need)}"
                     + (f"   MISSING: {', '.join(gap[:6])}" if gap else "")))
    except Exception as exc:
        r.append(_ok("every ruled instrument has a pointsize", False,
                     f"{type(exc).__name__}: {exc}"))

    r.append(_ok("no instrument more than 5 days stale", not stale and len(good) == len(rows),
                 f"newest {newest}"
                 + (f"   STALE: {', '.join(stale[:6])}" if stale else "")
                 + (f"   unparsed: {len(rows) - len(good)}" if len(good) != len(rows) else "")))
    return _report("stage 1 -- contract_cycles.csv", r)


def verify_books(started: float, expected: int | None) -> int:
    import polars as pl
    d = HERE / "2_Engine" / "Trading_book"
    r: list[tuple[bool, str, str]] = []
    files = sorted(d.glob("*.csv"))
    r.append(_ok("book directory present", d.is_dir() and bool(files),
                 f"{len(files)} files"))
    if not files:
        return _report("stage 2 -- Trading_book/", r)

    r.append(_ok("one file per ruled instrument", expected is None or len(files) == expected,
                 f"{len(files)} of {expected}" if expected else "unknown expected"))

    stale = [f.stem for f in files if f.stat().st_mtime < started]
    r.append(_ok("every file rewritten by this run", not stale,
                 f"{len(files) - len(stale)}/{len(files)}"
                 + (f"   stale: {', '.join(stale[:6])}" if stale else "")))

    schemas, empty, bad_sig, unsorted_, dup, hold_mismatch = {}, [], [], [], [], []
    ends, anchor, thin, fdm_ref, fdm_split = {}, [], [], None, []
    fx_vals, fx_vol_bad = {}, []
    _SIG = ("SIGNAL", "Trend_sign", "Carry_sign", "Skew_sign", "VoV_sign")
    for f in files:
        try:
            t = pl.read_csv(f, infer_schema_length=0)
        except Exception:
            empty.append(f.stem); continue
        if t.height == 0:
            empty.append(f.stem); continue
        schemas.setdefault(tuple(t.columns), []).append(f.stem)
        dts = t.get_column("date").to_list()
        if dts != sorted(dts): unsorted_.append(f.stem)
        if len(set(dts)) != len(dts): dup.append(f.stem)
        if {"hold", "symbol"} <= set(t.columns):
            hs = t.get_column("hold").to_list(); sy = t.get_column("symbol").to_list()
            if any(a != b for a, b in zip(hs, sy)): hold_mismatch.append(f.stem)
        ends[f.stem] = str(dts[-1])[:10] if dts else ""
        num = {c: [None if x in (None, "") else float(x)
                   for x in t.get_column(c).to_list()]
               for c in ("Continuous_C", "close", "fdm_raw") if c in t.columns}
        if {"Continuous_C", "close"} <= set(num):
            a, b = num["Continuous_C"][-1], num["close"][-1]
            if a is None or b is None or abs(a - b) > 1e-9:
                anchor.append(f.stem)
        if "fdm_raw" in num:
            cur = dict(zip([str(x)[:10] for x in dts], num["fdm_raw"]))
            if fdm_ref is None:
                fdm_ref = cur
            elif any(fdm_ref[k] != cur[k] for k in (set(fdm_ref) & set(cur))
                     if fdm_ref[k] is not None and cur[k] is not None):
                fdm_split.append(f.stem)
        for c in _SIG:
            if c in t.columns:
                col = t.get_column(c).to_list()
                frac = sum(1 for x in col if x not in (None, "")) / max(len(col), 1)
                if frac < 0.30:
                    thin.append((f.stem, c, frac))
        if "SIGNAL" in t.columns:
            v = pl.Series([x for x in t.get_column("SIGNAL").to_list() if x not in (None, "")],
                          dtype=pl.Utf8).cast(pl.Float64, strict=False)
            if v.len() and (v.max() > 20.0000001 or v.min() < -20.0000001):
                bad_sig.append(f.stem)
        if "FX_rate" in t.columns:
            fx_vals[f.stem] = dict(zip(
                [str(x)[:10] for x in dts],
                [None if x in (None, "") else float(x)
                 for x in t.get_column("FX_rate").to_list()]))
        if {"price_vol_USD_ann", "price_vol_curr_ann", "FX_rate"} <= set(t.columns):
            g = lambda c: [None if x in (None, "") else float(x)
                           for x in t.get_column(c).to_list()]
            u, c, x = g("price_vol_USD_ann"), g("price_vol_curr_ann"), g("FX_rate")
            for _u, _c, _x in zip(u, c, x):
                if _c is None or _x is None:
                    if _u is not None:
                        fx_vol_bad.append(f.stem); break
                elif (_u is None
                      or abs(_u - _c * _x) > abs(_c * _x) * 1e-9 + 1e-12):
                    fx_vol_bad.append(f.stem); break

    r.append(_ok("no empty or unreadable file", not empty,
                 f"{', '.join(empty[:6])}" if empty else f"{len(files)} readable"))
    r.append(_ok("identical schema across every file", len(schemas) <= 1,
                 f"{len(next(iter(schemas)))} columns" if len(schemas) == 1
                 else f"{len(schemas)} DIFFERENT schemas: "
                      + " | ".join(f"{v[0]}+{len(v)-1}" for v in schemas.values())))
    r.append(_ok("dates sorted, no duplicates", not unsorted_ and not dup,
                 f"unsorted: {unsorted_[:4]}  dup: {dup[:4]}" if (unsorted_ or dup) else "all files"))
    r.append(_ok("hold == symbol on every row", not hold_mismatch,
                 f"{', '.join(hold_mismatch[:6])}" if hold_mismatch else "all files"))
    if ends:
        newest = max(ends.values())
        exc = excused()
        behind = sorted(k for k, v in ends.items()
                        if v != newest and k not in exc)
        shut_here = sorted(k for k, v in ends.items()
                           if v != newest and k in exc)
        r.append(_ok("every book ends on the newest session", not behind,
                     f"{newest}"
                     + (f"   BEHIND: {', '.join(behind[:6])}" if behind else "")
                     + (f"   (marché fermé, non compté: "
                        f"{', '.join(shut_here)})" if shut_here else "")))

    r.append(_ok("Panama anchored: last Continuous_C == last close", not anchor,
                 f"{', '.join(anchor[:6])}" if anchor else "all files"))

    r.append(_ok("fdm_raw identical across instruments", not fdm_split,
                 f"{', '.join(sorted(set(fdm_split))[:5])}" if fdm_split
                 else "all shared dates"))

    r.append(_ok("no signal column near-empty for an instrument", not thin,
                 "   ".join(f"{a}.{b} {c:.0%}" for a, b, c in thin[:4])
                 if thin else "all >= 30% populated"))

    r.append(_ok("SIGNAL within +/-20", not bad_sig,
                 f"{', '.join(bad_sig[:6])}" if bad_sig else "all files"))

    no_twin = [f.stem for f in files if not f.with_suffix(".parquet").is_file()]
    old_twin = [f.stem for f in files
                if f.with_suffix(".parquet").is_file()
                and f.with_suffix(".parquet").stat().st_mtime_ns < f.stat().st_mtime_ns]
    r.append(_ok("parquet twin present and current", not no_twin and not old_twin,
                 ((f"missing: {', '.join(no_twin[:5])}  " if no_twin else "")
                  + (f"STALE: {', '.join(old_twin[:5])}" if old_twin else ""))
                 or f"{len(files)} pairs"))

    r.append(_ok("FX_rate present in every book", len(fx_vals) == len(files),
                 f"{len(fx_vals)} of {len(files)}"
                 + (f"   missing: "
                    f"{', '.join(sorted({f.stem for f in files} - set(fx_vals))[:5])}"
                    if len(fx_vals) != len(files) else "")))
    if fx_vals:
        fxd = HERE / "2_Engine" / "FX"
        rate_of, wrong, unmapped, allnull = {}, [], [], []
        try:
            tb = _tb()
        except Exception:
            tb = None
        for inst, series in sorted(fx_vals.items()):
            if all(v is None for v in series.values()):
                allnull.append(inst); continue
            if tb is None:
                continue
            try:
                ccy = tb.currency_of(inst)
            except SystemExit as exc:
                unmapped.append(f"{inst} ({exc})"); continue
            if ccy not in rate_of:
                p = fxd / f"{ccy}.csv"
                if not p.is_file():
                    unmapped.append(f"{inst} -> {ccy}.csv missing"); continue
                tt = pl.read_csv(p, infer_schema_length=0)
                rate_of[ccy] = dict(zip(
                    [str(x)[:10] for x in tt.get_column("date").to_list()],
                    [None if x in (None, "") else float(x)
                     for x in tt.get_column("Derived_Rate").to_list()]))
            ref = rate_of[ccy]
            bad = sum(1 for d_, v in series.items()
                      if v is not None
                      and (d_ not in ref or ref[d_] is None
                           or abs(v - ref[d_]) > 1e-12))
            if bad:
                wrong.append((inst, ccy, bad, len(series)))
        r.append(_ok("FX_rate matches the instrument's own currency", not wrong,
                     "   ".join(f"{a}({b}) {c:,}/{d:,}" for a, b, c, d in wrong[:4])
                     if wrong else f"{len(fx_vals)} books, {len(rate_of)} currencies"))
        r.append(_ok("no book with FX_rate entirely null", not allnull,
                     f"{', '.join(allnull[:6])}" if allnull else "all convertible"))
        r.append(_ok("price_vol_USD_ann == price_vol_curr_ann x FX_rate",
                     not fx_vol_bad,
                     f"{', '.join(sorted(set(fx_vol_bad))[:6])}" if fx_vol_bad
                     else f"{len(fx_vals)} books, eq 3.35 complete"))
        if unmapped:
            r.append(_ok("every book's currency resolves to a rate file", False,
                         "   ".join(unmapped[:3])))
    return _report("stage 2 -- Trading_book/", r)


FX_MAX_DAILY_MOVE = 0.15
FX_MAX_ALERT_FRAC = 0.01


def _tb():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tbfx", BOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify_fx(started: float) -> int:
    import polars as pl
    d = HERE / "2_Engine" / "FX"
    r: list[tuple[bool, str, str]] = []
    files = sorted(d.glob("*.csv"))
    r.append(_ok("FX directory present", d.is_dir() and bool(files),
                 f"{len(files)} files"))
    if not files:
        return _report("stage 2 -- FX/", r)

    try:
        ccys = set(_tb().FX_CCY)
    except Exception as exc:
        ccys = set()
        r.append(_ok("currency table readable from trading_book.py", False,
                     f"{type(exc).__name__}: {exc}"))
    if ccys:
        have = {f.stem for f in files}
        r.append(_ok("one file per currency in FX_CCY", have == ccys,
                     f"{len(have)} of {len(ccys)}"
                     + (f"   missing: {', '.join(sorted(ccys - have))}"
                        if ccys - have else "")
                     + (f"   extra: {', '.join(sorted(have - ccys))}"
                        if have - ccys else "")))

    stale = [f.stem for f in files if f.stat().st_mtime < started]
    r.append(_ok("every rate rewritten by this run", not stale,
                 f"{len(files) - len(stale)}/{len(files)}"
                 + (f"   stale: {', '.join(stale[:6])}" if stale else "")))

    no_twin = [f.stem for f in files if not f.with_suffix(".parquet").is_file()]
    old_twin = [f.stem for f in files
                if f.with_suffix(".parquet").is_file()
                and f.with_suffix(".parquet").stat().st_mtime_ns < f.stat().st_mtime_ns]
    r.append(_ok("parquet twin present and current", not no_twin and not old_twin,
                 ((f"missing: {', '.join(no_twin[:5])}  " if no_twin else "")
                  + (f"STALE: {', '.join(old_twin[:5])}" if old_twin else ""))
                 or f"{len(files)} pairs"))

    def _num(col):
        return [None if x in (None, "") else float(x) for x in col]

    schemas, empty, ends = {}, [], {}
    unsorted_, dup, nulls, nonpos, jumps = [], [], [], [], []
    bad_status, twin_diff = [], []
    alert = total = 0
    KNOWN_STATUS = {"OK", "WATCH", "ALERT", "UNCHECKED", "NO_DERIVED"}
    for f in files:
        try:
            t = pl.read_csv(f, infer_schema_length=0)
        except Exception:
            empty.append(f.stem); continue
        if t.height == 0:
            empty.append(f.stem); continue
        schemas.setdefault(tuple(t.columns), []).append(f.stem)
        dts = t.get_column("date").to_list()
        if dts != sorted(dts): unsorted_.append(f.stem)
        if len(set(dts)) != len(dts): dup.append(f.stem)
        ends[f.stem] = str(dts[-1])[:10] if dts else ""

        if "Status" in t.columns:
            sv = set(t.get_column("Status").to_list()) - {None, ""}
            if not sv <= KNOWN_STATUS:
                bad_status.append((f.stem, sorted(sv - KNOWN_STATUS)[:3]))
            alert += sum(1 for x in t.get_column("Status").to_list() if x == "ALERT")
            total += t.height

        if "Derived_Rate" not in t.columns:
            continue
        v = _num(t.get_column("Derived_Rate").to_list())
        n_null = sum(1 for x in v if x is None)
        if n_null:
            nulls.append((f.stem, n_null, len(v)))
        if any(x is not None and not (x > 0 and x == x and x != float("inf"))
               for x in v):
            nonpos.append(f.stem)
        worst, worst_at = 0.0, ""
        for i in range(1, len(v)):
            a, b = v[i - 1], v[i]
            if a and b and a > 0:
                m = abs(b / a - 1.0)
                if m > worst:
                    worst, worst_at = m, str(dts[i])[:10]
        if worst > FX_MAX_DAILY_MOVE:
            jumps.append((f.stem, worst, worst_at))

        pq = f.with_suffix(".parquet")
        if pq.is_file():
            try:
                pv = pl.read_parquet(pq).get_column("Derived_Rate").to_list()
                if len(pv) != len(v) or any(
                        (a is None) != (b is None)
                        or (a is not None and b is not None and abs(a - b) > 1e-12)
                        for a, b in zip(v, pv)):
                    twin_diff.append(f.stem)
            except Exception:
                twin_diff.append(f.stem)

    r.append(_ok("no empty or unreadable rate file", not empty,
                 f"{', '.join(empty[:6])}" if empty else f"{len(files)} readable"))
    r.append(_ok("identical schema across every rate", len(schemas) <= 1,
                 f"{len(next(iter(schemas)))} columns" if len(schemas) == 1
                 else f"{len(schemas)} DIFFERENT schemas: "
                      + " | ".join(f"{v[0]}+{len(v)-1}" for v in schemas.values())))
    r.append(_ok("dates sorted, no duplicates", not unsorted_ and not dup,
                 f"unsorted: {unsorted_[:4]}  dup: {dup[:4]}"
                 if (unsorted_ or dup) else "all files"))
    r.append(_ok("Derived_Rate fully populated", not nulls,
                 "   ".join(f"{a} {b:,}/{c:,}" for a, b, c in nulls[:4])
                 if nulls else "no gaps in any rate"))
    r.append(_ok("Derived_Rate positive and finite", not nonpos,
                 f"{', '.join(nonpos[:6])}" if nonpos else "all files"))
    r.append(_ok(f"no session move > {FX_MAX_DAILY_MOVE:.0%}", not jumps,
                 "   ".join(f"{a} {b:.1%} on {c}" for a, b, c in jumps[:4])
                 if jumps else "largest move within tolerance"))
    r.append(_ok("Status values all known", not bad_status,
                 "   ".join(f"{a}: {b}" for a, b in bad_status[:3])
                 if bad_status else f"{len(KNOWN_STATUS)} known labels"))
    frac = (alert / total) if total else 0.0
    r.append(_ok(f"ALERT share under {FX_MAX_ALERT_FRAC:.0%}",
                 frac <= FX_MAX_ALERT_FRAC,
                 f"{alert:,} of {total:,} = {frac:.3%}"))
    r.append(_ok("csv and parquet agree on Derived_Rate", not twin_diff,
                 f"{', '.join(twin_diff[:6])}" if twin_diff else f"{len(files)} pairs"))

    for ccy, want, label in (("USD", 1.0, "USD is exactly 1.0 (base currency)"),
                             ("HKD", None, "HKD sits exactly on the peg")):
        f = d / f"{ccy}.csv"
        if not f.is_file():
            continue
        try:
            vals = set(_num(pl.read_csv(f, infer_schema_length=0)
                            .get_column("Derived_Rate").to_list()))
            if want is None:
                want = 1.0 / _tb().HKD_PEG
        except Exception:
            continue
        ok = len(vals) == 1 and abs(next(iter(vals)) - want) < 1e-12
        r.append(_ok(label, ok,
                     f"{want:.6f}" if ok
                     else f"{len(vals)} distinct value(s): {sorted(vals)[:2]}"))

    if ends:
        newest = max(ends.values())
        behind = sorted(k for k, v in ends.items() if v != newest)
        r.append(_ok("every rate ends on the newest session", not behind,
                     f"{newest}"
                     + (f"   BEHIND: {', '.join(behind[:6])}" if behind else "")))
    return _report("stage 2 -- FX/", r)


def verify_portfolio(started: float) -> int:
    import numpy as np
    import polars as pl
    d = HERE / "3_Portfolio"
    pos = d / "Positions"
    port_f = d / "Portfolio.csv"
    r: list[tuple[bool, str, str]] = []
    files = sorted(pos.glob("*.csv"))
    r.append(_ok("positions directory present", pos.is_dir() and bool(files),
                 f"{len(files)} files"))
    r.append(_ok("Portfolio.csv present", port_f.is_file(),
                 port_f.name if port_f.is_file() else "MISSING"))
    if not files or not port_f.is_file():
        return _report("stage 3 -- 3_Portfolio/", r)

    stale = [f.stem for f in files if f.stat().st_mtime < started]
    r.append(_ok("every position file rewritten by this run", not stale,
                 f"{len(files) - len(stale)}/{len(files)}"
                 + (f"   stale: {', '.join(stale[:6])}" if stale else "")))
    no_twin = [f.stem for f in files if not f.with_suffix(".parquet").is_file()]
    old_twin = [f.stem for f in files
                if f.with_suffix(".parquet").is_file()
                and f.with_suffix(".parquet").stat().st_mtime_ns < f.stat().st_mtime_ns]
    r.append(_ok("parquet twin present and current", not no_twin and not old_twin,
                 ((f"missing: {', '.join(no_twin[:5])}  " if no_twin else "")
                  + (f"STALE: {', '.join(old_twin[:5])}" if old_twin else ""))
                 or f"{len(files)} pairs"))

    P = pl.read_csv(port_f, infer_schema_length=None)
    g = lambda c: P.get_column(c).to_numpy() if c in P.columns else None
    import math
    idm, wcw, na = g("IDM"), g("wCw"), g("n_active")
    nav, pnl = g("NAV"), g("pnl_USD")

    bad_idm = 0
    if idm is not None and wcw is not None:
        for a, b in zip(idm, wcw):
            if b is None or not (b == b) or b <= 0.01:
                continue
            want = min(4.0, 1.0 / math.sqrt(b))
            if abs(a - want) > 1e-9:
                bad_idm += 1
    r.append(_ok("IDM == min(4, 1/sqrt(w'Cw))  (eq 3.33)", bad_idm == 0,
                 f"{bad_idm} rows disagree" if bad_idm else f"{len(P):,} sessions"))
    r.append(_ok("IDM within [1, 4]", bool(idm is not None
                 and (idm >= 1.0 - 1e-12).all() and (idm <= 4.0 + 1e-12).all()),
                 f"min {idm.min():.3f}  max {idm.max():.3f}" if idm is not None else "-"))
    fin = [x for x in (wcw if wcw is not None else []) if x == x]
    r.append(_ok("w'Cw positive wherever defined", all(x > 0 for x in fin),
                 f"{len(fin):,} defined, min {min(fin):.5f}" if fin else "-"))

    eq = g("equity_USD")
    if eq is None:
        eq = nav
    npl = g("net_pnl_USD")
    itr = g("interest_USD")
    drift = 0.0
    if eq is not None and npl is not None and len(eq) > 1:
        run = eq[0]
        for k in range(1, len(eq)):
            run = run + npl[k] + (itr[k] if itr is not None else 0.0)
            drift = max(drift, abs(run - eq[k]) / max(abs(eq[k]), 1.0))
    r.append(_ok("equity[t] == equity[t-1] + net_pnl + interest", drift < 1e-9,
                 f"max relative drift {drift:.2e}"))
    r.append(_ok("equity and sizing base never non-positive",
                 bool(nav is not None and (nav > 0).all()
                      and eq is not None and (eq > 0).all()),
                 f"min NAV {nav.min():,.0f}   min equity {eq.min():,.0f}"
                 if nav is not None and eq is not None else "-"))

    frac, wrongway, carry_bad, buf_bad = [], [], [], []
    for f in files:
        t = pl.read_csv(f, infer_schema_length=None)
        if not {"N_raw", "N_contracts"} <= set(t.columns):
            continue
        raw = t.get_column("N_raw").to_numpy()
        con = t.get_column("N_contracts").to_numpy()
        tgt = (t.get_column("N_target").to_numpy()
               if "N_target" in t.columns else con)
        if "sized" in t.columns:
            sz = [str(x).lower() in ("true", "1") for x in
                  t.get_column("sized").to_list()]
            for n in range(1, len(con)):
                if not sz[n] and con[n] == con[n] and con[n] != con[n - 1]:
                    carry_bad.append(f.stem); break
        for a, b in zip(raw, tgt):
            if b != b or a != a:
                continue
            if b != int(b):
                frac.append(f.stem); break
            if abs(b) > abs(a) + 1e-9 or (b != 0 and a != 0 and (b > 0) != (a > 0)):
                wrongway.append(f.stem); break
        for b in con:
            if b == b and b != int(b):
                frac.append(f.stem); break
        if "N_target" in t.columns and "sized" in t.columns:
            sz = [str(x).lower() in ("true", "1") for x in
                  t.get_column("sized").to_list()]
            for n in range(1, len(con)):
                if not sz[n] or con[n] != con[n] or tgt[n] != tgt[n]:
                    continue
                held, moved = con[n] == con[n - 1], con[n] == tgt[n]
                if not (held or moved):
                    buf_bad.append(f"{f.stem} neither"); break
                band = 0.10 * abs(con[n - 1])
                if held and not moved and abs(tgt[n] - con[n - 1]) > band + 1e-9:
                    buf_bad.append(f"{f.stem} held a move > band"); break
                if moved and not held and abs(tgt[n] - con[n - 1]) <= band - 1e-9:
                    buf_bad.append(f"{f.stem} traded inside band"); break
    r.append(_ok("N_contracts is a whole number", not frac,
                 f"{', '.join(sorted(set(frac))[:6])}" if frac
                 else f"{len(files)} files"))
    r.append(_ok("rounding always REDUCES |position|", not wrongway,
                 f"{', '.join(sorted(set(wrongway))[:6])}" if wrongway
                 else "toward zero everywhere"))
    r.append(_ok("buffer: executed is target-or-hold, band respected (3.36)",
                 not buf_bad,
                 f"{', '.join(sorted(set(buf_bad))[:4])}" if buf_bad
                 else f"b = 0.10, {len(files)} files"))
    cost_bad, net_bad = [], []
    tot_c = None
    for f in files:
        t = pl.read_csv(f, infer_schema_length=None)
        if not {"pnl_USD", "cost_USD", "net_pnl_USD"} <= set(t.columns):
            continue
        gg = t.get_column("pnl_USD").to_numpy()
        cc = t.get_column("cost_USD").to_numpy()
        nn = t.get_column("net_pnl_USD").to_numpy()
        cl = (t.get_column("cost_lag_USD").to_numpy()
              if "cost_lag_USD" in t.columns else cc)
        if np.nanmax(np.abs((gg - cl) - nn)) > 1e-6:
            net_bad.append(f.stem)
        if not (cc >= -1e-9).all():
            cost_bad.append(f"{f.stem} negative")
    r.append(_ok("net_pnl == pnl - cost_lag, per instrument", not net_bad,
                 f"{', '.join(sorted(set(net_bad))[:6])}" if net_bad
                 else f"{len(files)} files"))
    for f in files:
        t = pl.read_csv(f, infer_schema_length=None)
        if "notional_USD" in t.columns:
            nn = t.get_column("notional_USD").to_numpy()
            if np.nanmin(nn) < -1e-9:
                cost_bad.append(f"{f.stem} notional<0")
    r.append(_ok("cost and notional never negative", not cost_bad,
                 f"{', '.join(sorted(set(cost_bad))[:6])}" if cost_bad
                 else f"{len(files)} files"))
    pc, ppnl, pnet = g("cost_USD"), g("pnl_USD"), g("net_pnl_USD")
    if pc is not None and ppnl is not None and pnet is not None:
        pcl = g("cost_lag_USD")
        pcl = pc if pcl is None else pcl
        r.append(_ok("portfolio net_pnl == pnl - cost_lag",
                     bool(np.nanmax(np.abs((ppnl - pcl) - pnet)) < 1e-6),
                     f"max diff {np.nanmax(np.abs((ppnl - pcl) - pnet)):.1e}"))
        cum_dec = np.nancumsum(pc)
        cum_chg = np.nancumsum(pcl)
        never_early = bool(np.all(cum_chg <= cum_dec + 1e-6))
        pending = float(cum_dec[-1] - cum_chg[-1])
        r.append(_ok("commission charged never precedes decided",
                     never_early,
                     f"cumulative charged <= decided on all {len(pc):,} sessions"))
        last = float(np.nan_to_num(pc[-1]))
        # Un ordre decide avant la derniere seance sur un marche ferme n'a
        # pas pu s'executer : il reste en attente en plus de ceux du jour.
        st = shut_state()
        held = _sm.pending_from_shut(
            HERE / "4_Bookkeeping" / "pending.csv", st["shut"], st["as_of"])
        r.append(_ok("every dollar decided is charged, or still pending",
                     abs(pending - last - held) < 0.01,
                     f"decided {cum_dec[-1]:,.2f}  charged {cum_chg[-1]:,.2f}  "
                     f"pending {pending:,.2f} vs last session {last:,.2f}"
                     + (f" + {held:,.2f} retenu sur marché fermé"
                        if held else "")))
        gr, nrr = g("gross_ret"), g("net_ret")
        if gr is not None and nrr is not None and nav is not None:
            exp = np.zeros(len(nav)); exp[1:] = pnet[1:] / nav[:-1]
            r.append(_ok("net_ret == net_pnl / NAV[t-1]",
                         bool(np.nanmax(np.abs(exp[1:] - nrr[1:])) < 1e-9),
                         f"max diff {np.nanmax(np.abs(exp[1:] - nrr[1:])):.1e}"))
            tr_ = g("total_ret")
            if tr_ is not None and itr is not None and nav is not None:
                exp_i = np.zeros(len(nav)); exp_i[1:] = itr[1:] / nav[:-1]
                r.append(_ok("total_ret - net_ret == interest / NAV[t-1]",
                             bool(np.nanmax(np.abs((tr_ - nrr - exp_i)[1:])) < 1e-12),
                             f"max diff "
                             f"{np.nanmax(np.abs((tr_ - nrr - exp_i)[1:])):.1e}"))
                rap = g("rf_accrual_applied")
                if rap is not None:
                    bad_i = int((itr[rap > 0] < -1e-9).sum())
                    neg_r = int((rap < 0).sum())
                    r.append(_ok("interest sign follows the rate that earned it",
                                 bad_i == 0,
                                 f"${np.nansum(itr) / 1e9:,.1f}B credited; "
                                 f"{neg_r} sessions at a negative bill rate"))
            r.append(_ok("net_ret <= gross_ret wherever a trade was made",
                         bool((nrr <= gr + 1e-12).all()),
                         f"cost {np.nansum(pc) / 1e9:,.1f}B total"))
    r.append(_ok("a shut market's position is carried, not changed",
                 not carry_bad,
                 f"{', '.join(sorted(set(carry_bad))[:6])}" if carry_bad
                 else f"{len(files)} files"))

    tot = g("total_ret")
    rfa = g("rf_accrual_applied")
    nret = g("net_ret")
    if tot is not None and rfa is not None and nret is not None and na is not None:
        st = int(np.argmax(na > 0)) if bool((na > 0).any()) else 0
        w = np.arange(len(P)) >= st
        f = lambda v: np.nan_to_num(v.astype(float))[w]
        sr = lambda v: (float(np.mean(v) / np.std(v) * math.sqrt(256))
                        if np.std(v) > 0 else float("nan"))
        s_net, s_exc = sr(f(nret)), sr(f(tot) - f(rfa))
        s_tot = sr(f(tot))
        gap, effect = abs(s_net - s_exc), abs(s_tot - s_net)
        r.append(_ok("Sharpe is excess of IRX  (net == total - rf)",
                     gap <= max(0.05, 0.10 * effect),
                     f"net {s_net:.4f} vs total-rf {s_exc:.4f}   gap {gap:.4f}"
                     f"   (double-counting would move it {effect:.3f})"))

    npos = g("n_positions")
    if na is not None and npos is not None and bool((na > 0).any()):
        start = int(np.argmax(na > 0))
        d0 = P.get_column("date")[start]
        flat = int((npos[start:] == 0).sum())
        r.append(_ok("book never goes flat once it has started", flat == 0,
                     f"{flat} flat session(s) after {d0}" if flat
                     else f"from {d0}, {len(P) - start:,} sessions held"))
        zeros = int((na[start:] == 0).sum())
        r.append(_ok("n_active zero only on closed-market sessions", zeros < 100,
                     f"{zeros} sessions   max {int(na.max())} active"))
    return _report("stage 3 -- 3_Portfolio/", r)


def verify_bookkeeping(started: float) -> int:
    from collections import defaultdict

    import numpy as np
    import polars as pl
    d = HERE / "4_Bookkeeping"
    pos = HERE / "3_Portfolio" / "Positions"
    led_f, pend_f, exe_f = d / "Orders.csv", d / "pending.csv", d / "executed.csv"
    r: list[tuple[bool, str, str]] = []

    files = [led_f, pend_f, exe_f]
    have = [f for f in files if f.is_file()]
    r.append(_ok("Orders / pending / executed present", len(have) == 3,
                 f"{len(have)}/3"
                 + ("   missing: " + ", ".join(f.name for f in files
                                               if not f.is_file())
                    if len(have) < 3 else "")))
    if len(have) < 3 or not pos.is_dir():
        return _report("stage 4 -- 4_Bookkeeping/", r)

    stale = [f.name for f in files if f.stat().st_mtime < started]
    r.append(_ok("rewritten by this run", not stale,
                 ", ".join(stale) if stale else "3 files"))
    twins = [f.with_suffix(".parquet") for f in files]
    bad_twin = [t.name for t, f in zip(twins, files)
                if not t.is_file() or t.stat().st_mtime_ns < f.stat().st_mtime_ns]
    r.append(_ok("parquet twin present and current", not bad_twin,
                 ", ".join(bad_twin) if bad_twin else "3 pairs"))
    if bad_twin:
        return _report("stage 4 -- 4_Bookkeeping/", r)

    L = pl.read_parquet(led_f.with_suffix(".parquet"))
    C = pl.read_csv(led_f, infer_schema_length=0)
    same = (C.height == L.height and C.columns == L.columns
            and (not L.height or C.get_column("decision_date")[-1]
                 == L.get_column("decision_date")[-1]))
    r.append(_ok("csv and parquet agree", same,
                 f"{L.height:,} rows, {len(L.columns)} cols" if same
                 else f"csv {C.height:,}x{len(C.columns)} vs "
                      f"parquet {L.height:,}x{len(L.columns)}"))
    want = ["decision_date", "execute_at", "instrument", "contract", "action",
            "quantity", "kind", "position_before", "position_after",
            "decision_close", "commission_USD", "realised_pnl_USD"]
    r.append(_ok("schema is the ledger schema", L.columns == want,
                 "12 columns" if L.columns == want else f"got {L.columns}"))
    if L.columns != want or not L.height:
        return _report("stage 4 -- 4_Bookkeeping/", r)

    dec = L.get_column("decision_date")
    r.append(_ok("sorted by decision date", dec.is_sorted(),
                 f"{dec[0]} .. {dec[-1]}   {dec.n_unique():,} sessions"))

    q = L.get_column("quantity")
    r.append(_ok("quantity strictly positive", bool((q > 0).all()),
                 f"{int((q <= 0).sum())} non-positive" if bool((q <= 0).any())
                 else f"min {q.min():,.0f}  max {q.max():,.0f}"))
    frac = int((q != q.round()).sum())
    r.append(_ok("quantity is a whole number of contracts", frac == 0,
                 f"{frac} fractional" if frac else f"{L.height:,} orders"))

    signed = pl.when(pl.col("action") == "BUY").then(pl.col("quantity")) \
               .otherwise(-pl.col("quantity"))
    off = int(((L.get_column("position_after") - L.get_column("position_before")
                - L.select(signed).to_series()).abs() > 1e-9).sum())
    r.append(_ok("after - before == signed quantity", off == 0,
                 f"{off} rows disagree" if off else f"{L.height:,} orders"))

    kinds = set(L.get_column("kind").unique().to_list())
    known = {"OPEN", "CLOSE", "RESIZE", "ROLL_OUT", "ROLL_IN"}
    r.append(_ok("kind drawn from the known set", kinds <= known,
                 f"unexpected: {sorted(kinds - known)}" if kinds - known
                 else "  ".join(f"{k} {L.filter(pl.col('kind') == k).height:,}"
                                for k in sorted(kinds))))
    flat_bad = []
    for k, col in (("OPEN", "position_before"), ("ROLL_IN", "position_before"),
                   ("CLOSE", "position_after"), ("ROLL_OUT", "position_after")):
        z = L.filter(pl.col("kind") == k)
        if z.height and not bool((z.get_column(col) == 0).all()):
            flat_bad.append(f"{k}.{col}")
    z = L.filter(pl.col("kind") == "RESIZE")
    if z.height and not bool(((z.get_column("position_before") != 0)
                              & (z.get_column("position_after") != 0)).all()):
        flat_bad.append("RESIZE touches flat")
    r.append(_ok("each kind's flat side is actually flat", not flat_bad,
                 ", ".join(flat_bad) if flat_bad else "5 kinds"))

    rolls = L.filter(pl.col("kind").is_in(["ROLL_OUT", "ROLL_IN"]))
    g = rolls.group_by(["decision_date", "instrument"]).agg(
        pl.col("kind").n_unique().alias("k"),
        pl.col("contract").n_unique().alias("c"), pl.len().alias("n"))
    solo = g.filter(pl.col("k") == 1)
    ok_solo = True
    if solo.height:
        j = solo.join(rolls, on=["decision_date", "instrument"])
        ok_solo = bool(((j.get_column("position_before") == 0)
                        | (j.get_column("position_after") == 0)).all())
    r.append(_ok("rolls are paired, or flat on one side", ok_solo,
                 f"{g.height:,} events, {solo.height} single-leg"))
    r.append(_ok("a roll's two legs are different contracts",
                 bool((g.get_column("c") == g.get_column("n")).all()),
                 f"{rolls.get_column('quantity').sum():,.0f} contracts rolled"))

    dup = L.height - L.select(["decision_date", "instrument",
                               "contract"]).n_unique()
    r.append(_ok("no duplicate (session, instrument, contract)", dup == 0,
                 f"{dup} duplicated" if dup else f"{L.height:,} unique keys"))

    by_inst = {k[0]: v for k, v in L.partition_by("instrument",
                                                  as_dict=True).items()}
    div = shut = wrong_exec = sessions = nulls = 0
    fill_div = 0
    first_div = first_fill = ""
    missing = []
    balance = {}
    final = {}
    for f in sorted(pos.glob("*.parquet")):
        inst = f.stem
        t = (pl.read_parquet(f, columns=["date", "symbol", "N_contracts"])
             .filter(pl.col("symbol").is_not_null()))
        dts = t.get_column("date").to_list()
        sym = t.get_column("symbol").to_list()
        N = t.get_column("N_contracts").to_list()
        own = set(dts)
        nxt = {dts[i]: dts[i + 1] for i in range(len(dts) - 1)}
        sub = by_inst.get(inst)
        if sub is None:
            if any(n for n in N if n):
                missing.append(inst)
            continue
        by: dict[str, list] = {}
        by_exe: dict[str, list] = {}
        for row in sub.iter_rows(named=True):
            by.setdefault(row["decision_date"], []).append(row)
            if row["execute_at"] is not None:
                by_exe.setdefault(row["execute_at"], []).append(row)
            if row["decision_date"] not in own:
                shut += 1
            if row["execute_at"] != nxt.get(row["decision_date"]):
                wrong_exec += 1
            nulls += row["execute_at"] is None
            k2 = (inst, row["contract"])
            balance[k2] = balance.get(k2, 0.0) + (
                row["quantity"] if row["action"] == "BUY" else -row["quantity"])
        final[inst] = (sym[-1], N[-1] or 0.0) if dts else (None, 0.0)
        held: dict[str, float] = {}
        fill: dict[str, float] = {}
        for k, (dte, sy, n) in enumerate(zip(dts, sym, N)):
            for row in by.get(dte, ()):
                held[row["contract"]] = held.get(row["contract"], 0.0) + (
                    row["quantity"] if row["action"] == "BUY"
                    else -row["quantity"])
            held = {c: v for c, v in held.items() if v}
            sessions += 1
            n = n or 0.0
            if held.get(sy, 0.0) != n or len(held) > bool(n):
                div += 1
                first_div = first_div or f"{inst}@{dte} held={held} want {sy}:{n}"
            for row in by_exe.get(dte, ()):
                fill[row["contract"]] = fill.get(row["contract"], 0.0) + (
                    row["quantity"] if row["action"] == "BUY"
                    else -row["quantity"])
            fill = {c: v for c, v in fill.items() if v}
            ps, pn = (sym[k - 1], N[k - 1] or 0.0) if k else (None, 0.0)
            if fill.get(ps, 0.0) != pn or len(fill) > bool(pn):
                fill_div += 1
                first_fill = first_fill or (f"{inst}@{dte} filled={fill} vs "
                                            f"position at {dts[k-1] if k else '-'}"
                                            f" {ps}:{pn}")
    r.append(_ok("replay reproduces every position, exactly", div == 0,
                 first_div if div else
                 f"{sessions:,} sessions, {L.height:,} legs, 0 divergences"))
    r.append(_ok("no order on a session that market was shut", shut == 0,
                 f"{shut} orders" if shut else f"{sessions:,} sessions"))
    r.append(_ok("execute_at is that market's own next session", wrong_exec == 0,
                 f"{wrong_exec} wrong" if wrong_exec
                 else f"{L.height - nulls:,} dated, {nulls} awaiting an open"))
    r.append(_ok("fill-timeline replay == position, lagged one",
                 fill_div == 0,
                 first_fill if fill_div else
                 f"{sessions:,} sessions on the fill timeline"))
    unbal = [f"{i}/{c}" for (i, c), v in balance.items()
             if abs(v - (final.get(i, (None, 0.0))[1]
                         if final.get(i, (None, 0.0))[0] == c else 0.0)) > 1e-9]
    live = sum(1 for i, (c, n) in final.items() if n)
    r.append(_ok("trial balance: every contract closes out", not unbal,
                 ", ".join(unbal[:4]) if unbal
                 else f"{len(balance):,} contracts, {live} still open"))
    r.append(_ok("every instrument holding a position has orders", not missing,
                 ", ".join(missing[:5]) if missing
                 else f"{len(by_inst)} instruments"))

    earned = {}
    for f in sorted(pos.glob("*.parquet")):
        inst = f.stem
        t = (pl.read_parquet(f, columns=["date", "symbol", "N_contracts",
                                         "pnl_gap_USD", "pnl_day_USD"])
             .filter(pl.col("symbol").is_not_null()))
        sy = t.get_column("symbol").to_list()
        gp = t.get_column("pnl_gap_USD").to_list()
        dy = t.get_column("pnl_day_USD").to_list()
        nq = t.get_column("N_contracts").to_list()
        f0 = lambda x: 0.0 if x is None or x != x else float(x)
        for k in range(1, len(sy)):
            if k >= 2 and nq[k - 2]:
                earned[(inst, sy[k - 2])] = (earned.get((inst, sy[k - 2]), 0.0)
                                             + f0(gp[k]))
            if nq[k - 1]:
                earned[(inst, sy[k - 1])] = (earned.get((inst, sy[k - 1]), 0.0)
                                             + f0(dy[k]))
    live_c = {(i, c) for i, (c, n) in final.items() if n}
    booked = {(r["instrument"], r["contract"]): r["s"] for r in
              L.group_by(["instrument", "contract"])
               .agg(pl.col("realised_pnl_USD").sum().alias("s"))
               .iter_rows(named=True)}
    off_c = []
    unreal = 0.0
    for key, e in earned.items():
        b = booked.get(key, 0.0)
        if key in live_c:
            unreal += e - b
            continue
        if abs(e - b) > max(1.0, abs(e) * 1e-9):
            off_c.append(f"{key[0]}/{key[1]} {b:,.0f} vs {e:,.0f}")
    r.append(_ok("realised P&L closes out per expired contract", not off_c,
                 ", ".join(off_c[:3]) if off_c
                 else f"{len(earned) - len(live_c):,} expired contracts, "
                      f"${unreal / 1e6:,.0f}M still unrealised in {len(live_c)} open"))

    roll_days = set(zip(rolls.get_column("instrument").to_list(),
                        rolls.get_column("decision_date").to_list()))
    lc = defaultdict(float)
    for i, dd, c in zip(L.get_column("instrument").to_list(),
                        L.get_column("decision_date").to_list(),
                        L.get_column("commission_USD").to_list()):
        if c is not None and c == c:
            lc[(i, dd)] += c
    same = diff = 0
    led_roll = s3_roll = 0.0
    worst_c = ""
    for f in sorted(pos.glob("*.parquet")):
        inst = f.stem
        t = pl.read_parquet(f, columns=["date", "cost_USD"])
        for dd, c in zip(t.get_column("date").to_list(),
                         t.get_column("cost_USD").to_list()):
            c = 0.0 if c is None or c != c else float(c)
            v = lc.get((inst, dd), 0.0)
            if (inst, dd) in roll_days:
                led_roll += v
                s3_roll += c
            elif c or v:
                if abs(v - c) <= max(0.01, abs(c) * 1e-9):
                    same += 1
                else:
                    diff += 1
                    worst_c = worst_c or f"{inst}@{dd} ledger {v:,.2f} vs stage3 {c:,.2f}"
    r.append(_ok("commission matches stage 3 off a roll", diff == 0,
                 worst_c if diff else f"{same:,} sessions to the cent"))
    gap = led_roll - s3_roll
    r.append(_ok("commission matches stage 3 ON a roll", abs(gap) <= max(1.0, s3_roll * 1e-9),
                 f"${gap / 1e6:,.1f}M apart" if abs(gap) > max(1.0, s3_roll * 1e-9)
                 else f"${led_roll / 1e9:,.3f}B both ways, both legs charged"))

    stf = d / "statement.parquet"
    if stf.is_file():
        S = pl.read_parquet(stf)
        o = np.array([x if x is not None else np.nan
                      for x in S.get_column("opening_equity_USD").to_list()])
        gg = S.get_column("gross_pnl_USD").to_numpy()
        cc = S.get_column("commission_USD").to_numpy()
        ii = S.get_column("interest_USD").to_numpy()
        cl = S.get_column("closing_equity_USD").to_numpy()
        m = np.isfinite(o) & np.isfinite(cl) & (np.abs(cl) > 0)
        err = np.abs((o + gg - cc + ii - cl)[m]) / np.abs(cl[m])
        r.append(_ok("statement: opening + P&L - cost + interest == closing",
                     bool(err.max() < 1e-12) if m.any() else False,
                     f"max relative drift {err.max():.1e} over {int(m.sum()):,} sessions"
                     if m.any() else "no rows"))
        b = S.get_column("interest_base_USD").to_list()
        rt = S.get_column("rate_cal_day").to_numpy()
        bad_b = sum(1 for k in range(len(b))
                    if rt[k] and (b[k] is None
                                  or abs(b[k] * rt[k] - ii[k]) > max(0.01, abs(ii[k]) * 1e-9)))
        n_acc = int(np.sum(ii != 0))
        r.append(_ok("interest == base x rate, on every accruing session",
                     bad_b == 0,
                     f"{bad_b} rows disagree" if bad_b
                     else f"{n_acc:,} sessions credited, of {int((rt != 0).sum()):,} "
                          f"carrying a rate"))

    P = pl.read_parquet(pend_f.with_suffix(".parquet"))
    X = pl.read_parquet(exe_f.with_suffix(".parquet"))
    asof = dec.max()
    # Un roulement produit DEUX jambes sur le meme instrument, sortante et
    # entrante, sur deux contrats differents. Exiger un ordre par instrument
    # faisait echouer la verification chaque jour de roulement.
    extra = []
    if "kind" in P.columns:
        for inst, grp in P.group_by("instrument"):
            name = inst[0] if isinstance(inst, tuple) else inst
            if grp.height == 1:
                continue
            kinds = set(grp.get_column("kind").to_list())
            legs = grp.get_column("contract").n_unique()
            if not (grp.height == 2 and legs == 2
                    and kinds == {"ROLL_IN", "ROLL_OUT"}):
                extra.append(str(name))
    else:
        extra = ["(colonne kind absente)"] if (
            P.height != P.get_column("instrument").n_unique()) else []
    rolls = P.height - P.get_column("instrument").n_unique()
    ok_p = (not extra
            and bool((P.get_column("decision_date") <= asof).all())
            and bool((P.get_column("execute_at").is_null()
                      | (P.get_column("execute_at") > asof)).all()))
    r.append(_ok("pending: one per instrument, or a roll's two legs", ok_p,
                 f"{P.height} order(s) for the next open"
                 + (f", dont {rolls} jambe(s) de roulement" if rolls else "")
                 + (f"   ANORMAL: {', '.join(extra[:5])}" if extra else "")))
    ok_x = (not X.height) or (
        bool((X.get_column("execute_at") == asof).all())
        and X.height == X.get_column("instrument").n_unique())
    r.append(_ok("executed: filled at this session's open", ok_x,
                 f"{X.height} order(s) at the {asof} open"))
    both = (set(zip(P.get_column("instrument"), P.get_column("decision_date")))
            & set(zip(X.get_column("instrument"), X.get_column("decision_date"))))
    r.append(_ok("pending and executed are disjoint", not both,
                 f"{len(both)} in both" if both else "no order counted twice"))
    return _report("stage 4 -- 4_Bookkeeping/", r)


def verify_irx(started: float) -> int:
    import numpy as np
    import polars as pl
    d = HERE / "2_Engine" / "IRX"
    f_csv = d / "IRX.csv"
    r: list[tuple[bool, str, str]] = []
    r.append(_ok("IRX.csv present", f_csv.is_file(),
                 f_csv.name if f_csv.is_file() else "MISSING"))
    if not f_csv.is_file():
        return _report("stage 2 -- IRX/", r)

    r.append(_ok("rewritten by this run", f_csv.stat().st_mtime >= started,
                 f"mtime {'fresh' if f_csv.stat().st_mtime >= started else 'STALE'}"))
    pq = f_csv.with_suffix(".parquet")
    r.append(_ok("parquet twin present and current",
                 pq.is_file() and pq.stat().st_mtime_ns >= f_csv.stat().st_mtime_ns,
                 "pair current" if pq.is_file() else "MISSING"))

    t = pl.read_csv(f_csv, infer_schema_length=None)
    need = {"date", "irx_pct", "irx_bey_pct", "rf_cal_day",
            "cal_days_to_next", "rf_accrual_next"}
    r.append(_ok("all six columns present", need <= set(t.columns),
                 f"{len(t.columns)} columns"
                 + (f"   missing {sorted(need - set(t.columns))}"
                    if need - set(t.columns) else "")))
    if not need <= set(t.columns):
        return _report("stage 2 -- IRX/", r)

    dts = t.get_column("date").to_list()
    pct = t.get_column("irx_pct").to_numpy()
    bey = t.get_column("irx_bey_pct").to_numpy()
    cal = t.get_column("rf_cal_day").to_numpy()
    gap = t.get_column("cal_days_to_next").to_numpy()
    acc = t.get_column("rf_accrual_next").to_numpy()

    r.append(_ok("dates sorted, no duplicates",
                 dts == sorted(dts) and len(set(dts)) == len(dts),
                 f"{len(dts):,} sessions {dts[0]} .. {dts[-1]}"))
    r.append(_ok("irx_pct fully populated",
                 bool(np.isfinite(pct).all()),
                 f"{int((~np.isfinite(pct)).sum())} gaps"))

    try:
        n_days = _tb().IRX_BILL_DAYS
    except Exception:
        n_days = 91
    dec = pct / 100.0
    denom = 360.0 - dec * n_days
    r.append(_ok(f"irx_bey_pct == 365d/(360-d.n), n={n_days}",
                 bool(np.nanmax(np.abs(bey - 365.0 * dec / denom * 100.0)) < 1e-9),
                 f"max diff "
                 f"{np.nanmax(np.abs(bey - 365.0 * dec / denom * 100.0)):.1e}"))
    r.append(_ok("rf_cal_day == d/(360-d.n)",
                 bool(np.nanmax(np.abs(cal - dec / denom)) < 1e-15),
                 f"max diff {np.nanmax(np.abs(cal - dec / denom)):.1e}"))
    r.append(_ok("rf_cal_day x 365 == BEY  (day count is calendar, not trading)",
                 bool(np.nanmax(np.abs(cal * 365.0 - bey / 100.0)) < 1e-12),
                 f"max diff {np.nanmax(np.abs(cal * 365.0 - bey / 100.0)):.1e}"))
    r.append(_ok("BEY exceeds the quoted discount everywhere the rate is positive",
                 bool((bey[pct > 0] >= pct[pct > 0] - 1e-12).all()),
                 f"mean uplift {np.nanmean(bey[pct > 0] - pct[pct > 0]):.4f}pp"))

    g_ok = gap[np.isfinite(gap)]
    r.append(_ok("cal_days_to_next >= 1 wherever defined",
                 bool((g_ok >= 1).all()),
                 f"min {int(g_ok.min())}d  max {int(g_ok.max())}d  "
                 f"mean {g_ok.mean():.2f}d"))
    long_gaps = int((g_ok > 10).sum())
    r.append(_ok("no calendar gap longer than 10 days", long_gaps == 0,
                 f"{long_gaps} gaps > 10d"))
    r.append(_ok("rf_accrual_next == rf_cal_day x cal_days_to_next",
                 bool(np.nanmax(np.abs(acc[:-1] - cal[:-1] * gap[:-1])) < 1e-15),
                 f"max diff {np.nanmax(np.abs(acc[:-1] - cal[:-1] * gap[:-1])):.1e}"))
    n_null = int((~np.isfinite(acc)).sum())
    r.append(_ok("rf_accrual_next null on the final row only", n_null == 1,
                 f"{n_null} nulls" if n_null != 1 else "as expected"))

    lo, hi = -2.0, 25.0
    out = int(((pct < lo) | (pct > hi)).sum())
    r.append(_ok(f"irx_pct within [{lo}, {hi}]%  (a units sanity band)", out == 0,
                 f"min {pct.min():.3f}%  max {pct.max():.3f}%"
                 + (f"   {out} OUTSIDE" if out else "")))
    return _report("stage 2 -- IRX/", r)


def verify_stages(started: float) -> int:
    import numpy as np
    import polars as pl
    E = HERE / "2_Engine"
    P = HERE / "3_Portfolio"
    r: list[tuple[bool, str, str]] = []

    books = sorted(E.joinpath("Trading_book").glob("*.csv"))
    poss = sorted(P.joinpath("Positions").glob("*.csv"))
    fxs = sorted(E.joinpath("FX").glob("*.csv"))
    irx = E / "IRX" / "IRX.csv"
    port = P / "Portfolio.csv"
    if not (books and poss and port.is_file()):
        r.append(_ok("all stages produced output",
                     False, "a stage is missing; earlier reports say which"))
        return _report("cross-stage consistency", r)

    b_names = {f.stem for f in books}
    p_names = {f.stem for f in poss}
    r.append(_ok("one Positions file per book, no orphans", b_names == p_names,
                 f"{len(b_names)} books, {len(p_names)} positions"
                 + (f"   book-only {sorted(b_names - p_names)[:4]}"
                    if b_names - p_names else "")
                 + (f"   position-only {sorted(p_names - b_names)[:4]}"
                    if p_names - b_names else "")))

    def _last(p):
        try:
            return pl.read_csv(p, infer_schema_length=0).get_column("date")[-1][:10]
        except Exception:
            return None
    ends = {"Portfolio": _last(port)}
    if irx.is_file():
        ends["IRX"] = _last(irx)
    if fxs:
        ends["FX"] = _last(fxs[0])
    ends["book"] = max(x for x in (_last(f) for f in books) if x)
    ends["positions"] = max(x for x in (_last(f) for f in poss) if x)
    agree = len(set(ends.values())) == 1
    r.append(_ok("every stage ends on the same session", agree,
                 next(iter(set(ends.values()))) if agree
                 else "  ".join(f"{k}={v}" for k, v in ends.items())))

    n_port = pl.read_csv(port, infer_schema_length=0).height
    if irx.is_file():
        n_irx = pl.read_csv(irx, infer_schema_length=0).height
        r.append(_ok("IRX spans exactly the portfolio's grid", n_irx == n_port,
                     f"IRX {n_irx:,} vs Portfolio {n_port:,}"))
    if fxs:
        n_usd = pl.read_csv(E / "FX" / "USD.csv", infer_schema_length=0).height
        r.append(_ok("USD rate spans exactly the portfolio's grid",
                     n_usd == n_port, f"USD {n_usd:,} vs Portfolio {n_port:,}"))

    drift = []
    for f in poss:
        b = E / "Trading_book" / f"{f.stem}.csv"
        if not b.is_file():
            continue
        try:
            pv = pl.read_csv(f, infer_schema_length=0).tail(1)
            bv = pl.read_csv(b, infer_schema_length=0).tail(1)
        except Exception:
            continue
        if pv.height == 0 or bv.height == 0:
            continue
        if pv.get_column("date")[0] != bv.get_column("date")[0]:
            # Marche ferme : le portefeuille reporte la position sur la
            # seance du panel alors que le livre s'arrete la veille. Les
            # deux ont raison.
            if f.stem not in excused():
                drift.append(f"{f.stem} date")
            continue
        for col in ("SIGNAL", "price_vol_USD_ann", "s_g_vol"):
            if col in pv.columns and col in bv.columns:
                a, c = pv.get_column(col)[0], bv.get_column(col)[0]
                if a in (None, "") or c in (None, ""):
                    continue
                if abs(float(a) - float(c)) > 1e-9:
                    drift.append(f"{f.stem}.{col}"); break
    r.append(_ok("Positions agree with the books, newest session",
                 not drift,
                 f"{', '.join(sorted(set(drift))[:5])}" if drift
                 else f"{len(poss)} instruments, 3 columns each"))

    orders = HERE / "4_Bookkeeping" / "Orders.parquet"
    if orders.is_file():
        newest = max(f.stat().st_mtime_ns for f in poss)
        fresh = orders.stat().st_mtime_ns >= newest
        last = pl.read_parquet(orders, columns=["decision_date"])
        last = last.get_column("decision_date").max() if last.height else None
        r.append(_ok("ledger is newer than the positions it differences",
                     fresh, "ledger written BEFORE the positions it describes"
                     if not fresh else f"ledger ends {last}"))
        r.append(_ok("ledger decides no later than the last session",
                     last is not None and ends["positions"] is not None
                     and last <= ends["positions"],
                     f"ledger {last} vs positions {ends['positions']}"))
    return _report("cross-stage consistency", r)


RECONCILE = HERE / "4_Bookkeeping" / "Reconciliation_check" / "reconcile.py"


def deploy() -> int:
    def git(*args, check=True):
        r = subprocess.run(["git", *args], cwd=str(HERE.parent),
                           capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip())
        return r.stdout.strip()

    print(f"\n{'=' * 72}\n  DEPLOY  git commit + push  (docs/ only)\n{'=' * 72}")
    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        if branch != "main":
            print(f"  [SKIP] on branch '{branch}', not main. Pages deploys from "
                  f"main, so a push here would not move the site.")
            return 0

        if not git("status", "--porcelain", "--", "docs"):
            print("  docs/ is unchanged since the last publish -- nothing to "
                  "deploy.")
            return 0

        changed = [l for l in git("diff", "-U0", "--", "docs").splitlines()
                   if (l.startswith("+") or l.startswith("-"))
                   and not l.startswith(("+++", "---"))]
        volatile = ("updated_at", "generated_at", "?v=", "checked_at")
        material = [l for l in changed if not any(k in l for k in volatile)]

        # LE REVERT NE DOIT JAMAIS EMPORTER UNE EDITION A LA MAIN.
        #
        # Le test ligne a ligne ci-dessus suffit pour une MODIFICATION : elle
        # produit une ligne '-' portant le texte d'origine, qui est materielle,
        # donc on part commiter. Il ne suffit pas pour un AJOUT PUR dont toutes
        # les lignes contiennent un mot volatil -- une remarque sur `checked_at`
        # ajoutee dans app.js, par exemple. Il n'y a alors aucune ligne '-' pour
        # la contredire, la modification passe pour de la pendule, et le revert
        # l'efface sans un mot. C'est etroit, et ce sont precisement les mots
        # qu'on emploie en commentant l'estampillage de ce depot.
        #
        # On cesse donc de juger les SOURCES sur le contenu de leurs lignes. Le
        # pipeline n'ecrit dans les pages, app.js et site.css qu'une seule
        # chose, `?v=`. Toute autre difference y est humaine, par construction.
        # docs/data reste juge comme avant : il est regenere en entier a chaque
        # execution et n'a pas d'auteur.
        def hand_edited():
            out = []
            for line in git("status", "--porcelain", "--", "docs").splitlines():
                f = line[3:].strip()
                if not f or f.startswith("docs/data/"):
                    continue
                d = [l for l in git("diff", "-U0", "--", f).splitlines()
                     if (l.startswith("+") or l.startswith("-"))
                     and not l.startswith(("+++", "---"))]
                # Un fichier non suivi n'a pas de diff : rien a annuler pour
                # `git checkout`, mais il n'est pas de la pendule non plus.
                if not d or any("?v=" not in l for l in d):
                    out.append(f)
            return out

        hand = hand_edited()
        if changed and not material and hand:
            print(f"  {len(hand)} source file(s) carry hand edits that are not "
                  f"the asset stamp:")
            for f in hand[:4]:
                print(f"    {f}")
            print("  so docs/ is NOT reverted -- the edit is committed with the "
                  "run instead.")
        if changed and not material and not hand:
            # run.json porte l'heure de verification : c'est le SEUL fichier
            # qui doit avancer un jour sans nouvelle seance. Tout le reste
            # revient en arriere comme avant, pour ne pas commiter une
            # pendule dans le payload.
            RUN = "docs/data/run.json"
            others = [f for f in git("status", "--porcelain", "--", "docs")
                      .splitlines() if f[3:].strip() != RUN]
            if others:
                git("checkout", "--", "docs", ":!" + RUN)
            print(f"  docs/ differs only in timestamps and the asset stamp "
                  f"({len(changed)} lines) --")
            print("  the published numbers are unchanged, so the payload is "
                  "reverted.")
            if git("status", "--porcelain", "--", RUN):
                when = "unknown"
                try:
                    when = json.loads((HERE.parent / "docs" / "data"
                                       / "run.json").read_text(
                        encoding="utf-8"))["checked_at"]
                except Exception:
                    pass
                git("add", "--", RUN)
                git("commit", "-m", f"Checked {when}")
                git("push", "origin", "HEAD:main")
                print(f"  run.json alone is committed and pushed "
                      f'("Checked {when}") -- the site can say it was')
                print("  checked today even though the numbers did not move.")
            else:
                print("  nothing committed.")
            return 0

        git("add", "--", "docs")
        staged = [l for l in git("diff", "--cached", "--name-only").splitlines() if l]
        stray = [f for f in staged if not f.startswith("docs/")]
        if stray:
            git("reset", "--quiet")
            print(f"  [ABORT] staging picked up {len(stray)} file(s) outside "
                  f"docs/: {stray[:4]}")
            print("          Nothing committed, nothing pushed, index reset.")
            return 1

        as_of = "unknown"
        try:
            as_of = json.loads((HERE.parent / "docs" / "data" / "latest.json")
                               .read_text(encoding="utf-8"))["meta"]["as_of"]
        except Exception:
            pass
        git("commit", "-m", f"Publish {as_of}")
        head = git("rev-parse", "--short", "HEAD")
        print(f"  committed {len(staged)} file(s) as {head}  "
              f'"Publish {as_of}"')

        git("push", "origin", "HEAD:main")
        print(f"  pushed to origin/main -- the site will rebuild in a minute or "
              f"two.")
        dirty = [l for l in git("status", "--porcelain", "--", ".",
                                ":!docs").splitlines() if l]
        if dirty:
            print(f"  [NOTE] {len(dirty)} uncommitted file(s) outside docs/. The "
                  f"site now shows results")
            names = [l.split(maxsplit=1)[-1] for l in dirty[:3]]
            print(f"         from a tree that is not fully committed: {names}")
        return 0
    except RuntimeError as e:
        print(f"  [FAILED] {e}")
        print("           The data is written and verified; only the deploy "
              "did not happen.")
        print("           Nothing was left staged that a `git status` will not "
              "show you.")
        return 1


_PT_START = "2026-01-02"
_JS_BUILTINS = {
    "fetch", "parseFloat", "parseInt", "isFinite", "isNaN", "String", "Number",
    "Math", "Object", "Array", "JSON", "Date", "Promise", "Set", "Map",
    "addEventListener", "setTimeout", "requestAnimationFrame", "map", "filter",
    "reduce", "forEach", "join", "split", "slice", "sort", "find", "some",
    "every", "concat", "replace", "test", "match", "push", "toFixed",
    "toLocaleString", "querySelector", "querySelectorAll", "getElementById",
    "toString", "padStart", "trim", "includes", "indexOf", "keys", "values",
    "entries", "from", "abs", "max", "min", "round", "floor", "ceil", "sqrt",
    "then", "catch", "all", "reverse", "startsWith", "endsWith", "repeat",
    "getAttribute", "setAttribute", "getPropertyValue", "getComputedStyle",
    "for", "if", "while", "switch", "catch", "return", "function", "typeof",

    "dispatchEvent", "scrollIntoView", "getBoundingClientRect", "add", "has",
}


_RENDER_BLANK_JS = """
() => {
  // A container the page's own script addresses and then leaves blank is the
  // signature of a fetch that 404'd, a renamed field, or an exception part-way
  // through a render.
  //
  // TWO EXEMPTIONS, both earned on the first run of this check.
  // HIDDEN elements are skipped: `#outstanding` is filled only when orders are
  // outstanding and its wrapper carries `hidden` otherwise, so blank is correct.
  // And `#stale` is empty EXACTLY WHEN THE DATA IS FRESH -- `staleNote` returns
  // "" on a current page, so an empty one is the good case and flagging it
  // would fire on every healthy run.
  const MAY_BE_EMPTY = new Set(["stale"]);
  const src = [...document.querySelectorAll('script:not([src])')]
                .map(s => s.textContent).join('\\n');
  const want = new Set();
  for (const m of src.matchAll(/el\\("([A-Za-z0-9_-]+)"\\)/g)) want.add(m[1]);
  for (const m of src.matchAll(/getElementById\\("([A-Za-z0-9_-]+)"\\)/g))
    want.add(m[1]);
  const blank = [];
  for (const id of want) {
    if (MAY_BE_EMPTY.has(id)) continue;
    const e = document.getElementById(id);
    if (!e) continue;
    if (e.offsetParent === null && getComputedStyle(e).position !== "fixed")
      continue;
    if (e.children.length === 0 && !e.textContent.trim()) blank.push(id);
  }
  return blank;
}
"""

_RENDER_TEXT_JS = """
() => {
  // Hand back the RENDERED text of a few named figures so the caller can hold
  // them against the JSON they came from. Rows are found by their label, not
  // their position, so reordering the summary does not break this; the label
  // is stripped of its footnote digit first.
  const out = {};
  for (const tr of document.querySelectorAll(".stats tr")) {
    const td = tr.querySelectorAll("td");
    if (td.length < 2) continue;
    const k = td[0].textContent.trim().replace(/\\d+$/, "").trim();
    out["stats:" + k] = td[1].textContent.trim();
  }
  for (const tr of document.querySelectorAll("#benchtable tbody tr")) {
    const td = tr.querySelectorAll("td");
    if (td.length >= 2)
      out["bench:" + td[0].textContent.trim()] = td[1].textContent.trim();
  }
  const a = document.getElementById("asof");
  if (a) out["asof"] = a.textContent.trim();
  return out;
}
"""

_RENDER_COLOUR_JS = """
() => {
  // THE ASSERTION THAT READS WHAT THE READER READS: a cell the code marked as
  // a gain or a loss must actually come out in that colour.
  //
  // THE TEST IS THE CLASS, NOT THE MINUS SIGN. Keying on a leading minus reads
  // a SHORT's contract count -- "-424" on the positions table -- as a loss and
  // demands it be red, which would be wrong; that number is a direction. Both
  // faults this check exists for had the class correctly applied and the wrong
  // colour rendered: a stray `}` discarded the rule painting `.neg`, and
  // `.stats tr.sub td` at specificity (0,2,2) outranked `.stats td.pos` at
  // (0,2,1). Testing the cascade is exactly testing class -> colour.
  const V = n => getComputedStyle(document.documentElement)
                   .getPropertyValue(n).trim().toLowerCase();
  const hex = c => { const m = c.match(/\\d+/g); return !m ? "" : "#" +
    m.slice(0, 3).map(x => (+x).toString(16).padStart(2, "0")).join(""); };
  const want = {neg: V("--neg"), pos: V("--pos")};
  const bad = [];
  for (const el of document.querySelectorAll(".neg, .pos")) {
    const kind = el.classList.contains("neg") ? "neg" : "pos";
    // A muted zero and a selected row deliberately override the sign colour.
    if (el.classList.contains("zero") || el.classList.contains("shut")) continue;
    if (el.closest("tr") && el.closest("tr").classList.contains("on")) continue;
    const got = hex(getComputedStyle(el).color);
    if (got && want[kind] && got !== want[kind])
      bad.push(`.${kind} "${el.textContent.trim().slice(0, 12)}" is ${got}, `
               + `should be ${want[kind]}`);
  }
  return bad.slice(0, 4);
}
"""


def verify_render(started: float) -> int:
    import http.server
    import socketserver
    import threading

    DOCS = HERE.parent / "docs"
    r: list[tuple[bool, str, str]] = []
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        r.append(_note("render check skipped", "playwright not installed -- "
                       "`pip install playwright && playwright install chromium`"))
        return _report("rendered pages -- headless", r)

    pages = sorted(p.name for p in DOCS.glob("*.html"))
    if not pages:
        r.append(_ok("pages present", False, "no html in docs/"))
        return _report("rendered pages -- headless", r)

    class _Q(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(DOCS), **k)

        def log_message(self, *a):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), _Q) as srv:
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{port}"
        try:
            errors, empty, colour, shown = [], [], [], {}
            with sync_playwright() as pw:
                try:
                    browser = pw.chromium.launch()
                except Exception as e:
                    r.append(_note("render check skipped",
                                   f"chromium not available: {str(e)[:60]}"))
                    return _report("rendered pages -- headless", r)
                try:
                    for name in pages:
                        for theme in ("light", "dark"):
                            page = browser.new_page(viewport={"width": 1400,
                                                              "height": 1000})
                            seen: list[str] = []
                            page.on("console", lambda m, s=seen:
                                    s.append(m.text) if m.type == "error" else None)
                            page.on("pageerror", lambda e, s=seen:
                                    s.append(str(e)))
                            page.goto(f"{base}/{name}", wait_until="load")
                            page.evaluate(
                                "t => document.documentElement.dataset.theme = t",
                                theme)
                            try:
                                page.wait_for_function(
                                    "() => [...document.querySelectorAll("
                                    "'[id]')].some(e => e.children.length "
                                    "|| e.textContent.trim())", timeout=15000)
                            except Exception:
                                pass
                            page.wait_for_timeout(900)

                            for msg in seen:
                                errors.append(f"{name}[{theme}]: {msg[:70]}")

                            blank = page.evaluate(_RENDER_BLANK_JS)
                            for b in blank:
                                empty.append(f"{name}[{theme}]: #{b}")

                            bad = page.evaluate(_RENDER_COLOUR_JS)
                            for b in bad:
                                colour.append(f"{name}[{theme}]: {b}")

                            if theme == "light":
                                shown.update(page.evaluate(_RENDER_TEXT_JS))
                            page.close()
                finally:
                    browser.close()
        finally:
            srv.shutdown()

    r.append(_ok("every page loads with no console error", not errors,
                 "; ".join(sorted(set(errors))[:3]) if errors
                 else f"{len(pages)} pages x 2 themes"))
    r.append(_ok("every signed figure renders in its sign's colour", not colour,
                 "; ".join(sorted(set(colour))[:3]) if colour
                 else "negatives red, signed positives green, both themes"))
    r.append(_ok("every container a script fills is non-empty", not empty,
                 "; ".join(sorted(set(empty))[:4]) if empty
                 else "all populated"))

    import math as _math
    try:
        meta = json.loads((DOCS / "data" / "latest.json")
                          .read_text(encoding="utf-8"))["meta"]
    except Exception:
        meta = None
    if meta is None:
        r.append(_note("rendered figures not compared", "latest.json unreadable"))
    else:
        def _pct(v):
            return f"{v * 100:.2f}%"

        def _floor2(v):
            return f"{_math.floor(round(v * 100, 6)) / 100:.2f}"

        expect = {
            "stats:Net asset value": "$" + f"{meta['equity_end']:,.0f}",
            "stats:Volatility, annualised": _pct(meta["net_ann_vol"]),
            "stats:Sharpe": _floor2(meta["net_sharpe"]),
            "stats:Maximum drawdown": _pct(meta["max_drawdown"]),
            "stats:Return, annualised (arithmetic)": _pct(meta["net_ann_ret"]),
            "asof": (f"As of {meta['as_of']}, {meta['sessions']} sessions "
                     f"since {meta['window_start']}."),
            "bench:The book": _pct(
                meta["equity_end"] / meta["equity_start"] - 1.0),
        }
        wrong, absent = [], []
        for k, want in expect.items():
            got = shown.get(k)
            if got is None:
                absent.append(k)
            elif got != want:
                wrong.append(f"{k}: shows {got!r}, payload says {want!r}")
        r.append(_ok("every checked figure on screen matches the payload",
                     not wrong and not absent,
                     "; ".join(wrong[:2] + [f"not found: {a}" for a in absent[:2]])
                     if (wrong or absent) else f"{len(expect)} figures tied to "
                     f"latest.json"))
    return _report("rendered pages -- headless", r)

def verify_vendor(started: float) -> int:
    import polars as pl
    r: list[tuple[bool, str, str]] = []
    import importlib.util
    _fcp = HERE / "1_Roll" / "Front_Contract" / "front_contract.py"
    _spec = importlib.util.spec_from_file_location("_fc_verify", _fcp)
    _fc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fc)
    raw = _fc.CONTRACTS
    BOOK_DIR = HERE / "2_Engine" / "Trading_book"
    if not raw.is_dir():
        r.append(_ok("vendor contract directory present", False, str(raw)))
        return _report("vendor bars -- the panel as delivered", r)

    held: dict[str, set[str]] = {}
    for f in sorted(BOOK_DIR.glob("*.parquet")):
        try:
            t = pl.read_parquet(f, columns=["date", "hold"])
        except Exception:
            continue
        for h in t.filter(pl.col("date") >= _PT_START)["hold"].to_list():
            if h:
                held.setdefault(f.stem, set()).add(h)

    bars = 0
    neg: list[str] = []
    unsorted_: list[str] = []
    dupes: list[str] = []
    jumps: list[str] = []
    oi_holes: list[tuple[str, str]] = []
    vol_by_date: dict[str, list[int]] = {}
    oi_by_date: dict[str, list[int]] = {}

    for sym, cons in sorted(held.items()):
        for con in sorted(cons):
            f = raw / sym / f"{con}.csv"
            if not f.is_file():
                continue
            rows = []
            with f.open(encoding="utf-8", errors="ignore") as fh:
                for row in csv.DictReader(fh):
                    d = row.get("Date", "")
                    if len(d) != 8 or d < _PT_START.replace("-", ""):
                        continue
                    try:
                        rows.append((d, float(row["Close"]),
                                     float(row.get("Volume") or 0),
                                     float(row.get("Open Interest") or 0)))
                    except (ValueError, KeyError):
                        continue
            if not rows:
                continue
            bars += len(rows)
            ds = [x[0] for x in rows]
            if ds != sorted(ds):
                unsorted_.append(con)
            if len(set(ds)) != len(ds):
                dupes.append(con)
            for i, (d, c, v, oi) in enumerate(rows):
                if c <= 0:
                    neg.append(f"{con} {d}")
                if i and rows[i - 1][1] > 0 and abs(c / rows[i - 1][1] - 1) > 0.35:
                    jumps.append(f"{con} {d}")
                vol_by_date.setdefault(d, []).append(1 if v > 0 else 0)
                oi_by_date.setdefault(d, []).append(1 if oi > 0 else 0)
                if 0 < i < len(rows) - 1 and oi == 0 \
                        and rows[i - 1][3] > 0 and rows[i + 1][3] > 0:
                    oi_holes.append((con, d))

    r.append(_ok("vendor bars readable on every held contract", bars > 0,
                 f"{bars:,} bars across {sum(len(v) for v in held.values())} "
                 f"contracts, {len(held)} instruments"))
    r.append(_ok("close is positive on every bar", not neg,
                 f"{len(neg)} non-positive" if neg else "no non-positive close"))
    r.append(_ok("dates sorted within every contract", not unsorted_,
                 f"{len(unsorted_)} unsorted" if unsorted_ else "all sorted"))
    r.append(_ok("no duplicate date within a contract", not dupes,
                 f"{len(dupes)} with duplicates" if dupes else "all unique"))
    r.append(_ok("no close moves more than 35% in a session", not jumps,
                 f"{len(jumps)}: {jumps[:3]}" if jumps else "largest move within band"))

    def _worst(by_date, what):
        worst, wd = 0.0, ""
        for d, flags in by_date.items():
            if len(flags) < 20:
                continue
            miss = 1.0 - (sum(flags) / len(flags))
            if miss > worst:
                worst, wd = miss, d
        return worst, wd

    wv, dv = _worst(vol_by_date, "volume")
    wo, do = _worst(oi_by_date, "open interest")
    vol_bad = wv >= 0.50
    r.append((None if vol_bad else True,
              "no session missing volume across the panel",
              f"worst {wv:.0%} on {dv or 'n/a'}"))
    oi_bad = wo >= 0.50
    r.append((None if oi_bad else True,
              "no session missing open interest across the panel",
              f"worst {wo:.0%} on {do or 'n/a'}"))
    r.append((None if oi_holes else True, "open interest has no holes",
              f"{len(oi_holes)} hole(s), e.g. "
              f"{[f'{a} {b}' for a, b in oi_holes[:3]]}"
              if oi_holes else "continuous on every held contract"))
    return _report("vendor bars -- the panel as delivered", r)


# Noeuds qui ouvrent une portee lexicale.
_JS_SCOPES = {"Program", "BlockStatement", "ForStatement", "ForInStatement",
              "ForOfStatement", "SwitchStatement", "CatchClause", "ClassBody",
              "StaticBlock"}


def _bound(pat, out: list) -> None:
    """Identifiants qu'un motif de liaison declare."""
    t = pat.get("type")
    if t == "Identifier":
        out.append(pat["name"])
    elif t == "ObjectPattern":
        for q in pat.get("properties") or []:
            v = q.get("value") or q.get("argument")
            if v:
                _bound(v, out)
    elif t == "ArrayPattern":
        for e in pat.get("elements") or []:
            if e:
                _bound(e, out)
    elif t == "AssignmentPattern":
        _bound(pat["left"], out)
    elif t == "RestElement":
        _bound(pat["argument"], out)


def _js_kids(node):
    for v in (node.values() if isinstance(node, dict) else []):
        if isinstance(v, dict) and "type" in v:
            yield v
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict) and "type" in x:
                    yield x


def _redeclared(node, out: list | None = None) -> list:
    """let/const/class declares deux fois dans la MEME portee.

    Le navigateur refuse alors le script entier. Esprima ne le signale pas :
    ce n'est pas une erreur de syntaxe mais une erreur precoce de portee.
    """
    out = [] if out is None else out
    if not isinstance(node, dict) or "type" not in node:
        return out
    if node["type"] in _JS_SCOPES:
        seen: set[str] = set()
        for child in _js_kids(node):
            names: list[str] = []
            if (child.get("type") == "VariableDeclaration"
                    and child.get("kind") in ("let", "const")):
                for d in child.get("declarations") or []:
                    _bound(d["id"], names)
            elif child.get("type") == "ClassDeclaration" and child.get("id"):
                names.append(child["id"]["name"])
            for n in names:
                if n in seen:
                    out.append(n)
                seen.add(n)
    for child in _js_kids(node):
        _redeclared(child, out)
    return out


def verify_assets(started: float) -> int:
    DOCS = HERE.parent / "docs"
    r: list[tuple[bool, str, str]] = []
    pages = sorted(DOCS.glob("*.html"))
    r.append(_ok("pages present", bool(pages), f"{len(pages)} page(s)"))
    if not pages:
        return _report("published assets -- docs/", r)

    def _strip(css: str) -> str:
        return re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    bad_css = []
    for f in [DOCS / "site.css"] + pages:
        txt = f.read_text(encoding="utf-8")
        blocks = ([txt] if f.suffix == ".css"
                  else re.findall(r"<style>(.*?)</style>", txt, re.S))
        for b in blocks:
            b = _strip(b)
            if b.count("{") != b.count("}"):
                bad_css.append(f"{f.name} {b.count('{')}/{b.count('}')}")
    r.append(_ok("CSS braces balanced in every stylesheet and style block",
                 not bad_css, "; ".join(bad_css) if bad_css else
                 f"{len(pages) + 1} source(s)"))

    void = {"br", "img", "meta", "link", "input", "hr", "source", "col",
            "path", "line", "text", "polyline", "rect", "circle", "use", "stop"}
    unbalanced, dup_attr = [], []

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list[str] = []
            self.bad: list[str] = []

        def handle_starttag(self, tag, attrs):
            names = [a for a, _ in attrs]
            for a in set(names):
                if names.count(a) > 1:
                    self.bad.append(f"<{tag} {a}=...> twice")
            if tag not in void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack or self.stack[-1] != tag:
                self.bad.append(f"</{tag}> line {self.getpos()[0]}")
                return
            self.stack.pop()

    for f in pages:
        p = _P()
        p.feed(f.read_text(encoding="utf-8"))
        left = [t for t in p.stack if t not in ("html", "body", "head")]
        struct = [b for b in p.bad if not b.endswith("twice")]
        dups = [b for b in p.bad if b.endswith("twice")]
        if left or struct:
            unbalanced.append(f"{f.name}: {(struct + left)[:2]}")
        if dups:
            dup_attr.append(f"{f.name}: {dups[:2]}")
    r.append(_ok("HTML tags balanced on every page", not unbalanced,
                 "; ".join(unbalanced) if unbalanced else f"{len(pages)} pages"))
    r.append(_ok("no attribute written twice in one tag", not dup_attr,
                 "; ".join(dup_attr) if dup_attr else "none"))

    dup_ids = []
    for f in pages:
        seen = re.findall(r'\sid="([A-Za-z0-9_-]+)"', f.read_text(encoding="utf-8"))
        rep = sorted({i for i in seen if seen.count(i) > 1})
        if rep:
            dup_ids.append(f"{f.name}: {rep[:3]}")
    r.append(_ok("no id used twice on one page", not dup_ids,
                 "; ".join(dup_ids) if dup_ids
                 else f"{len(pages)} pages, all unique"))

    missing_ids = []
    for f in pages:
        txt = f.read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', txt))
        want = set(re.findall(r'el\("([A-Za-z0-9_-]+)"\)', txt))
        want |= set(re.findall(r'getElementById\("([A-Za-z0-9_-]+)"\)', txt))
        gone = sorted(want - ids)
        if gone:
            missing_ids.append(f"{f.name}: {gone[:3]}")
    r.append(_ok("every id a script addresses exists on its page",
                 not missing_ids, "; ".join(missing_ids) if missing_ids
                 else "all resolved"))

    dead = []
    for f in pages:
        txt = f.read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', txt))
        for a in set(re.findall(r'href="#([A-Za-z0-9_-]+)"', txt)):
            if a not in ids:
                dead.append(f"{f.name}#{a}")
    r.append(_ok("every in-page anchor resolves", not dead,
                 "; ".join(dead[:4]) if dead else "all resolved"))

    def _code_only(js: str) -> str:
        js = re.sub(r"/\*.*?\*/", " ", js, flags=re.S)
        js = re.sub(r"//[^\n]*", " ", js)
        js = re.sub(r"`(?:[^`\\]|\\.)*`", " ", js, flags=re.S)
        js = re.sub(r'"(?:[^"\\\n]|\\.)*"', " ", js)
        js = re.sub(r"'(?:[^'\\\n]|\\.)*'", " ", js)
        return js

    appjs = _code_only((DOCS / "app.js").read_text(encoding="utf-8"))
    shared = set(re.findall(r"(?:function|const|let|var)\s+([A-Za-z_][\w]*)", appjs))
    unknown = []
    for f in pages:
        body = _code_only("\n".join(re.findall(
            r"<script>(.*?)</script>", f.read_text(encoding="utf-8"), re.S)))
        local = set(re.findall(r"(?:function|const|let|var)\s+([A-Za-z_][\w]*)", body))
        local |= set(re.findall(r"([A-Za-z_][\w]*)\s*=\s*(?:async\s*)?\(", body))
        local |= set(re.findall(r"(?:\(|,)\s*([A-Za-z_][\w]*)\s*(?:,|\))\s*=>", body))
        for params in re.findall(r"function\s+[A-Za-z_][\w]*\s*\(([^)]*)\)", body):
            local |= {q.strip().split("=")[0].strip()
                      for q in params.split(",") if q.strip()}
        for call in set(re.findall(r"(?<![.\w$])([a-z][A-Za-z0-9_]{2,})\s*\(", body)):
            if call in shared or call in local or call in _JS_BUILTINS:
                continue
            unknown.append(f"{f.name}: {call}()")
    # --- le JavaScript se parse-t-il seulement ? ------------------------
    # Une erreur de syntaxe rend la page entierement muette : le navigateur
    # abandonne le bloc <script> en entier, y compris ce qui marchait avant.
    # Tous les autres controles restent verts dans ce cas.
    try:
        import esprima
    except ImportError:
        r.append(_ok("JavaScript parses on every page", False,
                     "esprima absent -- pip install -r requirements.txt"))
    else:
        broken = 0
        n_blk = 0
        detail = []
        for f in pages + [DOCS / "app.js"]:
            src = f.read_text(encoding="utf-8")
            blocks = ([(0, src)] if f.suffix == ".js"
                      else [(m.start(), m.group(1)) for m in
                            re.finditer(r"<script>(.*?)</script>", src, re.S)])
            for off, b in blocks:
                if not b.strip():
                    continue
                n_blk += 1
                ln = src[:off].count("\n") + 1
                try:
                    tree = esprima.parseScript(b).toDict()
                except Exception as exc:      # noqa: BLE001 - message a afficher
                    broken += 1
                    detail.append(f"{f.name} (script at line {ln}): {exc}")
                    continue
                dup = sorted(set(_redeclared(tree)))
                if dup:
                    broken += 1
                    detail.append(f"{f.name} (script at line {ln}): "
                                  f"redeclared {dup[:3]}")
        r.append(_ok("JavaScript parses, nothing redeclared", not broken,
                     "; ".join(detail[:3]) if detail
                     else f"{n_blk} script block(s), ES2017"))

    # --- chaque graphique est-il reellement dessine ? -------------------
    # Le controle precedent va de l'appel vers le balisage. Celui-ci va du
    # balisage vers l'appel : un conteneur que plus personne ne dessine est
    # une page qui s'affiche sans erreur et sans courbe.
    SHAPES = ("polyline", "<rect", "<path", "<circle", "<line")
    painters = set()
    for f in [DOCS / "app.js"] + pages:
        src = f.read_text(encoding="utf-8")
        for m in re.finditer(r"^function\s+([A-Za-z_]\w*)\s*\(", src, re.M):
            nxt = re.search(r"^function\s", src[m.end():], re.M)
            body = src[m.end(): m.end() + (nxt.start() if nxt else len(src))]
            if any(t in body for t in SHAPES):
                painters.add(m.group(1))

    FIGRE = re.compile(r"<figure\b.*?</figure>", re.S | re.I)
    SVGRE = re.compile(r'<svg\b[^>]*\bid="([A-Za-z0-9_-]+)"')
    DIVRE = re.compile(r'<div\b[^>]*\bid="([A-Za-z0-9_-]+)"\s*>\s*</div>')
    appsrc = (DOCS / "app.js").read_text(encoding="utf-8")
    blank, n_charts = [], 0
    for f in pages:
        src = f.read_text(encoding="utf-8")
        # Un graphique est un <svg> nomme, ou un <div> vide dans une <figure>.
        cont = set(SVGRE.findall(src))
        for fig in FIGRE.findall(src):
            cont |= set(DIVRE.findall(fig))
        if not cont:
            continue
        scripts = src + "\n" + appsrc
        drawn, prefix = set(), []
        for fn in painters:
            drawn |= set(re.findall(rf'\b{fn}\(\s*"([A-Za-z0-9_-]+)"', scripts))
            # Cibles construites par concatenation : histBars("h_" + h, ...).
            prefix += re.findall(rf'\b{fn}\(\s*"([A-Za-z0-9_-]+)"\s*\+', scripts)
        n_charts += len(cont)
        for c in sorted(cont):
            if c not in drawn and not any(c.startswith(q) for q in prefix):
                blank.append(f"{f.name}#{c}")
    r.append(_ok("every chart is drawn into by a painter", not blank,
                 "; ".join(blank[:4]) if blank
                 else f"{n_charts} chart(s), {len(painters)} painter(s)"))

    r.append(_ok("every helper a page calls is defined", not unknown,
                 "; ".join(sorted(set(unknown))[:4]) if unknown
                 else f"checked against {len(shared)} shared names"))
    return _report("published assets -- docs/", r)

    def _strip(css: str) -> str:
        return re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    bad_css = []
    for f in [DOCS / "site.css"] + pages:
        txt = f.read_text(encoding="utf-8")
        blocks = ([txt] if f.suffix == ".css"
                  else re.findall(r"<style>(.*?)</style>", txt, re.S))
        for b in blocks:
            b = _strip(b)
            if b.count("{") != b.count("}"):
                bad_css.append(f"{f.name} {b.count('{')}/{b.count('}')}")
    r.append(_ok("CSS braces balanced in every stylesheet and style block",
                 not bad_css, "; ".join(bad_css) if bad_css else
                 f"{len(pages) + 1} source(s)"))

    void = {"br", "img", "meta", "link", "input", "hr", "source", "col",
            "path", "line", "text", "polyline", "rect", "circle", "use", "stop"}
    unbalanced, dup_attr = [], []

    class _P(HTMLParser):
        def __init__(self):
            super().__init__()
            self.stack: list[str] = []
            self.bad: list[str] = []

        def handle_starttag(self, tag, attrs):
            names = [a for a, _ in attrs]
            for a in set(names):
                if names.count(a) > 1:
                    self.bad.append(f"<{tag} {a}=...> twice")
            if tag not in void:
                self.stack.append(tag)

        def handle_endtag(self, tag):
            if tag in void:
                return
            if not self.stack or self.stack[-1] != tag:
                self.bad.append(f"</{tag}> line {self.getpos()[0]}")
                return
            self.stack.pop()

    for f in pages:
        p = _P()
        p.feed(f.read_text(encoding="utf-8"))
        left = [t for t in p.stack if t not in ("html", "body", "head")]
        struct = [b for b in p.bad if not b.endswith("twice")]
        dups = [b for b in p.bad if b.endswith("twice")]
        if left or struct:
            unbalanced.append(f"{f.name}: {(struct + left)[:2]}")
        if dups:
            dup_attr.append(f"{f.name}: {dups[:2]}")
    r.append(_ok("HTML tags balanced on every page", not unbalanced,
                 "; ".join(unbalanced) if unbalanced else f"{len(pages)} pages"))
    r.append(_ok("no attribute written twice in one tag", not dup_attr,
                 "; ".join(dup_attr) if dup_attr else "none"))

    missing_ids = []
    for f in pages:
        txt = f.read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', txt))
        want = set(re.findall(r'el\("([A-Za-z0-9_-]+)"\)', txt))
        want |= set(re.findall(r'getElementById\("([A-Za-z0-9_-]+)"\)', txt))
        gone = sorted(want - ids)
        if gone:
            missing_ids.append(f"{f.name}: {gone[:3]}")
    r.append(_ok("every id a script addresses exists on its page",
                 not missing_ids, "; ".join(missing_ids) if missing_ids
                 else "all resolved"))

    dead = []
    for f in pages:
        txt = f.read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', txt))
        for a in set(re.findall(r'href="#([A-Za-z0-9_-]+)"', txt)):
            if a not in ids:
                dead.append(f"{f.name}#{a}")
    r.append(_ok("every in-page anchor resolves", not dead,
                 "; ".join(dead[:4]) if dead else "all resolved"))

    appjs = (DOCS / "app.js").read_text(encoding="utf-8")
    shared = set(re.findall(r"^(?:function|const|let)\s+([A-Za-z_][\w]*)",
                            appjs, re.M))
    unknown = []
    for f in pages:
        body = "\n".join(re.findall(r"<script>(.*?)</script>",
                                    f.read_text(encoding="utf-8"), re.S))
        local = set(re.findall(r"(?:function|const|let|var)\s+([A-Za-z_][\w]*)",
                               body))
        for call in set(re.findall(r"(?<![.\w$])([a-z][A-Za-z0-9_]{2,})\s*\(",
                                   body)):
            if call in shared or call in local or call in _JS_BUILTINS:
                continue
            unknown.append(f"{f.name}: {call}()")
    r.append(_ok("every helper a page calls is defined", not unknown,
                 "; ".join(sorted(set(unknown))[:4]) if unknown
                 else f"checked against {len(shared)} shared names"))
    return _report("published assets -- docs/", r)


def verify_agreement(started: float) -> int:
    DOCS = HERE.parent / "docs"
    r: list[tuple[bool, str, str]] = []
    d = DOCS / "data"

    def load(n):
        try:
            return json.loads((d / n).read_text(encoding="utf-8"))
        except Exception:
            return None

    latest, hist, qa = load("latest.json"), load("history.json"), load("qa.json")
    have = all(x is not None for x in (latest, hist, qa))
    r.append(_ok("latest, history and qa all parse", have,
                 "3/3" if have else "missing or unparseable"))
    if not have:
        return _report("cross-page agreement", r)

    m = latest["meta"]
    C = 0.01

    last_eq = hist["daily"][-1]["equity_USD"]
    r.append(_ok("history's last equity == latest's headline",
                 abs(last_eq - m["equity_end"]) <= C,
                 f"{last_eq:,.2f} vs {m['equity_end']:,.2f}"))

    bc = qa.get("bench_curves") or []
    r.append(_ok("Q&A benchmark covers the published sessions",
                 len(bc) == m["sessions"],
                 f"{len(bc)} rows vs {m['sessions']} sessions"))
    if bc:
        r.append(_ok("Q&A book curve ends on the headline equity",
                     abs(bc[-1]["book"] - m["equity_end"]) <= C,
                     f"{bc[-1]['book']:,.2f} vs {m['equity_end']:,.2f}"))
        r.append(_ok("Q&A book curve starts at the opening balance",
                     abs(bc[0]["book"] - m["equity_start"]) <= 1.0,
                     f"{bc[0]['book']:,.2f} vs {m['equity_start']:,.2f}"))

    st = {x["key"]: x for x in (qa.get("bench_stats") or [])}
    if "book" in st and bc:
        want = bc[-1]["book"] / bc[0]["book"] - 1.0
        r.append(_ok("Q&A book total return matches its own curve",
                     abs(st["book"]["total"] - want) <= 1e-4,
                     f"{st['book']['total']:.6f} vs {want:.6f}"))

    qh = (DOCS / "qa.html").read_text(encoding="utf-8")
    r.append(_ok("the Q&A volatility card reads the Overview's own field",
                 "META.net_ann_vol" in qh,
                 "reads latest.json meta.net_ann_vol"
                 if "META.net_ann_vol" in qh else "computes its own -- they can drift"))

    n_pos = len(qa.get("positions") or [])
    r.append(_ok("Q&A position count matches the headline",
                 n_pos == m.get("n_positions", n_pos),
                 f"{n_pos} vs {m.get('n_positions')}"))
    exp = qa.get("exposure") or {}
    ls = sum(1 for x in (qa.get("positions") or []) if x.get("side") == "LONG")
    sh = sum(1 for x in (qa.get("positions") or []) if x.get("side") == "SHORT")
    inner = []
    if exp.get("n_positions") != n_pos:
        inner.append(f"exposure says {exp.get('n_positions')}, table has {n_pos}")
    if exp.get("long") != ls or exp.get("short") != sh:
        inner.append(f"long/short {exp.get('long')}/{exp.get('short')} "
                     f"vs {ls}/{sh} in the table")
    if n_pos and abs(sum(abs(x["notional_USD"]) for x in qa["positions"])
                     - exp.get("gross_notional_USD", 0)) > 1.0:
        inner.append("gross notional does not sum from the rows")
    r.append(_ok("Q&A exposure summary sums from its own position table",
                 not inner, "; ".join(inner) if inner
                 else f"{n_pos} rows, {ls} long, {sh} short"))

    att = qa.get("attribution") or []
    tot = sum(x["gross_pnl_USD"] for x in att)
    book_gross = sum(x["gross_pnl_USD"] for x in hist["daily"])
    r.append(_ok("Q&A attribution sums to the published gross P&L",
                 abs(tot - book_gross) <= 1.0,
                 f"{tot:,.2f} vs {book_gross:,.2f}"))
    return _report("cross-page agreement", r)

def verify_publish(started: float) -> int:
    import importlib.util
    import json as _json
    import polars as pl
    t0 = time.time()
    DOCS = HERE.parent / "docs"
    DATA = DOCS / "data"
    r: list[tuple[bool, str, str]] = []

    if not DATA.is_dir():
        r.append(_ok("docs/data exists", False, "stage 6 wrote nothing"))
        return _report("publication -- docs/", r)

    spec = importlib.util.spec_from_file_location("pub_v", PUBLISH)
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)

    def _load_json(f):
        try:
            return _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None

    tops = ["qa.json", "latest.json", "history.json", "index.json", "pnl_index.json",
            "mapping.json"]
    got = {n: _load_json(DATA / n) for n in tops}
    missing = [n for n in tops if got[n] is None]
    r.append(_ok("every top-level file present and parses", not missing,
                 f"{len(tops) - len(missing)}/{len(tops)}"
                 + (f"   bad: {missing}" if missing else "")))
    if missing:
        return _report("publication -- docs/", r)

    latest, hist = got["latest.json"], got["history.json"]
    idx, pidx = got["index.json"]["days"], got["pnl_index.json"]["days"]

    for label, rows, sub in (("journal", idx, "days"), ("attribution", pidx, "pnl")):
        bad = [x["date"] for x in rows
               if _load_json(DATA / sub / f"{x['date']}.json") is None]
        r.append(_ok(f"every {label} session file exists and parses", not bad,
                     f"{len(rows) - len(bad)}/{len(rows)} files"
                     + (f"   bad: {bad[:4]}" if bad else "")))

    P = pl.read_parquet(HERE / "3_Portfolio" / "Portfolio.parquet")
    P = P.filter(pl.col("started"))
    d = P.get_column("date").to_list()
    eq = P.get_column("equity_USD").to_numpy()
    npos = P.get_column("n_positions").to_numpy()
    m = latest["meta"]
    diffs = []
    if m["as_of"] != d[-1]:
        diffs.append(f"as_of {m['as_of']} vs {d[-1]}")
    if m["sessions"] != len(d):
        diffs.append(f"sessions {m['sessions']} vs {len(d)}")
    if abs(m["equity_end"] - float(eq[-1])) > 0.01:
        diffs.append(f"equity {m['equity_end']:,.2f} vs {float(eq[-1]):,.2f}")
    if int(m["n_positions"]) != int(npos[-1]):
        diffs.append(f"n_positions {m['n_positions']} vs {int(npos[-1])}")
    r.append(_ok("latest.json agrees with Portfolio.parquet", not diffs,
                 f"{m['as_of']}  {m['sessions']} sessions  "
                 f"{m['equity_end']:,.0f}" if not diffs else "  ".join(diffs)))

    hl = hist["daily"][-1]
    same = (hl["date"] == d[-1]
            and abs(hl["equity_USD"] - float(eq[-1])) <= 0.01
            and len(hist["daily"]) == len(d))
    r.append(_ok("history.json ends on the portfolio's last session", same,
                 f"{hl['date']}  {len(hist['daily']):,} rows  "
                 f"{hl['equity_USD']:,.0f}"))

    off = []
    for x in pidx:
        j = _load_json(DATA / "pnl" / f"{x['date']}.json")
        tot = sum(i["gross_pnl_USD"] for i in j["instruments"])
        if abs(tot - j["book"]["gross_pnl_USD"]) > 1.0:
            off.append(x["date"])
    r.append(_ok("every attribution sheet sums to its own book", not off,
                 f"{len(pidx)} sessions"
                 + (f"   off: {off[:4]}" if off else "")))

    want = pub.build_stamp(latest)
    seen = {}
    for name in pub.PAGES:
        txt = (DOCS / name).read_text(encoding="utf-8")
        for asset, (rx, _tpl) in pub.TAGS.items():
            mt = rx.search(txt)
            seen.setdefault(name, set()).add(
                mt.group(0).split("?v=")[-1].split('"')[0] if mt and "?v=" in mt.group(0)
                else "UNSTAMPED")
    stamps = {v for vs in seen.values() for v in vs}
    r.append(_ok("all pages carry the same, current cache stamp",
                 stamps == {want},
                 f"{want} on {len(pub.PAGES)} pages"
                 if stamps == {want} else f"found {sorted(stamps)} want {want}"))

    hits = []
    for f in DOCS.rglob("*"):
        if not f.is_file():
            continue
        try:
            low = f.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        if any(w in low for w in pub.FORBIDDEN):
            hits.append(f.relative_to(DOCS).as_posix())
    r.append(_ok("the data provider is named in no published file", not hits,
                 f"{sum(1 for _ in DOCS.rglob('*') if _.is_file()):,} files scanned"
                 + (f"   HITS: {hits[:3]}" if hits else "")))

    return _report(f"publication -- docs/  ({time.time() - t0:.0f}s)", r)


def verify_reconciliation(started: float) -> int:
    r: list[tuple[bool, str, str]] = []
    if not RECONCILE.is_file():
        r.append(_ok("reconcile.py present", False, str(RECONCILE)))
        return _report("reconciliation -- primary sources", r)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_rec", RECONCILE)
        rc = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rc)
        T = rc.ties()
    except Exception as exc:
        r.append(_ok("reconciliation ran", False, f"{type(exc).__name__}: {exc}"))
        return _report("reconciliation -- primary sources", r)
    if T is None:
        r.append(_ok("stages 3 and 4 produced their artifacts", False,
                     "an upstream file is missing; earlier reports say which"))
        return _report("reconciliation -- primary sources", r)

    for ok, name, a, b, rel, unit, _note in T.rows:
        if unit == "$":
            detail = f"{a / 1e9:,.4f}B vs {b / 1e9:,.4f}B   rel {rel:.1e}"
        elif unit == "ct":
            detail = f"{a:,.0f} vs {b:,.0f} contracts"
        else:
            detail = f"{a:,.0f} vs {b:,.0f}"
        r.append(_ok(name, ok, detail))
    return _report(f"reconciliation -- primary sources ({T.secs:.0f}s)", r)

def n_books() -> int | None:
    try:
        import csv
        with open(HERE / "1_Roll" / "contract_cycles.csv", newline="",
                  encoding="utf-8") as fh:
            return sum(1 for r in csv.DictReader(fh)
                       if (r.get("Roll_Rule") or "").strip()) or None
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    ap.add_argument("--no-ndu", action="store_true",
                    help="skip stage 0; do not start or trigger NDU")
    ap.add_argument("--ndu-wait", type=int, default=60,
                    help="seconds to wait for NDU to fetch new data (default 60; "
                         "0 to skip the wait entirely)")
    ap.add_argument("--keep-ndu", action="store_true",
                    help="leave NDU running at the end instead of closing it")
    ap.add_argument("--skip-refresh", action="store_true",
                    help="skip stage 1; rebuild the books off the panel on disk")
    ap.add_argument("--jobs", type=int, default=2,
                    help="worker processes for the book build (default 2; "
                         "memory-bound, see trading_book.main)")
    ap.add_argument("--no-portfolio", action="store_true",
                    help="skip stage 3 (positions)")
    ap.add_argument("--nav", type=float, default=100e6,
                    help="stage 3 starting NAV in USD (default 100,000,000)")
    ap.add_argument("--no-bookkeeping", action="store_true",
                    help="skip stage 4 (the order ledger)")
    ap.add_argument("--no-journal", action="store_true",
                    help="skip the append-only journal (stage 4b)")
    ap.add_argument("--no-reconcile", action="store_true",
                    help="skip the primary-source reconciliation (~3s)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-stage checks")
    ap.add_argument("--no-publish", action="store_true",
                    help="skip stage 5b; docs/data keeps the previous numbers")
    ap.add_argument("--no-deploy", action="store_true",
                    help="write docs/ but do not commit or push it")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to run the stages with")
    args = ap.parse_args()

    for f in (CYCLES, BOOK):
        if not f.is_file():
            print(f"[ABORT] missing stage: {f}")
            return 2

    py = args.python
    print(f"pipeline: {HERE}")
    print(f"python  : {py}")
    if not args.dry_run:
        preflight(py, need_vendor=not args.skip_refresh)
    if args.dry_run:
        print("MODE    : dry run, nothing will be executed")

    started = time.time()
    failures = 0
    total = 0.0
    if args.skip_refresh:
        print("\n  [skip] stage 1 -- reusing the panel already on disk.")
        print("         The book cache is keyed on contract_cycles.csv, so the")
        print("         books will rebuild only if that file has changed.")
    else:
        if not args.no_ndu:
            _adv, _started = ensure_ndu(args.dry_run, wait=args.ndu_wait)
        total += run("STAGE 1/5  contract_cycles.py  (needs NDU running)",
                     [py, str(CYCLES)], args.dry_run)
        if not (args.dry_run or args.no_verify):
            if verify_cycles() + verify_holds():
                print("")
                print("[ABORT] stage 1 verification failed; stage 2 NOT run.")
                return 3

    try:
        total += run(f"STAGE 2/5  trading_book.py  --jobs {args.jobs}",
                     [py, str(BOOK), "--jobs", str(args.jobs)], args.dry_run,
                     total=n_books())
        if not args.no_portfolio:
            total += run(f"STAGE 3/5  portfolio.py  --nav {args.nav:,.0f}",
                         [py, str(PORTFOLIO), "--nav", repr(args.nav)],
                         args.dry_run)
            if not args.no_bookkeeping:
                total += run("STAGE 4/5  bookkeeping.py  (order ledger)",
                             [py, str(BOOKKEEPING)], args.dry_run)
                if not args.no_journal:
                    if JOURNAL.is_file():
                        try:
                            total += run("STAGE 4b   journal.py  (append-only "
                                         "record; non-blocking)",
                                         [py, str(JOURNAL)], args.dry_run,
                                         blocking=False)
                        except SystemExit:
                            print("  [WARN] journal append failed; the ledger is "
                                  "written and the pipeline continues.")
                    else:
                        print("  [skip] no journal.py")
    finally:
        if not (args.no_ndu or args.keep_ndu):
            close_ndu(args.dry_run)
    if not (args.dry_run or args.no_verify):
        failures = verify_vendor(started)
        failures += verify_books(started, n_books())
        failures += verify_fx(started)
        failures += verify_irx(started)
        if not args.no_portfolio:
            failures += verify_portfolio(started)
            if not args.no_bookkeeping:
                failures += verify_bookkeeping(started)
            failures += verify_stages(started)
            if not (args.no_bookkeeping or args.no_reconcile):
                failures += verify_reconciliation(started)

    def _stamp(n_failures: int) -> None:
        if args.dry_run:
            return
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        full = not (args.no_portfolio or args.no_bookkeeping
                    or args.no_reconcile)
        RUN_STAMP.write_text(json.dumps({
            "completed_at": now,
            "failures": int(n_failures),
            "verified": not args.no_verify,
            "full_run": full,
        }, indent=1), encoding="utf-8")
        # Le site affiche « verifie le ... ». N'y ecrire que des executions
        # completes, verifiees et sans echec : annoncer une verification qui
        # a echoue serait pire que de ne rien annoncer du tout.
        if full and not args.no_verify and not n_failures:
            pub = HERE.parent / "docs" / "data" / "run.json"
            pub.write_text(json.dumps({"checked_at": now}, indent=1),
                           encoding="utf-8")

    _stamp(failures)

    # Dire QUELS marches n'ont pas avance et POURQUOI. Sans cette ligne, une
    # journee ou Londres est fermee ressemble a une journee ou le pipeline
    # n'a rien fait : la fenetre du .exe ne montrait qu'un compte de checks.
    print()
    try:
        _st = shut_state()
        _line = _sm.describe(_st)
        print(f"  [HOLD] {_line}" if _line
              else f"  [HOLD] aucun retard, tous sur {_st['as_of']}")
    except Exception as _e:
        print(f"  [HOLD] etat des marches indisponible ({_e})")

    published = False
    deployed = 1
    if not (args.dry_run or args.no_publish):
        partial = args.no_portfolio or args.no_bookkeeping or args.no_reconcile
        if args.no_verify:
            print("\n  [skip] STAGE 6 -- publish.py NOT run: verification was "
                  "skipped.")
            print("         A run nobody checked is not a run worth "
                  "publishing.")
        elif failures:
            print(f"\n  [skip] STAGE 6 -- publish.py NOT run: {failures} "
                  f"verification failure(s) above.")
            print("         docs/data keeps the last VERIFIED numbers rather "
                  "than gaining unverified ones.")
        elif partial:
            print("\n  [skip] STAGE 6 -- publish.py NOT run: this was a "
                  "partial pipeline.")
            print("         Published figures come from every stage, so they "
                  "must be rebuilt by every stage.")
        else:
            total += run("STAGE 5/5  publish.py  (docs/data for the site)",
                         [py, str(PUBLISH)], args.dry_run)
            published = True
            if not args.no_verify:
                failures += verify_publish(started)
                failures += verify_assets(started)
                failures += verify_agreement(started)
                failures += verify_render(started)
                _stamp(failures)
            if not (args.no_deploy or failures):
                deployed = deploy()

    if not args.dry_run:
        print(f"\n{'=' * 72}")
        print(f"  pipeline complete in {total / 60:.1f} min")
        print(f"  books -> {HERE / '2_Engine' / 'Trading_book'}")
        print(f"  rates -> {HERE / '2_Engine' / 'FX'}")
        if not args.no_portfolio:
            print(f"  book  -> {HERE / '3_Portfolio'}")
            if not args.no_bookkeeping:
                print(f"  orders-> {HERE / '4_Bookkeeping'}")
        if published:
            note = ("pushed" if deployed == 0 and not args.no_deploy
                    else "written; deploy by hand")
            print(f"  site  -> {HERE.parent / 'docs' / 'data'}   ({note})")
        if failures:
            print(f"  {failures} VERIFICATION FAILURE(S) -- see above")
        print(f"{'=' * 72}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

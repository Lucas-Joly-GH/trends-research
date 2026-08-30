from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent
BK = HERE.parent
LIVE = BK.parent
POS = LIVE / "3_Portfolio" / "Positions"
PORT = LIVE / "3_Portfolio" / "Portfolio.parquet"
CYCLES = LIVE / "1_Roll" / "contract_cycles.csv"
BOOK_PY = LIVE / "2_Engine" / "trading_book.py"

MANIFEST = HERE / "MANIFEST.json"
ORDERS = HERE / "orders"
OUTST = HERE / "outstanding"
FILLS = HERE / "fills"
SUMS = HERE / "_checksums.jsonl"
SCHEMA_VERSION = 1

PROV = ["SIGNAL", "price_vol_USD_ann", "s_g_vol", "s_g_dd", "w_i", "IDM",
        "NAV", "N_raw", "N_target"]
CONTEXT = ["code_commit", "panel_edge", "cycles_fingerprint", "written_at"]
DECISION = (["order_id", "decision_date", "execute_at", "instrument",
             "contract", "action", "quantity", "kind", "position_before",
             "position_after", "decision_close"] + PROV)
ORDER_COLS = DECISION + CONTEXT
FILL_COLS = ["order_id", "filled_at", "fill_price", "fill_qty", "status",
             "source", "written_at"]
OUTST_COLS = ["order_id", "as_of", "decision_date", "instrument", "contract",
              "action", "quantity", "own_sessions_since", "carried_sessions",
              "reason", "written_at"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def order_id(decision_date: str, instrument: str, contract: str) -> str:
    return hashlib.sha256(
        f"{decision_date}|{instrument}|{contract}".encode()).hexdigest()[:16]


def manifest() -> dict | None:
    if not MANIFEST.is_file():
        return None
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except Exception:
        return None


WIPEABLE = {"test"}
MODELLED_FILLS_OK = {"test", "paper"}
MODES = ["test", "paper", "live"]


def promote(to: str) -> int:
    m = manifest()
    if m is None:
        print("  no store to promote")
        return 1
    cur = m.get("mode", "test")
    if to not in MODES:
        print(f"  [REFUSED] unknown mode '{to}'.  One of {MODES}.")
        return 1
    if MODES.index(to) <= MODES.index(cur):
        print(f"  [REFUSED] store is '{cur}'; '{to}' is not forward of it. "
              f"Promotion is one way.")
        return 1
    if to not in WIPEABLE:
        import subprocess
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(ORDERS)],
            cwd=str(HERE), capture_output=True).returncode == 0
        if ignored:
            print(f"  [REFUSED] the store is gitignored, so promoting it to "
                  f"'{to}' would make an")
            print("            unrewritable record that exists on exactly one "
                  "disk.")
            print("            Remove the Journal/ entries from .gitignore and "
                  "commit the store first.")
            return 1
    m["mode"] = to
    m["promoted_at"] = _now()
    MANIFEST.write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(f"  store promoted {cur} -> {to}")
    if to not in WIPEABLE:
        print("  --reset will now refuse. The store is no longer deletable "
              "from here.")
        print("  IT IS ALSO NO LONGER DISPOSABLE: check that it is committed, "
              "or it lives on one disk.")
    if to not in MODELLED_FILLS_OK:
        print("  --fills will now refuse. Fills must come from a broker.")
    return 0


def ensure_store(mode: str = "test") -> dict:
    m = manifest()
    if m is None:
        ORDERS.mkdir(parents=True, exist_ok=True)
        OUTST.mkdir(parents=True, exist_ok=True)
        FILLS.mkdir(parents=True, exist_ok=True)
        m = {"mode": mode, "schema_version": SCHEMA_VERSION,
             "created_at": _now()}
        MANIFEST.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"  created store  mode={mode}  schema v{SCHEMA_VERSION}")
    return m


def _record(p: Path, rows: int) -> None:
    with SUMS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"path": str(p.relative_to(HERE)).replace("\\", "/"),
                             "sha256": _sha(p), "rows": rows,
                             "written_at": _now()}) + "\n")


def _sums() -> dict[str, dict]:
    if not SUMS.is_file():
        return {}
    out = {}
    for line in SUMS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["path"]] = r
    return out


def _path(kind: Path, date: str) -> Path:
    return kind / date[:4] / f"{date}.parquet"


def _run_ids() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                cwd=str(LIVE), capture_output=True, text=True,
                                timeout=20).stdout.strip() or "unknown"
    except Exception:
        commit = "unknown"
    edge = "unknown"
    if PORT.is_file():
        try:
            edge = pl.read_parquet(PORT, columns=["date"])["date"].max()
        except Exception:
            pass
    fp = _sha(CYCLES)[:12] if CYCLES.is_file() else "unknown"
    return {"code_commit": commit, "panel_edge": edge,
            "cycles_fingerprint": fp}


def build_rows(asof: str | None, led=None, tb=None,
               prov_cache: dict | None = None) -> tuple[str, pl.DataFrame]:
    bk = _load(BK / "bookkeeping.py", "bk")
    if led is None:
        tb = tb or _load(BOOK_PY, "tb")
        led = bk.build(tb)
    dates = sorted(set(led.get_column("decision_date").to_list()))
    if not dates:
        raise SystemExit("[ABORT] derived ledger is empty; run stage 4 first")
    asof = asof or dates[-1]
    if asof not in dates:
        raise SystemExit(f"[ABORT] no orders decided on {asof}")
    pend, _exe = bk.views(led, asof)

    ids = _run_ids()
    if prov_cache is None:
        prov_cache = {}
    prov: dict[str, dict] = {}
    for inst in sorted(set(pend.get_column("instrument").to_list())):
        if inst not in prov_cache:
            f = POS / f"{inst}.parquet"
            if not f.is_file():
                prov_cache[inst] = {}
            else:
                t = pl.read_parquet(f, columns=["date"] + PROV)
                cols = {c: t.get_column(c).to_list() for c in PROV}
                prov_cache[inst] = {
                    d: {c: cols[c][k] for c in PROV}
                    for k, d in enumerate(t.get_column("date").to_list())}
        got = prov_cache[inst].get(asof)
        if got:
            prov[inst] = got

    out = []
    for r in pend.iter_rows(named=True):
        p = prov.get(r["instrument"], {})
        out.append([order_id(r["decision_date"], r["instrument"], r["contract"]),
                    r["decision_date"], r["execute_at"], r["instrument"],
                    r["contract"], r["action"], r["quantity"], r["kind"],
                    r["position_before"], r["position_after"],
                    r["decision_close"]]
                   + [p.get(c) for c in PROV]
                   + [ids["code_commit"], ids["panel_edge"],
                      ids["cycles_fingerprint"], _now()])
    schema = {c: (pl.Utf8 if c in ("order_id", "decision_date", "execute_at",
                                   "instrument", "contract", "action", "kind",
                                   "code_commit", "panel_edge",
                                   "cycles_fingerprint", "written_at")
                  else pl.Float64) for c in ORDER_COLS}
    return asof, pl.DataFrame(out, orient="row", schema=schema)


def append(asof: str | None) -> int:
    ensure_store()
    date, df = build_rows(asof)
    p = _path(ORDERS, date)
    known = set()
    for q in sorted(ORDERS.glob("*/*.parquet")):
        if q == p:
            continue
        known |= set(pl.read_parquet(q, columns=["order_id"])
                     .get_column("order_id").to_list())
    carried = df.filter(pl.col("order_id").is_in(list(known))) if known         else df.head(0)
    if known:
        df = df.filter(~pl.col("order_id").is_in(list(known)))
    if carried.height:
        _write_outstanding(date, carried)
    if not df.height:
        print(f"  {date}: nothing newly given")
        return 0
    if p.is_file():
        old = pl.read_parquet(p)
        a = old.drop(CONTEXT).sort("order_id")
        b = df.drop(CONTEXT).sort("order_id")
        if a.equals(b):
            one = lambda f, c: sorted({str(x) for x in f.get_column(c).to_list()})
            moved = [c for c in CONTEXT if c != "written_at"
                     and one(old, c) != one(df, c)]
            print(f"  {date}: already journalled, decision identical "
                  f"({df.height} orders) -- no write")
            for c in moved:
                print(f"      context moved: {c} "
                      f"{','.join(one(old, c))} -> {','.join(one(df, c))}")
            return 0
        print(f"  [CONFLICT] {date} is already journalled and the rebuild "
              f"disagrees.")
        print(f"      stored {old.height} orders, rebuilt {df.height}")
        ids_old = set(old.get_column("order_id").to_list())
        ids_new = set(df.get_column("order_id").to_list())
        print(f"      only in store: {len(ids_old - ids_new)}   "
              f"only in rebuild: {len(ids_new - ids_old)}   "
              f"common but changed: "
              f"{sum(1 for k in (ids_old & ids_new) if not old.filter(pl.col('order_id') == k).drop(CONTEXT).equals(df.filter(pl.col('order_id') == k).drop(CONTEXT)))}")
        print(f"      NOTHING WRITTEN.  The stored file is the record; the "
              f"rebuild is today's opinion of it.")
        return 1
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    _record(p, df.height)
    print(f"  {date}: journalled {df.height} orders -> {p.relative_to(HERE)}")
    return 0


def _write_outstanding(date: str, carried: pl.DataFrame) -> None:
    tb = _load(BOOK_PY, "tb")
    prior = read_all(OUTST)
    seen: dict[str, int] = {}
    if prior.height:
        for k in prior.filter(pl.col("as_of") < date).get_column("order_id").to_list():
            seen[k] = seen.get(k, 0) + 1
    rows = []
    for r in carried.iter_rows(named=True):
        inst = r["instrument"]
        b = tb.load_book(inst).select(["date"])
        own = [x for x in b.get_column("date").to_list()
               if r["decision_date"] < x <= date]
        reason = "MARKET_CLOSED" if not own else "UNEXPLAINED"
        rows.append([r["order_id"], date, r["decision_date"], inst,
                     r["contract"], r["action"], r["quantity"],
                     float(len(own)), float(seen.get(r["order_id"], 0) + 1),
                     reason, _now()])
    schema = {c: (pl.Float64 if c in ("quantity", "own_sessions_since",
                                      "carried_sessions")
                  else pl.Utf8) for c in OUTST_COLS}
    df = pl.DataFrame(rows, orient="row", schema=schema)
    p = _path(OUTST, date)
    if p.is_file():
        old = pl.read_parquet(p)
        if old.drop("written_at").sort("order_id").equals(
                df.drop("written_at").sort("order_id")):
            return
        print(f"  [CONFLICT] outstanding book for {date} already written and "
              f"disagrees -- nothing written")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)
    _record(p, df.height)
    by = {}
    for r in rows:
        by[r[8]] = by.get(r[8], 0) + 1
    print(f"  {date}: {df.height} order(s) still outstanding  "
          + "  ".join(f"{k} {v}" for k, v in sorted(by.items())))


def resolve_fills(asof: str | None) -> int:
    m = ensure_store()
    if m.get("mode") not in MODELLED_FILLS_OK:
        print(f"  [REFUSED] modelled fills may not be written to a "
              f"'{m.get('mode')}' store.")
        return 1
    tb = _load(BOOK_PY, "tb")
    have = {r["order_id"] for r in read_all(FILLS).iter_rows(named=True)} \
        if read_all(FILLS).height else set()
    orders = read_all(ORDERS)
    if not orders.height:
        print("  no orders journalled yet")
        return 0
    books: dict[str, dict] = {}
    new: dict[str, list] = {}
    for r in orders.iter_rows(named=True):
        if r["order_id"] in have:
            continue
        inst = r["instrument"]
        if inst not in books:
            b = tb.load_book(inst).select(["date", "open"])
            d = b.get_column("date").to_list()
            books[inst] = {"dates": d, "open": b.get_column("open").to_list(),
                           "ix": {x: k for k, x in enumerate(d)}}
        bk = books[inst]
        k = bk["ix"].get(r["decision_date"])
        if k is None or k + 1 >= len(bk["dates"]):
            continue
        fdate = bk["dates"][k + 1]
        if asof and fdate > asof:
            continue
        px = bk["open"][k + 1]
        if px is None or px != px:
            continue
        new.setdefault(fdate, []).append(
            [r["order_id"], fdate, float(px), float(r["quantity"]),
             "FILLED", "modelled", _now()])
    if not new:
        print("  no new fills to resolve")
        return 0
    n = 0
    for fdate, rows in sorted(new.items()):
        p = _path(FILLS, fdate)
        if p.is_file():
            prev = pl.read_parquet(p)
            rows = [r for r in rows
                    if r[0] not in set(prev.get_column("order_id").to_list())]
            if not rows:
                continue
            df = pl.concat([prev, pl.DataFrame(rows, orient="row",
                                               schema=_fill_schema())])
        else:
            df = pl.DataFrame(rows, orient="row", schema=_fill_schema())
            p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)
        _record(p, df.height)
        n += len(rows)
    print(f"  resolved {n} fill(s) across {len(new)} session(s)")
    return 0


def _fill_schema() -> dict:
    return {c: (pl.Float64 if c in ("fill_price", "fill_qty") else pl.Utf8)
            for c in FILL_COLS}


def read_all(kind: Path) -> pl.DataFrame:
    fs = sorted(kind.glob("*/*.parquet")) if kind.is_dir() else []
    if not fs:
        cols = (ORDER_COLS if kind == ORDERS
                else OUTST_COLS if kind == OUTST else FILL_COLS)
        return pl.DataFrame({c: [] for c in cols})
    return pl.concat([pl.read_parquet(f) for f in fs], how="vertical_relaxed")


def status() -> int:
    m = manifest()
    if m is None:
        print("  no store yet")
        return 0
    o, f = read_all(ORDERS), read_all(FILLS)
    import subprocess as _sp
    _ig = _sp.run(["git", "check-ignore", "-q", str(ORDERS)],
                  cwd=str(HERE), capture_output=True).returncode == 0
    print(f"  backup  {'NOT tracked by git -- one disk only' if _ig else 'tracked by git'}"
          f"{'   (fine while mode=test)' if _ig and m.get('mode') in WIPEABLE else ''}")
    print(f"  mode {m['mode']}   schema v{m['schema_version']}   "
          f"created {m['created_at']}")
    if o.height:
        d = sorted(set(o.get_column("decision_date").to_list()))
        print(f"  orders  {o.height:,} rows over {len(d)} session(s)"
              f"   {d[0]} .. {d[-1]}")
        print(f"          commits {sorted(set(o.get_column('code_commit').to_list()))}"
              f"   panel edges {sorted(set(o.get_column('panel_edge').to_list()))}")
    else:
        print("  orders  empty")
    if f.height:
        print(f"  fills   {f.height:,} rows   "
              f"{sorted(set(f.get_column('source').to_list()))}")
        print(f"  unfill  {o.height - f.height:,} order(s) never filled")
    else:
        print("  fills   empty")
    u = read_all(OUTST)
    if u.height:
        by = (u.group_by("reason").len().sort("len", descending=True)
              .iter_rows(named=True))
        print(f"  outstd  {u.height:,} carry-forward row(s) over "
              f"{u.get_column('as_of').n_unique()} session(s)   "
              + "  ".join(f"{r['reason']} {r['len']}" for r in by))
        print(f"          affecting {u.get_column('order_id').n_unique()} "
              f"distinct order(s); longest carry "
              f"{u.get_column('as_of').n_unique() and int(u.group_by('order_id').len().get_column('len').max())} session(s)")
    else:
        print("  outstd  empty")
    return 0


def verify() -> int:
    m = manifest()
    if m is None:
        print("  no store yet")
        return 1
    bad = 0
    sums = _sums()
    files = sorted(list(ORDERS.glob("*/*.parquet"))
                   + list(OUTST.glob("*/*.parquet"))
                   + list(FILLS.glob("*/*.parquet")))
    print(f"  {len(files)} file(s), {len(sums)} checksum record(s)")
    for p in files:
        rel = str(p.relative_to(HERE)).replace("\\", "/")
        rec = sums.get(rel)
        if rec is None:
            print(f"  [FAIL] no checksum recorded for {rel}")
            bad += 1
        elif rec["sha256"] != _sha(p):
            print(f"  [FAIL] {rel} has been MODIFIED since it was written")
            bad += 1
    for rel in sums:
        if not (HERE / rel).is_file():
            print(f"  [FAIL] {rel} was written and is now missing")
            bad += 1
    o = read_all(ORDERS)
    if o.height:
        ids = o.get_column("order_id").to_list()
        dup = len(ids) - len(set(ids))
        if dup:
            print(f"  [FAIL] {dup} duplicate order_id(s)")
            bad += 1
        miss = [c for c in ORDER_COLS if c not in o.columns]
        if miss:
            print(f"  [FAIL] missing column(s): {miss}")
            bad += 1
        nulls = [c for c in PROV
                 if o.get_column(c).null_count() == o.height and o.height]
        if nulls:
            print(f"  [FAIL] provenance never populated: {nulls}")
            bad += 1
        d = sorted(set(o.get_column("decision_date").to_list()))
        led_f = BK / "Orders.parquet"
        if led_f.is_file():
            had = {x for x in pl.read_parquet(led_f, columns=["decision_date"])
                   .get_column("decision_date").to_list() if d[0] <= x <= d[-1]}
            gaps = sorted(had - set(d))
            if gaps:
                print(f"  [WARN] {len(gaps)} session(s) decided orders but were "
                      f"never journalled: {gaps[:5]}")
    u = read_all(OUTST)
    if u.height:
        oid0 = set(o.get_column("order_id").to_list())
        orph = [x for x in u.get_column("order_id").to_list() if x not in oid0]
        if orph:
            print(f"  [FAIL] {len(orph)} outstanding row(s) with no given order")
            bad += 1
        late = sum(1 for r in u.iter_rows(named=True)
                   if r["as_of"] <= r["decision_date"])
        if late:
            print(f"  [FAIL] {late} outstanding row(s) dated on or before their "
                  f"own order")
            bad += 1
        odd = sorted({r["reason"] for r in u.iter_rows(named=True)}
                     - {"MARKET_CLOSED", "UNEXPLAINED", "NOT_SENT", "REJECTED"})
        if odd:
            print(f"  [FAIL] unknown reason code(s): {odd}")
            bad += 1
    f = read_all(FILLS)
    if f.height:
        oid = set(o.get_column("order_id").to_list())
        orph = [x for x in f.get_column("order_id").to_list() if x not in oid]
        if orph:
            print(f"  [FAIL] {len(orph)} fill(s) with no journalled order")
            bad += 1
        q = dict(zip(o.get_column("order_id").to_list(),
                     o.get_column("quantity").to_list()))
        over = sum(1 for r in f.iter_rows(named=True)
                   if r["order_id"] in q
                   and r["fill_qty"] > q[r["order_id"]] + 1e-9)
        if over:
            print(f"  [FAIL] {over} fill(s) exceed the order quantity")
            bad += 1
    print(f"  {'VERIFIED' if not bad else str(bad) + ' PROBLEM(S)'}")
    return 1 if bad else 0


def replay(date: str) -> int:
    p = _path(ORDERS, date)
    if not p.is_file():
        print(f"  nothing journalled for {date}")
        return 1
    df = pl.read_parquet(p)
    r0 = df.to_dicts()[0]
    print(f"  {date}: {df.height} orders   commit {r0['code_commit']}"
          f"   panel edge {r0['panel_edge']}   cycles {r0['cycles_fingerprint']}")
    with pl.Config(tbl_rows=50, tbl_width_chars=200, fmt_str_lengths=12):
        print(df.select(["instrument", "contract", "action", "quantity", "kind",
                         "decision_close", "SIGNAL", "price_vol_USD_ann",
                         "s_g_vol", "s_g_dd", "IDM", "NAV", "N_raw"]))
    return 0


def backfill(since: str | None = None) -> int:
    ensure_store()
    t0 = time.time()
    bk = _load(BK / "bookkeeping.py", "bk")
    tb = _load(BOOK_PY, "tb")
    led = bk.build(tb)
    dates = sorted(set(led.get_column("decision_date").to_list()))
    if since:
        dates = [d for d in dates if d >= since]
    if not dates:
        print("  nothing to backfill")
        return 0

    known: set[str] = set()
    carried_count: dict[str, int] = {}
    for q in sorted(ORDERS.glob("*/*.parquet")):
        known |= set(pl.read_parquet(q, columns=["order_id"])
                     .get_column("order_id").to_list())
    for q in sorted(OUTST.glob("*/*.parquet")):
        for k in pl.read_parquet(q, columns=["order_id"]).get_column("order_id").to_list():
            carried_count[k] = carried_count.get(k, 0) + 1

    prov_cache: dict = {}
    own_cache: dict[str, list] = {}
    ids = _run_ids()
    n_new = n_out = n_skip = 0
    print(f"  backfilling {len(dates):,} session(s) {dates[0]} .. {dates[-1]}")
    for k, d in enumerate(dates):
        if _path(ORDERS, d).is_file():
            n_skip += 1
            known |= set(pl.read_parquet(_path(ORDERS, d), columns=["order_id"])
                         .get_column("order_id").to_list())
            continue
        _d, df = build_rows(d, led=led, tb=tb, prov_cache=prov_cache)
        carried = df.filter(pl.col("order_id").is_in(list(known))) \
            if known else df.head(0)
        fresh = df.filter(~pl.col("order_id").is_in(list(known))) \
            if known else df

        if carried.height:
            rows = []
            for r in carried.iter_rows(named=True):
                inst = r["instrument"]
                if inst not in own_cache:
                    own_cache[inst] = tb.load_book(inst).get_column("date").to_list()
                own = [x for x in own_cache[inst]
                       if r["decision_date"] < x <= d]
                c = carried_count.get(r["order_id"], 0) + 1
                carried_count[r["order_id"]] = c
                rows.append([r["order_id"], d, r["decision_date"], inst,
                             r["contract"], r["action"], r["quantity"],
                             float(len(own)), float(c),
                             "MARKET_CLOSED" if not own else "UNEXPLAINED",
                             _now()])
            schema = {c: (pl.Float64 if c in ("quantity", "own_sessions_since",
                                              "carried_sessions") else pl.Utf8)
                      for c in OUTST_COLS}
            odf = pl.DataFrame(rows, orient="row", schema=schema)
            p = _path(OUTST, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            odf.write_parquet(p)
            _record(p, odf.height)
            n_out += odf.height

        if fresh.height:
            p = _path(ORDERS, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            fresh.write_parquet(p)
            _record(p, fresh.height)
            known |= set(fresh.get_column("order_id").to_list())
            n_new += fresh.height
        if (k + 1) % 500 == 0:
            print(f"    {k + 1:,}/{len(dates):,}  {d}  "
                  f"{n_new:,} given  {n_out:,} carried  ({time.time() - t0:.0f}s)")
    print(f"  backfilled {n_new:,} order(s), {n_out:,} carry-forward row(s), "
          f"{n_skip:,} session(s) already present  ({time.time() - t0:.0f}s)")
    return 0


def drift() -> int:
    o = read_all(ORDERS)
    if not o.height:
        print("  journal is empty")
        return 0
    bk = _load(BK / "bookkeeping.py", "bk")
    tb = _load(BOOK_PY, "tb")
    led = bk.build(tb)
    key = lambda d, i, c: order_id(d, i, c)
    now = {key(r["decision_date"], r["instrument"], r["contract"]): r
           for r in led.iter_rows(named=True)}
    same = changed = gone = 0
    rows = []
    for r in o.iter_rows(named=True):
        c = now.get(r["order_id"])
        if c is None:
            gone += 1
            rows.append((r, None))
            continue
        if (abs(c["quantity"] - r["quantity"]) > 1e-9
                or c["action"] != r["action"] or c["kind"] != r["kind"]):
            changed += 1
            rows.append((r, c))
        else:
            same += 1
    jd = set(o.get_column("decision_date").to_list())
    jids = set(o.get_column("order_id").to_list())
    extra = sum(1 for k, c in now.items()
                if c["decision_date"] in jd and k not in jids)
    print(f"  journalled {o.height:,} orders over {len(jd)} session(s)")
    print(f"    unchanged in today's ledger : {same:,}")
    print(f"    quantity/side/kind changed  : {changed:,}")
    print(f"    no longer in the ledger     : {gone:,}")
    print(f"    in the ledger, never journalled, on a journalled session: {extra:,}")
    for r, c in rows[:6]:
        if c is None:
            print(f"      {r['decision_date']} {r['instrument']:<6} "
                  f"{r['action']} {r['quantity']:,.0f} -> GONE")
        else:
            print(f"      {r['decision_date']} {r['instrument']:<6} "
                  f"{r['action']} {r['quantity']:,.0f} -> "
                  f"{c['action']} {c['quantity']:,.0f}")
    if changed or gone:
        print("")
        print("  Drift is EXPECTED after any pipeline change.  The journal is "
              "what was sent;")
        print("  the ledger is today's opinion of it.  Neither is corrected "
              "from the other.")
    return 0


def reset() -> int:
    m = manifest()
    if m is None:
        print("  no store to reset")
        return 0
    if m.get("mode") not in WIPEABLE:
        print(f"  [REFUSED] store mode is '{m.get('mode')}'.  Only a test "
              f"journal is deletable from here.")
        print(f"  Move {HERE} by hand if you really mean it.")
        return 1
    for p in (ORDERS, OUTST, FILLS):
        if p.is_dir():
            shutil.rmtree(p)
    for p in (SUMS, MANIFEST):
        if p.is_file():
            p.unlink()
    print("  store reset (mode was test)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None,
                    help="treat this session as 'today'")
    ap.add_argument("--fills", action="store_true",
                    help="resolve modelled fills for orders whose open has come")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--replay", default=None, metavar="DATE")
    ap.add_argument("--backfill", action="store_true",
                    help="replay every session in the derived ledger, in order")
    ap.add_argument("--since", default=None, metavar="DATE",
                    help="with --backfill: start here")
    ap.add_argument("--outstanding", default=None, metavar="DATE",
                    help="the open book at the close of a session")
    ap.add_argument("--drift", action="store_true",
                    help="journal vs today's derived ledger (a report)")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--promote", metavar="MODE",
                    help="move the store forward: test -> paper -> live")
    a = ap.parse_args()
    if a.promote:
        return promote(a.promote)
    if a.reset:
        return reset()
    if a.status:
        return status()
    if a.verify:
        return verify()
    if a.replay:
        return replay(a.replay)
    if a.backfill:
        return backfill(a.since)
    if a.outstanding:
        p = _path(OUTST, a.outstanding)
        if not p.is_file():
            print(f"  nothing outstanding at the close of {a.outstanding}")
            return 0
        with pl.Config(tbl_rows=40, tbl_width_chars=160, fmt_str_lengths=14):
            print(pl.read_parquet(p).select(
                ["decision_date", "instrument", "contract", "action",
                 "quantity", "own_sessions_since", "carried_sessions",
                 "reason"]))
        return 0
    if a.drift:
        return drift()
    if a.fills:
        return resolve_fills(a.date)
    return append(a.date)


if __name__ == "__main__":
    sys.exit(main())

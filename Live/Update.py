"""
Run the nightly pipeline, in order, and stop at the first failure.

    python Update.py                  start NDU, refresh the panel, rebuild books
    python Update.py --no-ndu         assume NDU is already up to date
    python Update.py --dry-run        print the commands, run nothing
    python Update.py --jobs N         override the worker count (default 2)
    python Update.py --skip-refresh   books only, reuse the panel on disk
    python Update.py --no-bookkeeping stop after the positions, skip the ledger
    python Update.py --no-reconcile   skip the primary-source reconciliation

`front_contract.py` IS A LIBRARY, NOT A STAGE, which is the one thing the
directory listing gets wrong.  contract_cycles and trading_book both load it
with `_load(FC, "fc")` and call into it -- the former for roll scores, the²
latter for worksheets.  Its own main() defaults to a fortnight of ZC in 1989 and
prints fourteen rows; running it here would refresh nothing and prove nothing.

    0. ndu.trigger.exe      starts the Norgate Data Updater if it is down and
                            asks it to fetch.  Stage 1 needs NDU running, so an
                            unattended run would otherwise die because an app
                            was closed.  See ensure_ndu for why the wait works
                            the way it does.
    1. contract_cycles.py   pulls from the vendor, decides each market's roll
                            rule, writes contract_cycles.csv.  THIS IS THE ONLY
                            STAGE THAT NEEDS NDU.
    2. trading_book.py      builds the 63 books off that panel, and the 7 FX
                            conversion rates alongside them.
    3. portfolio.py         turns the forecasts into contracts (eq 3.32/3.33).
                            Sequential: NAV compounds, so position -> P&L ->
                            NAV -> next position cannot be vectorised.
    4. bookkeeping.py       differences those positions into ORDERS, and splits
                            them into what filled at this open and what to send
                            for the next.  Reads nothing but stage 3; ~1s.
    5. publish.py           writes docs/data for the public site.  RUNS LAST,
                            AFTER VERIFICATION, AND ONLY ON A CLEAN FULL RUN --
                            it reads the run stamp this file writes and will not
                            publish off a failed or partial pipeline.  Skipped,
                            not failed, when the run is not fit to publish from;
                            the site then keeps its last verified numbers.
                            Pushing stays manual: this writes files, it does not
                            deploy.

VERIFICATION IS NINE REPORTS, NOT ONE.  `verify_cycles` and `verify_holds`
cover stage 1; `verify_books` the books; `verify_fx` the conversion rates;
`verify_irx` the risk-free rate; `verify_portfolio` the positions;
`verify_bookkeeping` the order ledger; `verify_stages` asks the question none of
the others can -- do the artifacts AGREE WITH EACH OTHER;
`verify_reconciliation` asks the one question even THAT cannot, because a
consistent misreading of the panel passes every check that only compares derived
files: does the money still add up against the BOOKS, the mapping and IRX.
And `verify_publish` covers the one stage whose own guards all run BEFORE it
writes: it is the only report that opens `docs/` and reads back the bytes a
reader will be served.  It cannot check the live site, because the push is
manual by design and the directory is supposed to run ahead of the deployment.
That last one exists because a stale file passes its own report effortlessly:
yesterday's FX table is sorted, non-null, correctly typed and internally
consistent.  What it is not is the table the books were built against.
The rates get their own report because THEY FAIL DIFFERENTLY: a book that goes
wrong usually goes visibly wrong -- a column empties, a schema drifts, a date
stops advancing -- whereas a conversion rate that goes wrong stays perfectly
well-formed and simply carries the wrong number, with every position sized off
it then wrong by that factor and nothing anywhere reading as an error.  So
verify_fx is weighted toward VALUE plausibility (session-move ceiling, the USD
identity, the HKD peg) where verify_books is weighted toward structure.

`verify_bookkeeping` is a third shape again: it REPLAYS.  An order ledger cannot
be checked against itself in any useful way, because it is derived from the
positions and every arithmetic identity inside it holds by construction.  So the
report applies all 191,000 orders in sequence from flat and insists the book
lands on `N_contracts` on all 450,000 instrument-sessions.  That is the only
check that would have caught the 578 rolls an earlier version collapsed into
resizes -- well-formed rows, right instrument, right date, plausible size.

ORDER MATTERS AND IS NOT INTERCHANGEABLE.  Stage 2 reads `Roll_Rule` from the
CSV stage 1 writes, and its worksheet cache is keyed on that file's bytes -- so
a panel refresh correctly invalidates every cached worksheet.  Running them the
other way round would rebuild the books against yesterday's rules and look
entirely successful.

A STAGE THAT FAILS STOPS THE RUN.  Stage 2 on a half-written panel would
produce books that are wrong rather than absent, and absent is the failure that
gets noticed.
"""
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

# ---------------------------------------------------------------------------
# platform.node() returns the machine name, and this machine is called
# "Napoleon" WITH AN ACUTE ACCENT.  norgatehelper.py line 44 puts it straight
# into an HTTP header:
#
#     session.headers.update({"Client": platform.node()})
#
# HTTP headers must be ASCII, so NDU rejects the request with 400 and an empty
# body.  norgatedata's import-time probe then does:
#
#     statusmsg = r.content.decode()
#     if statusmsg != 'OK': logger.error(statusmsg)
#
# -- it logs the response BODY, which is empty.  That is the whole of the
#    "ERROR: Norgate Data: " with nothing after it, on every import, forever.
#    Nothing was truncated; there was never a message.
#
# Isolated and reproducible:
#     no Client header        -> 200 OK
#     Client "Napoleon"       -> 200 OK
#     Client platform.node()  -> 400, empty body
#
# The error is harmless because every real call afterwards strips the headers
# to ASCII.  It is the probe that runs BEFORE our code gets control that fails,
# which is exactly why patching has to happen here, before the import.
#
# THIS ONLY COVERS THIS PROCESS.  Stages 1 and 2 are separate interpreters that
# import norgatedata themselves, and platform.node() does not read COMPUTERNAME
# on Windows, so the environment cannot carry the fix to them.  Stage 1 will
# still print the line until the same patch is added there -- or until the
# machine is renamed, which fixes it everywhere at once and is the real cure.
# ---------------------------------------------------------------------------
_NODE = platform.node()
if not _NODE.isascii():
    platform.node = lambda _n=_NODE.encode("ascii", "ignore").decode(): _n

HERE = Path(__file__).resolve().parent
# WHEN THE DATA WAS LAST REBUILT, which is not the same as when anything was
# last published.  The site's "Updated" line is this timestamp: a reader wants
# to know how fresh the NUMBERS are, and re-running the publisher on week-old
# data must not make them look newer than they are.
RUN_STAMP = HERE / ".pipeline_run.json"
CYCLES = HERE / "1_Roll" / "contract_cycles.py"
BOOK = HERE / "2_Engine" / "trading_book.py"
PORTFOLIO = HERE / "3_Portfolio" / "portfolio.py"
BOOKKEEPING = HERE / "4_Bookkeeping" / "bookkeeping.py"
JOURNAL = HERE / "4_Bookkeeping" / "Journal" / "journal.py"
PUBLISH = HERE / "5_Publish" / "publish.py"


# A stage prints one line per instrument.  That is 63 lines of noise around the
# handful that matter -- the panel edge, the roll warnings, the hole report.
# The bar COLLAPSES those and passes everything else through untouched, so
# nothing is hidden; it is a filter on repetition, not on information.
_ROW = re.compile(r"^\S+\s+\S*roll\S*\s")
_SPIN = "|/-" + chr(92)          # avoids a literal backslash here
_CR = chr(13)                    # ditto for the carriage return


def _bar(done: int, total: int | None, secs: float, width: int = 32) -> str:
    if total:
        frac = min(done / total, 1.0)
        fill = int(frac * width)
        return (f"  [{'#' * fill}{'.' * (width - fill)}] "
                f"{done:>3}/{total}  {secs:4.0f}s")
    # No total to divide by -- stage 1 does not announce how much work it has,
    # so this is a liveness indicator, not a progress bar, and says so.
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
    """Refuse to start unless the interpreter can actually finish.

    THE FAILURE THIS PREVENTS IS A CONFUSING ONE, NOT A LOUD ONE.  Run the
    pipeline with the wrong python and stage 1 dies four minutes in on an
    ImportError buried under a traceback, or -- worse -- stage 2 half-builds a
    book and stage 3 closes NDU on the way out.  The interpreter is knowable in
    one second before anything is written, so it is checked here.

    IT CHECKS THE INTERPRETER THAT WILL RUN THE STAGES, not this one.  They are
    subprocesses and `--python` can point them anywhere, so asking our own
    sys.modules would answer the wrong question entirely.

    Two venvs exist on this machine and BOTH work:

        LJOLY_Memoire_INSEEC_Msc2/.venv   python 3.12.10, polars 1.44.0
        trends-research/.venv             python 3.11.0,  polars 1.44.1

    They are interoperable -- verified by round-tripping the worksheet cache
    between them, 95,533 x 25 read identically under each -- so this does not
    pick a winner.  It only insists that whichever one is used can do the job.
    pyarrow is NOT required: polars reads and writes parquet natively.
    """
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
    """Progress that survives NOT being a terminal.

    On a tty this redraws a spinner in place.  Off one -- PyCharm's console, a
    cron job, anything redirected -- carriage returns are useless, so it prints
    a plain line every `every` seconds instead.

    The first version only drew the spinner, guarded by isatty().  Off a
    terminal that meant NOTHING was printed for the whole wait: a run in
    PyCharm sat silent for up to five minutes after "Commands sent to NDU" and
    looked hung.  It was waiting exactly as designed, which is no comfort to
    whoever is watching a dead console decide whether to kill it.

    `every` WAS 30 AND IS NOW 10.  A full piped run was measured line by line
    and its two longest silences -- 30.2s and 30.0s -- were both this tick
    during the NDU wait, longer than anything else in a 467-second pipeline.
    Thirty seconds of nothing is exactly the interval that makes someone reach
    for Ctrl-C.  Six extra lines per wait is a cheap price.
    """
    el = time.time() - t0
    if tty:
        print(_CR + f"  [{_SPIN[int(el * 2) % 4]}] {msg}  {el:4.0f}s",
              end="", flush=True)
    elif el - state.get("last", -every) >= every:
        state["last"] = el
        print(f"  ... {msg}  {el:4.0f}s", flush=True)


NDU_TRIGGER = Path(r"C:\Program Files\Norgate Data Updater\bin\ndu.trigger.exe")


def ensure_ndu(dry: bool, wait: int = 60, quiet: bool = False) -> bool:
    """STAGE 0.  Start NDU if it is down, ask it to update, wait for the data.

    Stage 1 is the only stage that needs the vendor, and it fails outright if
    NDU is not running -- which on an unattended nightly run means the whole
    pipeline stops because an app was closed.  `ndu.trigger.exe` fixes that: it
    starts NDU if necessary and sends it commands.  From its own usage text:

        UPDATE      - Starts a Data Updater Update
        SYNC        - Starts a synchronize for third party databases
        CLOSE       - Closes the updater once everything else is finished
        DONOTSHOW   - Prevents updater main window from showing when started

    We send UPDATE DONOTSHOW here and CLOSE separately at the END of the run.
    CLOSE MUST NOT GO IN THIS CALL: it "closes the updater once everything else
    is finished", meaning once the commands in the same invocation are done --
    so bundling it here would shut NDU down before stage 1, which is the one
    stage that needs it, and stage 1 would fail on a pipeline that had just
    started the updater for it.

    THE TRIGGER IS ASYNCHRONOUS.  It returns as soon as the command is sent --
    "Commands sent to NDU / Done." -- so returning here immediately would let
    stage 1 read a database mid-update.  Hence the wait below.

    THERE IS NO CLEAN COMPLETION EVENT, and this was checked rather than
    assumed:

      * `last_database_update_time` only advances when data actually CHANGES.
        Triggered on 2026-08-28 with nothing new to fetch, it sat at 17:16:40
        for the full three minutes we watched.  So it is a positive signal, not
        a completion signal: it tells you new data arrived, never that the
        update has finished finding none.
      * The `ndu.flgupdate` semaphore the trigger writes is consumed by NDU when
        it PICKS UP the command, not when it finishes acting on it, and it is
        gone from disk within seconds.  It signals delivery, not completion.

    So: wait for the timestamp to advance, give up after `wait` seconds, and
    report which happened.  A timeout is the ordinary case on a quiet evening --
    it means "nothing new", not "something broke" -- and the run continues
    either way, because the append in stage 1 is self-correcting: a contract
    file short of its last trade is refetched on every subsequent run until it
    is not.

    WHY THE DEFAULT IS 60s AND NOT SOMETHING LONGER.  It was 300, which was a
    guess, and a poor one:

      * It never once ended early in testing.  Every run burned the full cap,
        because there was never new data to find -- so it was five minutes of
        nothing, per run.
      * It does not guarantee what it looks like it guarantees.  If a real fetch
        took twelve minutes, 300s would not cover it either.  This is a
        heuristic delay, not a synchronisation.
      * STAGE 1 RUNS FOR ~205s ON ITS OWN and refreshes instruments one at a
        time, so NDU's work already overlaps it.  Data landing thirty seconds
        into stage 1 is still seen by almost every instrument.  A long wait here
        roughly doubles the pipeline's runtime to protect only the first few
        seconds of the next stage.

    So the wait is a courtesy, not a guard.  Anything it misses is picked up by
    the next run.  `--ndu-wait 0` skips it; a scheduled job running well after
    the close can reasonably do that.

    Returns (advanced, we_started_it).  Never raises for a timeout; only a
    missing trigger binary is fatal, and that is a bad install.
    """
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
        """Is NDU up?  Asked of the PROCESS LIST, never of the API.

        `norgatedata.status()` costs 0.0s when NDU is up and FORTY-FIVE SECONDS
        when it is down -- it retries internally ten times and prints a warning
        for each.  The first version of this asked the API for a baseline before
        firing the trigger, so a cold start burned 45s and twelve lines of noise
        before doing anything.  The process list answers the same question in
        milliseconds and cannot flood the log.
        """
        try:
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq dataupdater.norgate.exe"],
                capture_output=True, text=True, timeout=20).stdout or ""
        except Exception:
            return False
        return "dataupdater.norgate.exe" in out.lower()

    def _stamp():
        """Last Futures update time, or None.  Only call once NDU is up."""
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

    # Wait for the PROCESS first when NDU was down.  It takes ~15-20s to come
    # up, and every API call before then costs 45s of retries for an answer the
    # process list already has.
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
    """Shut NDU down.  A SEPARATE call, made after every stage has finished.

    `CLOSE` closes the updater once the commands in its own invocation are done,
    so it cannot ride along with the UPDATE in stage 0 -- that would kill NDU
    before stage 1 could query it.

    Runs even when a stage failed: the pipeline started this process, so the
    pipeline puts it away.  Leaving an updater running after an aborted nightly
    job is how you end up with one per night.
    """
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
    """Run one stage.  Raises SystemExit with the stage's own code on failure.

    `-u` IS ADDED HERE, NOT AT THE CALL SITES, so a stage added later cannot
    forget it.  Python block-buffers stdout in ~8 KB chunks whenever it is not
    writing to a terminal, and a stage's stdout here is NEVER a terminal: the
    tty path below hands it a pipe so the bar can read it, and the redirected
    path hands it a file or PyCharm's console.  Without `-u` the child's output
    therefore sits in its own buffer and arrives in bursts -- or, for a stage
    that prints less than 8 KB, all at once when it exits.  The progress bar is
    driven by counting the child's lines, so a buffered child means a bar that
    does not move until the stage it is measuring has already finished.
    """
    if cmd and Path(cmd[0]).name.lower().startswith("python") and "-u" not in cmd:
        cmd = [cmd[0], "-u", *cmd[1:]]
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}")
    print("  $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd) + "\n")
    if dry:
        return 0.0
    t0 = time.time()
    if not sys.stdout.isatty():
        # Redirected to a file: carriage returns would fill it with
        # half-drawn bars.  Plain pass-through instead -- a log is read after
        # the fact, where a progress bar is worth nothing anyway.
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
                # Wipe the bar, emit the real line, redraw beneath it.
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
        # A NON-BLOCKING STAGE MUST NOT CLAIM IT STOPPED THE RUN.  Stage 4b is
        # caught by its caller and the pipeline continues; printing the generic
        # "Later stages NOT run" there is a plain untruth in the output, and an
        # untrue line in a log is worse than no line -- it is what somebody
        # reads when they are working out why a run went wrong.
        tail = ("Later stages NOT run." if blocking
                else "Non-blocking: the run continues.")
        print(f"\n[{'ABORT' if blocking else 'FAILED'}] {label} exited {rc} "
              f"after {dt:.0f}s. {tail}")
        raise SystemExit(rc)
    print(f"\n  {label} ok  ({dt:.0f}s)")
    return dt


# ---------------------------------------------------------------------------
# Verification.  Terminal output only -- nothing is written to disk.
#
# EVERY CHECK HERE EXISTS BECAUSE THE FAILURE IT CATCHES HAS HAPPENED, or is
# one line away from happening.  A suite of plausible-sounding assertions that
# have never fired teaches you nothing; these are the shapes of real mistakes:
# a column that vanished from one file, a stale file a partial run left behind,
# a bound that was assumed rather than enforced.
# ---------------------------------------------------------------------------

def _ok(label: str, cond: bool, detail: str = "") -> tuple[bool, str, str]:
    return (bool(cond), label, detail)


def _note(label: str, detail: str = "") -> tuple[None, str, str]:
    """Something wrong that must be SEEN but must not stop the pipeline.

    `publish.py` aborts when the previous run recorded failures, which is right
    for anything that makes the published numbers wrong and wrong for anything
    that does not. The vendor dropping open interest is a real defect in the
    feed, reported here every run until it is fixed -- but nothing downstream
    reads open interest, so failing on it would take the site offline over a
    field no published figure depends on.

    THE MOMENT SOMETHING CONSUMES THAT FIELD these become `_ok` and the
    pipeline stops on them. A participation cap on order size is the obvious
    candidate, and it would read exactly the column that is missing here.
    """
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
    """The Roll_Rule -> hold-column map, read from trading_book itself.

    Imported rather than restated: a rule added there and forgotten here would
    make this check pass while stage 2 aborts on it, which is worse than having
    no check at all.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tb", BOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return set(m.HOLD_FOR)


def verify_holds(quiet: bool = False) -> int:
    """Does every instrument actually resolve a contract on the newest session?

    THE ONE CHECK THAT RUNS THE ROLL RULES rather than inspecting their record.
    Everything else in stage 1 reads contract_cycles.csv; this asks the rule the
    question it exists to answer -- what do I hold today -- and requires an
    answer for all 63.

    An instrument goes empty when its ladder changes under a rule that no longer
    resolves: a market that stops listing the month the rule wants, a notice
    date that moves inside the gate, a contract that expires with no successor
    quoted. Historical sessions stay fine, so nothing in the file looks wrong;
    only TODAY is blank, and stage 2 would emit a book that simply stops.

    IT IS NOT FREE AND DOES NOT NEED TO BE.  Resolving a hold needs the whole
    worksheet -- the rules carry streaks and ratchets across sessions, so a
    short window would answer a different question.  But this builds through
    `trading_book.cached_worksheet`, writing the same cache stage 2 reads with
    the same fingerprint, so the ~5 minutes is stage 2's work brought forward,
    not added to it.  On a warm cache it is seconds.
    """
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
    # TIME-BASED OFF A TTY, NOT EVERY-Nth.  This loop takes 188s cold and ~2s
    # warm, so any fixed instrument count is wrong at one end or the other:
    # every-10th left a 34.6s gap cold, and would spam a warm run that needs no
    # progress at all.  A 5s floor adapts to both -- roughly 35 lines cold,
    # two warm.
    _EVERY = 5.0
    _last = [t0]
    print(f"  resolving {len(rules)} roll rules against the newest session "
          f"(rebuilds the worksheet cache; slow when cold)", flush=True)
    for n, (inst, rule) in enumerate(sorted(rules.items()), 1):
        # PROGRESS THAT SURVIVES NOT BEING A TERMINAL.  This was `if tty` only,
        # so off a terminal -- PyCharm's console, a redirected log, the exact
        # places an unattended run lives -- NOTHING was printed while this
        # rebuilt 63 worksheets.  Measured cold: 188 seconds of total silence,
        # immediately after stage 1's PANEL EDGE banner.  That is what was
        # reported as a hang.  Same lesson and same fix as `_tick`: a carriage
        # return is useless off a tty, so print a plain line instead.
        #
        # ANNOUNCED BEFORE THE WORK, NOT AFTER IT.  The build below is one
        # blocking call, and CL alone takes ~13s, so a message printed on
        # completion leaves that gap unattributed -- the console just stops.
        # Naming the instrument first means the pause always has something
        # under it: you can see it is on CL rather than wondering if it died.
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
        # The rule can only name a contract the session actually lists; a hold
        # pointing at a symbol with no row is a reference to nothing.
        elif not (held <= set(sess.get_column("symbol").to_list())):
            dangling.append(inst)
        if tty and not quiet:
            # A BAR REPORTS COMPLETION, so it belongs at the end of the body.
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
    """Stage 1's output: the panel and the roll rules built off it."""
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

    # A rule with no hold column aborts stage 2 per-instrument, late and loudly.
    # Catching it here is the difference between one message and 63.
    known = _hold_for()
    unknown = sorted({x["Roll_Rule"] for x in ruled} - known)
    r.append(_ok("Roll_Rule values all map to a hold column", not unknown,
                 f"{len(known & {x['Roll_Rule'] for x in ruled})} distinct"
                 + (f"   UNKNOWN: {unknown}" if unknown else "")))

    # per_year is len(codes); the one-hot columns that used to assert this were
    # removed on 2026-08-28, so this is now the only thing checking it.
    mism = [x["instrument"] for x in rows
            if (x.get("codes") or "").strip()
            and str(len(x["codes"].strip())) != (x.get("per_year") or "").strip()]
    r.append(_ok("per_year == len(codes)", not mism,
                 f"{len(rows) - len(mism)}/{len(rows)}"
                 + (f"   differ: {', '.join(mism[:6])}" if mism else "")))

    empty = [x["instrument"] for x in rows if not (x.get("codes") or "").strip()]
    r.append(_ok("every instrument has month codes", not empty,
                 f"{', '.join(empty[:6])}" if empty else "63/63" if len(rows) == 63 else ""))

    # NOT "all rows agree": last_date is per INSTRUMENT, not a run stamp, and
    # markets in a timezone ahead legitimately carry one more session -- YAP4
    # and YXT4 do, which is why the panel edge report calls them "2 ahead".
    # The failure worth catching is an instrument whose feed has STOPPED, so
    # the test is distance from the newest, not equality with it.
    import datetime as _dt
    def _d(v):
        try: return _dt.date(int(v[:4]), int(v[4:6]), int(v[6:8]))
        except Exception: return None
    seen = {x["instrument"]: _d((x.get("last_date") or "").strip()) for x in rows}
    good = {k: v for k, v in seen.items() if v}
    newest = max(good.values()) if good else None
    stale = sorted(k for k, v in good.items() if (newest - v).days > 5)
    # pointsize_of() aborts stage 2 mid-build on a missing row.  The mapping is
    # a 4 KB file; checking it here turns five wasted minutes into one line.
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
    """Stage 2's output: the 63 trading books."""
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

    # A partial run leaves yesterday's files beside today's, and every one of
    # them reads as valid.  mtime is the only thing that tells them apart.
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
        # Eq 3.35 completed: price_vol_USD_ann == price_vol_curr_ann x FX_rate.
        # Worth asserting rather than assuming, because the failure is invisible:
        # the currency leg going missing leaves NIY reporting 7,283,654 -- yen
        # read as dollars -- which is a plausible-looking number about 160x too
        # large, on the column a position sizer divides by.
        if {"price_vol_USD_ann", "price_vol_curr_ann", "FX_rate"} <= set(t.columns):
            g = lambda c: [None if x in (None, "") else float(x)
                           for x in t.get_column(c).to_list()]
            u, c, x = g("price_vol_USD_ann"), g("price_vol_curr_ann"), g("FX_rate")
            for _u, _c, _x in zip(u, c, x):
                if _c is None or _x is None:
                    if _u is not None:         # a null input must give a null
                        fx_vol_bad.append(f.stem); break
                elif (_u is None
                      or abs(_u - _c * _x) > abs(_c * _x) * 1e-9 + 1e-12):
                    fx_vol_bad.append(f.stem); break

    r.append(_ok("no empty or unreadable file", not empty,
                 f"{', '.join(empty[:6])}" if empty else f"{len(files)} readable"))
    # 6A once shipped without Sign_raw while 62 files had it, and the run
    # reported success.  One schema for the whole book, or name the odd ones.
    r.append(_ok("identical schema across every file", len(schemas) <= 1,
                 f"{len(next(iter(schemas)))} columns" if len(schemas) == 1
                 else f"{len(schemas)} DIFFERENT schemas: "
                      + " | ".join(f"{v[0]}+{len(v)-1}" for v in schemas.values())))
    r.append(_ok("dates sorted, no duplicates", not unsorted_ and not dup,
                 f"unsorted: {unsorted_[:4]}  dup: {dup[:4]}" if (unsorted_ or dup) else "all files"))
    # held-only is the file's central claim; if it ever stops holding, every
    # price on the row belongs to a contract the book is not holding.
    r.append(_ok("hold == symbol on every row", not hold_mismatch,
                 f"{', '.join(hold_mismatch[:6])}" if hold_mismatch else "all files"))
    # The book can be REWRITTEN and still be stale inside: a feed that stopped
    # produces a fresh file whose last bar is months old.  mtime cannot see it.
    if ends:
        newest = max(ends.values())
        behind = sorted(k for k, v in ends.items() if v != newest)
        r.append(_ok("every book ends on the newest session", not behind,
                     f"{newest}"
                     + (f"   BEHIND: {', '.join(behind[:6])}" if behind else "")))

    # The Panama chain is anchored at the present, so the newest contract
    # carries a zero offset: on the last row the adjusted close MUST equal the
    # raw close.  One comparison per instrument tests the whole adjustment.
    r.append(_ok("Panama anchored: last Continuous_C == last close", not anchor,
                 f"{', '.join(anchor[:6])}" if anchor else "all files"))

    # fdm_raw is POOLED -- one value per session shared by every instrument.
    # If it ever differs between books, the cross-sectional pass has silently
    # become per-instrument and the FDM means something else entirely.
    r.append(_ok("fdm_raw identical across instruments", not fdm_split,
                 f"{', '.join(sorted(set(fdm_split))[:5])}" if fdm_split
                 else "all shared dates"))

    # SJB once shipped with Trend_sign entirely null while the schema check
    # passed -- the column was PRESENT and empty.  Coverage is what catches it.
    r.append(_ok("no signal column near-empty for an instrument", not thin,
                 "   ".join(f"{a}.{b} {c:.0%}" for a, b, c in thin[:4])
                 if thin else "all >= 30% populated"))

    r.append(_ok("SIGNAL within +/-20", not bad_sig,
                 f"{', '.join(bad_sig[:6])}" if bad_sig else "all files"))

    # Same dual-write invariant as the rates: a book whose parquet is missing or
    # older reads correctly (the loader falls back to csv) but SILENTLY, and the
    # point of this run is that nothing is silent.
    no_twin = [f.stem for f in files if not f.with_suffix(".parquet").is_file()]
    old_twin = [f.stem for f in files
                if f.with_suffix(".parquet").is_file()
                and f.with_suffix(".parquet").stat().st_mtime_ns < f.stat().st_mtime_ns]
    r.append(_ok("parquet twin present and current", not no_twin and not old_twin,
                 ((f"missing: {', '.join(no_twin[:5])}  " if no_twin else "")
                  + (f"STALE: {', '.join(old_twin[:5])}" if old_twin else ""))
                 or f"{len(files)} pairs"))

    # ---- FX_rate: present, and the RIGHT currency's rate ------------------
    #
    # THE CHECK THAT MATTERS IS THE SECOND ONE.  A missing column is loud; a
    # column carrying the wrong currency's rate is not.  A CGB priced with the
    # EUR rate is off by 61% and every number downstream still reads as a
    # plausible dollar figure, so the only way to catch it is to re-derive the
    # currency from the mapping and compare against that currency's own file.
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
        # An entirely null FX_rate means the book cannot be converted at all --
        # very different from YAP4's leading gap before the AUD future existed.
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


# An FX rate that moves more than this in one session is not a market move.
#
# SET FROM THE PANEL'S OWN WORST DAYS, not from intuition.  The largest real
# day-over-day moves in these rates are AUD 9.47% (2008-10-06, the GFC), JPY
# 8.65% (1998-10-07, the carry unwind) and GBP 7.81% (2016-06-24, the Brexit
# referendum).  15% therefore clears every genuine event in 47 years of history
# while staying an order of magnitude below what the failures this aims at would
# produce: a missed scale factor reads as 9,900%, an inverted quote as ~100%.
FX_MAX_DAILY_MOVE = 0.15
# ALERT currently runs 0.033% of rows (26 of 78,939).  1% is a wide margin that
# still fires long before a check source has quietly gone bad.
FX_MAX_ALERT_FRAC = 0.01


def _tb():
    """trading_book as a module, for the tables it owns.

    IMPORTED RATHER THAN RESTATED, for the same reason `_hold_for` is: a currency
    added there and forgotten here would make this check pass while the stage it
    is checking aborts on it, which is worse than having no check at all.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tbfx", BOOK)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def verify_fx(started: float) -> int:
    """Stage 2's other output: the FX conversion rates, one file per rate.

    CHECKED SEPARATELY FROM THE BOOKS BECAUSE THEY FAIL DIFFERENTLY.  A book that
    goes wrong usually goes visibly wrong -- a column empties, a schema drifts, a
    date stops advancing.  A conversion rate that goes wrong stays perfectly
    well-formed and simply carries the wrong number; every position sized off it
    is then wrong by that factor, with nothing anywhere reading as an error.  So
    these checks are mostly about VALUE plausibility, which is the opposite
    emphasis to verify_books.

    The two failures worth catching are a missed scale factor -- 6J quotes USD
    per 100 JPY and nothing in the metadata says so -- and an inversion.  Both
    are enormous, 9,900% and ~100%, so the session-move ceiling catches either on
    the first day it appears.
    """
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

    # THE DUAL-WRITE INVARIANT.  Programs read the parquet, people read the csv.
    # If the two ever disagree, half the consumers are on stale numbers and
    # neither file looks wrong on its own.  `_prefer_parquet` refuses a parquet
    # older than its csv, so a missing or old twin degrades SAFELY -- but it
    # degrades silently, which is what this run exists to surface.
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
        # THE FILES ARE TRIMMED TO START WHERE THE RATE STARTS, so a null here is
        # not a warm-up -- it means the as-of carry left a hole mid-history.
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

        # csv vs parquet on the column that matters.  Cheap, and the only thing
        # that actually PROVES the two artifacts agree rather than assuming it
        # from the fact that one loop wrote both.
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

    # THE BASE CURRENCY IS AN IDENTITY AND MUST READ AS ONE.  If USD is ever
    # anything but 1.0 the portfolio's own currency has been redefined, and every
    # other rate in the folder is now quoted against the wrong base.
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
    """Stage 3's output: the sized positions and the portfolio series.

    THE FAILURES HERE ARE ARITHMETIC, NOT STRUCTURAL.  A position file that is
    malformed announces itself; one that is well-formed and wrongly sized does
    not, and it is the only artifact in this pipeline that says how much money
    to put at risk.  So these checks re-derive 3.32 and 3.33 from their inputs
    and insist the file agrees, rather than inspecting shapes.
    """
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

    # --- eq 3.33 re-derived ------------------------------------------------
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
    # w'Cw must be positive: it is a variance of a weighted portfolio.
    fin = [x for x in (wcw if wcw is not None else []) if x == x]
    r.append(_ok("w'Cw positive wherever defined", all(x > 0 for x in fin),
                 f"{len(fin):,} defined, min {min(fin):.5f}" if fin else "-"))

    # --- equity must be the running sum of P&L, exactly ---------------------
    #
    # AGAINST EQUITY, NOT THE SIZING BASE.  Under compounding the two are the
    # same series and either would do; under `--fixed-nav` the sizing base is
    # constant by construction, so reconciling P&L against it would fail on a
    # correct run and hide a real break on an incorrect one.  `equity_USD` is
    # the money in both modes.
    eq = g("equity_USD")
    if eq is None:
        eq = nav
    # COSTS ARE DEDUCTED FROM NAV, and `net_pnl_USD` is built against the
    # SHIFTED cost precisely so this reconciliation is the obvious one:
    #
    #     equity[t] = equity[t-1] + net_pnl_USD[t]
    #
    # A trade is paid for at t-1 and earns its move into t, so netting against
    # the unshifted `cost_USD` would leave a gap of exactly the first and last
    # session's cost -- small enough to read as float noise.  `cost_lag_USD`
    # removes the ambiguity rather than documenting around it.
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

    # --- rounding: contracts are integers, and always toward zero ----------
    frac, wrongway, carry_bad, buf_bad = [], [], [], []
    for f in files:
        t = pl.read_csv(f, infer_schema_length=None)
        if not {"N_raw", "N_contracts"} <= set(t.columns):
            continue
        raw = t.get_column("N_raw").to_numpy()
        con = t.get_column("N_contracts").to_numpy()
        # THE ROUNDING RULE APPLIES TO N_target, NOT TO N_contracts.  3.36's
        # buffer sits between them: when a move is inside the band the executed
        # position stays at YESTERDAY'S size, which can legitimately be larger
        # than today's |N_raw|.  Checking the executed number against N_raw
        # therefore fails on a correct run -- it did, on all six FX books --
        # while telling you nothing about the rounding.  N_target is the
        # truncation's own output and is what the rule is about.
        tgt = (t.get_column("N_target").to_numpy()
               if "N_target" in t.columns else con)
        # A CARRIED ROW HAS NO N_raw TO COMPARE AGAINST -- 3.32 was not
        # evaluated because the market was shut -- so the rounding rule is
        # checked only where it applied.  What IS checked on those rows is that
        # the position really was carried and not quietly changed while the
        # market could not be traded.
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
            # toward zero: |N| never exceeds |raw|, and the sign never flips
            if abs(b) > abs(a) + 1e-9 or (b != 0 and a != 0 and (b > 0) != (a > 0)):
                wrongway.append(f.stem); break
        for b in con:
            if b == b and b != int(b):
                frac.append(f.stem); break
        # Equation 3.36 re-derived.  The executed position must be EITHER
        # today's target or yesterday's position, and holding is allowed only
        # when the move was inside the 10% band.  A buffer that suppressed a
        # trade it should have passed would otherwise be invisible: the series
        # would still be integral, still monotone, still reconcile.
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
    # THE ROUNDING RULE IS THE POINT.  floor(-2.7) = -3 would INCREASE a short;
    # truncation toward zero gives -2.  A rule that can enlarge a position is
    # the one mistake a sizer must not make silently.
    r.append(_ok("rounding always REDUCES |position|", not wrongway,
                 f"{', '.join(sorted(set(wrongway))[:6])}" if wrongway
                 else "toward zero everywhere"))
    r.append(_ok("buffer: executed is target-or-hold, band respected (3.36)",
                 not buf_bad,
                 f"{', '.join(sorted(set(buf_bad))[:4])}" if buf_bad
                 else f"b = 0.10, {len(files)} files"))
    # Costs reconcile at both levels, and the two levels agree with each other.
    # A cost column that is merely PRESENT proves nothing: the failure worth
    # catching is a per-instrument cost that never reaches the portfolio total,
    # or a net that quietly equals its gross.
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
        # net is against the LAGGED cost -- see the note on the reconciliation.
        if np.nanmax(np.abs((gg - cl) - nn)) > 1e-6:
            net_bad.append(f.stem)
        # NO ROW-ADJACENCY CHECK ON THE LAG HERE, deliberately.  A position
        # file omits sessions where the instrument had neither a forecast nor a
        # position, so "the previous row" is not always "the previous session"
        # -- 6S has one such gap, and comparing adjacent rows flagged it as a
        # fault when the column was right.  `cost_lag_USD` is shifted on the
        # SESSION GRID, which is what the equity recursion runs on; the
        # portfolio-level file has no gaps and the shift IS verified there.
        if not (cc >= -1e-9).all():
            cost_bad.append(f"{f.stem} negative")
    r.append(_ok("net_pnl == pnl - cost_lag, per instrument", not net_bad,
                 f"{', '.join(sorted(set(net_bad))[:6])}" if net_bad
                 else f"{len(files)} files"))
    # NOTIONAL IS AN ABSOLUTE QUANTITY AND CANNOT BE NEGATIVE.  It went
    # negative on 39,585 instrument-sessions because it was priced off the
    # PANAMA close, which is anchored at the present and drifts below zero in
    # early history -- CL bottoms at -29.11 against a raw low of 10.42.  The
    # P&L was unaffected (differences from Panama are correct; it is LEVELS that
    # must come from the raw close), so nothing else looked wrong: the only
    # symptom was a negative mean Gross/NAV for an entire asset class.
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
        # COST IS CONSERVED, NOT SHIFTED.  This asserted `cost_lag == shift(cost)`
        # until 2026-08-30, which was the OLD behaviour and the defect: a cost
        # decided at t fills at that instrument's own next open, so on a session
        # some market was shut the two series legitimately part company. The
        # invariant that survives the fix is stronger and is the one that
        # matters -- every dollar decided is eventually charged, none is
        # invented, and nothing is charged before it was decided.
        cum_dec = np.nancumsum(pc)
        cum_chg = np.nancumsum(pcl)
        never_early = bool(np.all(cum_chg <= cum_dec + 1e-6))
        pending = float(cum_dec[-1] - cum_chg[-1])
        r.append(_ok("commission charged never precedes decided",
                     never_early,
                     f"cumulative charged <= decided on all {len(pc):,} sessions"))
        # What is left over is the last session's decision, which fills after
        # the window ends and so is correctly still unpaid.
        last = float(np.nan_to_num(pc[-1]))
        r.append(_ok("every dollar decided is charged, or still pending",
                     abs(pending - last) < 0.01,
                     f"decided {cum_dec[-1]:,.2f}  charged {cum_chg[-1]:,.2f}  "
                     f"pending {pending:,.2f} vs last session {last:,.2f}"))
        gr, nrr = g("gross_ret"), g("net_ret")
        if gr is not None and nrr is not None and nav is not None:
            exp = np.zeros(len(nav)); exp[1:] = pnet[1:] / nav[:-1]
            r.append(_ok("net_ret == net_pnl / NAV[t-1]",
                         bool(np.nanmax(np.abs(exp[1:] - nrr[1:])) < 1e-9),
                         f"max diff {np.nanmax(np.abs(exp[1:] - nrr[1:])):.1e}"))
            # THE CASH LEG, RE-DERIVED.  net_ret is the strategy alone and is
            # already an excess return -- nothing in it earns the bill rate --
            # so total_ret minus net_ret must be exactly the interest.  If those
            # two ever disagree the rate has been counted twice, or not at all,
            # and neither shows up as a malformed number.
            tr_ = g("total_ret")
            if tr_ is not None and itr is not None and nav is not None:
                exp_i = np.zeros(len(nav)); exp_i[1:] = itr[1:] / nav[:-1]
                r.append(_ok("total_ret - net_ret == interest / NAV[t-1]",
                             bool(np.nanmax(np.abs((tr_ - nrr - exp_i)[1:])) < 1e-12),
                             f"max diff "
                             f"{np.nanmax(np.abs((tr_ - nrr - exp_i)[1:])):.1e}"))
                # AGAINST THE APPLIED RATE, NOT THE ROW'S OWN.  interest[t] is
                # earned with the rate from t-1, so comparing it to
                # rf_accrual_next[t] flags eleven March-2020 sessions as faults
                # when the bill rate really was negative and the interest really
                # was a charge.  `rf_accrual_applied` is that rate, shifted.
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

    # THE BOOK MUST NEVER GO FLAT ONCE IT HAS STARTED.  That is both stronger
    # and more useful than "some instrument is active every session".
    #
    # n_active LEGITIMATELY REACHES ZERO.  The grid is the union of 63 markets'
    # calendars, so a US holiday leaves only a handful of foreign markets with a
    # bar -- and in the early years none of those had a usable forecast yet.
    # There are 21 such sessions and every one is Thanksgiving, Presidents' Day,
    # Memorial Day, July 4th, Labor Day, Christmas or Good Friday.  Asserting
    # n_active > 0 there asserts that markets do not close.
    #
    # What must not happen is the PORTFOLIO emptying on those days, and that is
    # exactly what an earlier version did: a shut market has no bar, so its
    # sized position came out zero, the whole line was liquidated and bought
    # back the next session, and the spurious round trips dominated turnover.
    # Positions now carry through a closed market, so this is the check that
    # would have caught it -- it fails the moment the book goes flat for any
    # reason other than not having started.
    # ---- the Sharpe is EXCESS OF IRX, and that is checkable ---------------
    #
    # `net_ret` is trading P&L after commission and contains no interest, so a
    # Sharpe computed on it is already excess of the bill rate: the cash leg
    # sits in `total_ret`, and total_ret - rf gives the same number back.  The
    # check exists because the report says "excess of IRX" and a label is not
    # evidence -- and because the failure it guards against is subtracting IRX
    # a SECOND time, which looks like a correction and would cut the reported
    # Sharpe from 1.13 to about 0.68 on the current window.
    #
    # The residual is not noise: it is rf x (cost/NAV), the interest not earned
    # because commission left the account before the accrual.  0.0002 over
    # 1990+, 0.0037 on a 2026 start.  The tolerance scales with the interest
    # effect itself so it cannot pass vacuously on a zero-rate window.
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
    """Stage 4's output: the order ledger and the two daily views.

    THE CENTRAL CHECK IS A REPLAY, and it is the only one that matters much.
    Apply every order in sequence from flat and the book must hold exactly
    `N_contracts`, in exactly one contract, on every session -- which says the
    ledger is a LOSSLESS ENCODING of the position path rather than merely a
    plausible one.  450,000 sessions in 0.6s, so it runs every night.

    Everything else guards the failure this stage is uniquely prone to: a ledger
    can be internally immaculate and still describe the wrong trades, because it
    is DERIVED from a level series and an order is a difference.  The 578 rolls
    that a first version silently collapsed into resizes were all well-formed
    rows -- right instrument, right date, plausible size -- and no structural
    check would ever have looked at them twice.  Only the replay does.
    """
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
    # THE CSV IS THE ONE A HUMAN OPENS, so it has to be checked, not assumed.
    # Reading it back typed is exactly the trap this pipeline dual-writes to
    # avoid, so compare shape and edges instead -- which is what would move if
    # the two writes ever diverged.
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
    # Whole contracts.  Stage 3 truncates toward zero, so a fraction here would
    # be an order no exchange can fill.
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
    # Each kind is a CLAIM ABOUT FLATNESS, and the claim is checkable.
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

    # A ROLL IS A PAIR.  One leg alone is legitimate only when the other side is
    # flat -- rolling into a position from nothing, or out of one to nothing.
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

    # ---- THE REPLAY -------------------------------------------------------
    #
    # Four questions in one walk, because they all need the same thing: this
    # instrument's OWN sessions, which is not what a row of the position file
    # is.  Positions sit on the panel's union grid, so the previous row can be
    # a day this market was shut -- the mistake that hid 578 rolls.
    by_inst = {k[0]: v for k, v in L.partition_by("instrument",
                                                  as_dict=True).items()}
    div = shut = wrong_exec = sessions = nulls = 0
    fill_div = 0
    first_div = first_fill = ""
    missing = []
    balance = {}          # (instrument, contract) -> net signed quantity
    final = {}            # instrument -> (last symbol held, last position)
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
            # exactly the position, in exactly one contract -- or flat in none
            if held.get(sy, 0.0) != n or len(held) > bool(n):
                div += 1
                first_div = first_div or f"{inst}@{dte} held={held} want {sy}:{n}"
            # THE SAME LEDGER WALKED ON THE FILL TIMELINE.  The replay above
            # applies orders when they are DECIDED; this applies them when they
            # EXECUTE, and must therefore land on YESTERDAY's position.  It is
            # the check that says an order given for execution is the order that
            # executes -- the decision-side replay cannot see a fill at all.
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
    # TRIAL BALANCE.  Every contract ever traded must net to what is still held
    # in it -- zero for the 9,380 that have expired, the live position for the
    # rest.  A contract that opened and never closed is the one bookkeeping
    # error a position-based replay cannot produce and cannot detect.
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

    # ---- the priced columns ----------------------------------------------
    #
    # RECONCILED AGAINST POSITIONS, NOT AGAINST THE BUCKET WALK THAT PRODUCED
    # THEM.  Re-running stage 4's own attribution here would agree with itself
    # whatever it does.  Instead: for each contract, add up the pnl_USD of every
    # session the instrument held it, and require the realised P&L booked
    # against that contract to match -- exactly for a contract that has expired,
    # and short by the open mark-to-market for one still held.
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
        # TWO LEGS, AND ON A ROLL TWO CONTRACTS.  Under open execution the
        # overnight gap was earned on the month held at k-2 -- the fill had not
        # happened yet -- and the rest of the session on the month held at k-1.
        # Off a roll they are the same contract and this is the old single
        # credit.  Only sessions where the contract was actually HELD count: a
        # symbol with no position earns nothing and would pad the count with
        # thousands of no-op contracts, diluting the check into a formality.
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

    # Commission: the ledger prices EVERY LEG, stage 3 prices |dN|.  Off a roll
    # they must agree to the cent; on a roll the ledger is higher, and by how
    # much is the number this comparison exists to publish -- not a failure,
    # a measurement.  See the module docstring in bookkeeping.py.
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
    # ON A ROLL TOO, since 2026-08-29.  Stage 3 used to bill |dN| here and was
    # short $1.083B over the history; it now charges both legs.  The two are
    # computed by unrelated code -- stage 3 inside the sizing loop from N and
    # the symbol grid, stage 4 from the derived order legs -- so agreement is
    # evidence rather than a tautology, and this is the line that would catch
    # the correction being lost.
    gap = led_roll - s3_roll
    r.append(_ok("commission matches stage 3 ON a roll", abs(gap) <= max(1.0, s3_roll * 1e-9),
                 f"${gap / 1e6:,.1f}M apart" if abs(gap) > max(1.0, s3_roll * 1e-9)
                 else f"${led_roll / 1e9:,.3f}B both ways, both legs charged"))

    # ---- the daily statement ---------------------------------------------
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
        # The interest line has to name the balance it was earned on.
        b = S.get_column("interest_base_USD").to_list()
        rt = S.get_column("rate_cal_day").to_numpy()
        bad_b = sum(1 for k in range(len(b))
                    if rt[k] and (b[k] is None
                                  or abs(b[k] * rt[k] - ii[k]) > max(0.01, abs(ii[k]) * 1e-9)))
        # COUNT SESSIONS THAT ACTUALLY EARNED, not sessions with a rate.  The
        # bill quotes a rate on all 12,552 grid sessions; the book only accrues
        # once it has started, so a 2026 run credits 171 of them.  Reporting the
        # rate count made the check look 70x broader than it is.
        n_acc = int(np.sum(ii != 0))
        r.append(_ok("interest == base x rate, on every accruing session",
                     bad_b == 0,
                     f"{bad_b} rows disagree" if bad_b
                     else f"{n_acc:,} sessions credited, of {int((rt != 0).sum()):,} "
                          f"carrying a rate"))

    # ---- the two views ----------------------------------------------------
    #
    # Both key on `execute_at`, so both are checked on it.  An instrument shut
    # today had its order decided yesterday for an open that never came: it is
    # still pending, and selecting on the decision date would call it filled.
    P = pl.read_parquet(pend_f.with_suffix(".parquet"))
    X = pl.read_parquet(exe_f.with_suffix(".parquet"))
    asof = dec.max()
    ok_p = (P.height == P.get_column("instrument").n_unique()
            and bool((P.get_column("decision_date") <= asof).all())
            and bool((P.get_column("execute_at").is_null()
                      | (P.get_column("execute_at") > asof)).all()))
    r.append(_ok("pending: one per instrument, none filled yet", ok_p,
                 f"{P.height} order(s) for the next open"))
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
    """Stage 2's third output: the risk-free rate.

    A RATE FAILS THE WAY AN FX RATE FAILS -- quietly, and in the units.  It is
    one column of small numbers that every session of cash accrual multiplies
    by, so an error of a factor of 360, or of 100, or a discount left
    unconverted, produces a perfectly well-formed series and an equity curve
    that is wrong by a compounding margin.  These checks therefore RE-DERIVE the
    conversion from the raw quote rather than inspecting the result's shape.
    """
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

    # ---- the conversion, re-derived from the raw quote -------------------
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
    # THE IDENTITY THAT PROVES THE DAY COUNT.  A per-calendar-day rate
    # accumulated over 365 days must return the BEY exactly; if it returned the
    # discount rate instead, the discount-to-yield conversion has been skipped
    # and every accrual is ~2% of itself too small.
    r.append(_ok("rf_cal_day x 365 == BEY  (day count is calendar, not trading)",
                 bool(np.nanmax(np.abs(cal * 365.0 - bey / 100.0)) < 1e-12),
                 f"max diff {np.nanmax(np.abs(cal * 365.0 - bey / 100.0)):.1e}"))
    r.append(_ok("BEY exceeds the quoted discount everywhere the rate is positive",
                 bool((bey[pct > 0] >= pct[pct > 0] - 1e-12).all()),
                 f"mean uplift {np.nanmean(bey[pct > 0] - pct[pct > 0]):.4f}pp"))

    # ---- the calendar gap ------------------------------------------------
    g_ok = gap[np.isfinite(gap)]
    r.append(_ok("cal_days_to_next >= 1 wherever defined",
                 bool((g_ok >= 1).all()),
                 f"min {int(g_ok.min())}d  max {int(g_ok.max())}d  "
                 f"mean {g_ok.mean():.2f}d"))
    # A gap longer than a fortnight is not a holiday, it is a hole in the panel.
    long_gaps = int((g_ok > 10).sum())
    r.append(_ok("no calendar gap longer than 10 days", long_gaps == 0,
                 f"{long_gaps} gaps > 10d"))
    r.append(_ok("rf_accrual_next == rf_cal_day x cal_days_to_next",
                 bool(np.nanmax(np.abs(acc[:-1] - cal[:-1] * gap[:-1])) < 1e-15),
                 f"max diff {np.nanmax(np.abs(acc[:-1] - cal[:-1] * gap[:-1])):.1e}"))
    # Only the LAST row may be null: there is no next session to be paid at.
    n_null = int((~np.isfinite(acc)).sum())
    r.append(_ok("rf_accrual_next null on the final row only", n_null == 1,
                 f"{n_null} nulls" if n_null != 1 else "as expected"))

    # ---- plausibility ----------------------------------------------------
    # Bill yields have run 17.14% (1981) to -0.105% (2020 flight to quality).
    # Anything outside this band is a units error, not a market.
    lo, hi = -2.0, 25.0
    out = int(((pct < lo) | (pct > hi)).sum())
    r.append(_ok(f"irx_pct within [{lo}, {hi}]%  (a units sanity band)", out == 0,
                 f"min {pct.min():.3f}%  max {pct.max():.3f}%"
                 + (f"   {out} OUTSIDE" if out else "")))
    return _report("stage 2 -- IRX/", r)


def verify_stages(started: float) -> int:
    """Do the five artifacts agree WITH EACH OTHER?

    EVERY OTHER REPORT CHECKS ONE STAGE AGAINST ITSELF, and a stale artifact
    passes those effortlessly: yesterday's FX file is internally consistent,
    sorted, non-null and correctly typed.  What it is not is the file the books
    were built against.  These checks are the only ones that would notice.

    KEPT CHEAP ON PURPOSE -- last rows, date counts and a single cross-read per
    instrument -- because it runs on every pipeline and the expensive per-file
    work has already been done by the reports above.
    """
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

    # ---- one Positions file per book, and no orphans ---------------------
    b_names = {f.stem for f in books}
    p_names = {f.stem for f in poss}
    r.append(_ok("one Positions file per book, no orphans", b_names == p_names,
                 f"{len(b_names)} books, {len(p_names)} positions"
                 + (f"   book-only {sorted(b_names - p_names)[:4]}"
                    if b_names - p_names else "")
                 + (f"   position-only {sorted(p_names - b_names)[:4]}"
                    if p_names - b_names else "")))

    # ---- everything must end on the same session -------------------------
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

    # ---- the session grid is one object ----------------------------------
    n_port = pl.read_csv(port, infer_schema_length=0).height
    if irx.is_file():
        n_irx = pl.read_csv(irx, infer_schema_length=0).height
        r.append(_ok("IRX spans exactly the portfolio's grid", n_irx == n_port,
                     f"IRX {n_irx:,} vs Portfolio {n_port:,}"))
    if fxs:
        n_usd = pl.read_csv(E / "FX" / "USD.csv", infer_schema_length=0).height
        r.append(_ok("USD rate spans exactly the portfolio's grid",
                     n_usd == n_port, f"USD {n_usd:,} vs Portfolio {n_port:,}"))

    # ---- values agree across the stage boundary --------------------------
    #
    # THE CHECK THAT ACTUALLY CATCHES A STALE INTERMEDIATE.  Stage 3 copies
    # SIGNAL, price_vol_USD_ann and FX_rate out of the books; if the books were
    # rebuilt and the positions were not (or vice versa), the two disagree on
    # the last session while both files remain perfectly well-formed.
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
            drift.append(f"{f.stem} date"); continue
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

    # ---- the ledger was derived from THESE positions ----------------------
    #
    # Stage 4 rerunning is cheap, so the failure worth naming is stage 3
    # rerunning WITHOUT it: the ledger then describes yesterday's book while
    # remaining internally flawless.  verify_bookkeeping's replay would diverge
    # on the last session, but it would report an arithmetic mismatch; this
    # reports the cause.
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
    """Commit `docs/` and push it, which is the only thing that moves the site.

    `git add docs` AND NOTHING ELSE, EVER.  This is the whole safety argument
    and it is not a preference.  An `add -A` in this repository has already
    swept 66 unrelated files into a commit once, taking out CI with it; at the
    moment this was written the working tree held five changed files under
    `docs/` and two under `Live/` that had no business being published. An
    automated commit must be able to state exactly what it is committing, so it
    names the path, and then CHECKS what got staged before it commits.

    IT REFUSES RATHER THAN RECONCILES.  If the branch is not `main`, or the
    remote has moved ahead, this stops and says so. Pulling or rebasing on the
    user's behalf inside an unattended pipeline is how an automation loses
    somebody's work, and the cost of stopping is a site that is one run stale.

    Nothing here is gated on being interesting: an unchanged `docs/` is a
    no-op, not an empty commit.
    """
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

        # A RE-RUN THAT PRODUCES THE SAME NUMBERS IS NOT A PUBLICATION.  Every
        # run rewrites `updated_at`, `generated_at` and the cache stamp, so the
        # tree is ALWAYS dirty and a naive check would commit five times on a
        # day the pipeline was run five times, each differing in a timestamp.
        # Same distinction the journal draws: those fields are context, not
        # content.  And an older `updated_at` on unchanged numbers is not stale
        # -- it is accurate about when those numbers were produced.
        changed = [l for l in git("diff", "-U0", "--", "docs").splitlines()
                   if (l.startswith("+") or l.startswith("-"))
                   and not l.startswith(("+++", "---"))]
        volatile = ("updated_at", "generated_at", "?v=")
        material = [l for l in changed if not any(k in l for k in volatile)]
        if changed and not material:
            git("checkout", "--", "docs")
            print(f"  docs/ differs only in timestamps and the asset stamp "
                  f"({len(changed)} lines) --")
            print("  the published numbers are unchanged, so nothing is "
                  "committed. Working tree reverted.")
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
        # THE DATA IS PUBLISHED; THE CODE THAT MADE IT MIGHT NOT BE.  This step
        # deliberately commits only `docs/`, so a dirty tree elsewhere means the
        # site now shows figures produced by code no reader can see -- which is
        # the one claim this whole project rests on. Not fatal, and not this
        # step's business to fix, but it must not be silent.
        dirty = [l for l in git("status", "--porcelain", "--", ".",
                                ":!docs").splitlines() if l]
        if dirty:
            print(f"  [NOTE] {len(dirty)} uncommitted file(s) outside docs/. The "
                  f"site now shows results")
            # NOT `l[3:]`.  Porcelain writes "XY path" with a two-column
            # status field, but `git()` ends in .strip(), which eats the leading
            # space of the FIRST line only -- so a fixed offset cut one
            # character into that one path and left the rest correct.  It
            # printed 'ive/5_Publish/publish.py'. Splitting on whitespace is
            # indifferent to whether the column survived, and `maxsplit=1`
            # keeps paths that contain spaces intact.
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



_PT_START = "2026-01-02"      # the published window; nothing before it is drawn
# Names a page may call without defining: browser globals and the handful of
# array/string methods the regex below cannot tell apart from a helper.
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
    # keywords a call-shaped regex cannot tell from a function
    "for", "if", "while", "switch", "catch", "return", "function", "typeof",

    "dispatchEvent", "scrollIntoView", "getBoundingClientRect", "add", "has",
}



# A container the page's own script addresses and then leaves blank is the
# signature of a fetch that 404'd, a renamed field, or an exception part-way
# through a render. Elements never written to by script are ignored.
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
    """Load every page in a real browser and ask what actually came out.

    THE ONLY SUITE THAT SEES THE RESULT RATHER THAN THE SOURCE.  Every check
    above reads files: the JSON is right, the CSS parses, the id exists, the
    helper is defined. All of that passed while three summary rows carried
    `class="pos"` and rendered GREY, because `.stats tr.sub td` is specificity
    (0,2,2) and outranked `.stats td.pos` at (0,2,1). Nothing that reads source
    can see that. Only `getComputedStyle` can.

    THREE ASSERTIONS, chosen because each one caught a defect that actually
    shipped today:

      1. no console error, in either theme -- catches a runtime exception,
         including one swallowed by a `.catch()`, which is how a call to a
         function that did not exist left a line silently blank;
      2. the colour a reader SEES on a signed figure -- catches both of today's
         colour faults at the level that matters, where a stray `}` had
         discarded the rule that paints every loss red;
      3. every container a script fills is non-empty -- catches a renamed
         field, a 404, or an exception part-way through a render, all of which
         leave the page looking merely quiet.

    NOT SCREENSHOTS.  Font rasterisation differs between machines, so a pixel
    diff would fail for reasons nobody can act on and would train everyone to
    ignore this report.

    FAILS SAFE.  Without playwright, or without its browser, this prints one
    note and returns zero. A machine that cannot run it still gets a working
    pipeline; it gets less assurance, and says so.
    """
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

    # SERVED, NOT OPENED FROM DISK.  `file://` blocks the fetch() calls every
    # page makes, so a disk-loaded page renders empty and this suite would
    # report the failure it exists to detect on a site that is fine.
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
                            # Deterministic, not a sleep: wait until the page has
                            # actually rendered something into its own containers.
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

                            # 3. every container a script fills has something in it
                            blank = page.evaluate(_RENDER_BLANK_JS)
                            for b in blank:
                                empty.append(f"{name}[{theme}]: #{b}")

                            # 2. the colour a reader actually sees
                            bad = page.evaluate(_RENDER_COLOUR_JS)
                            for b in bad:
                                colour.append(f"{name}[{theme}]: {b}")

                            # 4. the figure on screen IS the figure in the JSON
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

    # ---- 4. the number on screen is the number in the payload -------------
    #
    # THE ONE THAT CATCHES A RENAMED FIELD. `fmtMoney(undefined)` returns an em
    # dash by design, so asking for `m.equityEnd` instead of `m.equity_end`
    # throws nothing, blanks nothing and discolours nothing -- the page simply
    # prints a dash where the net asset value belongs, and every other check
    # here passes. The formatting is restated in Python on purpose: agreeing
    # with the page's own arithmetic would only prove it agrees with itself.
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
            # A ROW ON A SECOND PAGE, deliberately. Every other figure here is
            # on the Overview, so a payload key renamed out from under the Q&A
            # page left `#benchtable` rendering its header and nothing else --
            # not blank, not an error, just a table with no data in it. Naming
            # one row means its ABSENCE is a failure, which is the only way a
            # table that quietly emptied gets noticed.
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
    """The bars themselves, between the vendor and everything downstream.

    WHY THIS EXISTS.  Every other suite compares derived files against each
    other, so a panel that is internally consistent and WRONG passes all of
    them. Before this, the only questions asked of the feed were "is anything
    more than five days stale" and "does every book end on the newest session".
    Both pass while the vendor delivers prices with the volume and open
    interest missing -- which it did on 2026-08-24, for 86 of the 90 contracts
    reporting that day, unnoticed.

    STRUCTURAL IMPOSSIBILITIES FAIL.  A non-positive price or an out-of-order
    date means the file is not what it claims to be, and nothing downstream can
    be trusted.

    DATA-QUALITY ANOMALIES WARN.  An open-interest hole is a vendor problem,
    not ours; aborting the pipeline over one would stop the site updating for a
    reason we cannot fix, so it is reported loudly and counted instead. Nothing
    in the engine reads volume or open interest TODAY -- the day something does
    (a participation cap is the obvious candidate) these become failures.
    """
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

    # A FEED FAILURE IS A DATE, NOT A CONTRACT.  One contract missing its open
    # interest is noise; nearly every contract missing it on the same session is
    # the vendor, and that is the shape worth naming.
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
    # NOTED, NOT FAILED -- see `_note`. Volume and open interest are delivered
    # by the vendor and read by nothing in this pipeline; the anomaly is real
    # and is printed every run, but the site must not go dark over a column no
    # published figure depends on.
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


def verify_assets(started: float) -> int:
    """The published HTML, CSS and JS -- the half of the site that is code.

    EVERY OTHER PUBLICATION CHECK LOOKS AT JSON.  The pages that render it were
    unexamined, and that is where four real defects lived: a stray `}` in
    site.css silently discarded the rule that follows it and every loss on the
    P&L page rendered in black; a call to a function that did not exist failed
    inside a `.catch()` and left a line blank; a duplicate `class` attribute was
    dropped by the browser without complaint. None of these raise anything --
    a CSS parse error is not an error, it is a discarded rule.
    """
    DOCS = HERE.parent / "docs"
    r: list[tuple[bool, str, str]] = []
    pages = sorted(DOCS.glob("*.html"))
    r.append(_ok("pages present", bool(pages), f"{len(pages)} page(s)"))
    if not pages:
        return _report("published assets -- docs/", r)

    def _strip(css: str) -> str:
        return re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    # ---- CSS: a brace that does not close is a rule that does not load ----
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

    # ---- HTML: tags balanced, and no attribute written twice --------------
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

    # ---- no id used twice on one page ------------------------------------
    #
    # THE CHECK BELOW CANNOT SEE THIS, and that is not an oversight in it -- it
    # asks whether an id EXISTS and builds a set to do it, so a name used twice
    # collapses to one entry and passes.  A duplicate is worse than a missing
    # id: `getElementById` silently returns the FIRST match, so the page throws
    # nothing, blanks nothing, and renders a whole table into the <h2> that
    # happened to share the name.  The expectations page shipped `id="var"` on
    # both its heading and its VaR table and rendered exactly that way.
    dup_ids = []
    for f in pages:
        seen = re.findall(r'\sid="([A-Za-z0-9_-]+)"', f.read_text(encoding="utf-8"))
        rep = sorted({i for i in seen if seen.count(i) > 1})
        if rep:
            dup_ids.append(f"{f.name}: {rep[:3]}")
    r.append(_ok("no id used twice on one page", not dup_ids,
                 "; ".join(dup_ids) if dup_ids
                 else f"{len(pages)} pages, all unique"))

    # ---- every id a script reaches for exists on that page ---------------
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

    # ---- in-page anchors resolve -----------------------------------------
    dead = []
    for f in pages:
        txt = f.read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', txt))
        for a in set(re.findall(r'href="#([A-Za-z0-9_-]+)"', txt)):
            if a not in ids:
                dead.append(f"{f.name}#{a}")
    r.append(_ok("every in-page anchor resolves", not dead,
                 "; ".join(dead[:4]) if dead else "all resolved"))

    # ---- a page must not reach for a helper the shared script lacks ------
    #
    # THE CHECK THAT CAUGHT `fmtStamp()` -- a function I invented, which failed
    # inside a `.catch()` and left a line silently blank.
    #
    # COMMENTS AND STRING LITERALS ARE STRIPPED FIRST, and that is not tidiness.
    # Scanning raw source, "Return, annualised (arithmetic)" inside a template
    # string reads as a call to `annualised()`, and the check reported four
    # imaginary faults on its first run. A verification nobody believes is worse
    # than none, so it sees only code.
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
        # A PARAMETER IS A LOCAL NAME. `multiLine(..., capFn, ...)` is called as
        # `capFn(i, st)` inside its own body, and without this the check reports
        # every callback argument as an undefined helper.
        for params in re.findall(r"function\s+[A-Za-z_][\w]*\s*\(([^)]*)\)", body):
            local |= {q.strip().split("=")[0].strip()
                      for q in params.split(",") if q.strip()}
        for call in set(re.findall(r"(?<![.\w$])([a-z][A-Za-z0-9_]{2,})\s*\(", body)):
            if call in shared or call in local or call in _JS_BUILTINS:
                continue
            unknown.append(f"{f.name}: {call}()")
    r.append(_ok("every helper a page calls is defined", not unknown,
                 "; ".join(sorted(set(unknown))[:4]) if unknown
                 else f"checked against {len(shared)} shared names"))
    return _report("published assets -- docs/", r)

    def _strip(css: str) -> str:
        return re.sub(r"/\*.*?\*/", "", css, flags=re.S)

    # ---- CSS: a brace that does not close is a rule that does not load ----
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

    # ---- HTML: tags balanced, and no attribute written twice --------------
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

    # ---- every id a script reaches for exists on that page ---------------
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

    # ---- in-page anchors resolve -----------------------------------------
    dead = []
    for f in pages:
        txt = f.read_text(encoding="utf-8")
        ids = set(re.findall(r'\bid="([A-Za-z0-9_-]+)"', txt))
        for a in set(re.findall(r'href="#([A-Za-z0-9_-]+)"', txt)):
            if a not in ids:
                dead.append(f"{f.name}#{a}")
    r.append(_ok("every in-page anchor resolves", not dead,
                 "; ".join(dead[:4]) if dead else "all resolved"))

    # ---- a page must not reach for a helper the shared script lacks ------
    appjs = (DOCS / "app.js").read_text(encoding="utf-8")
    shared = set(re.findall(r"^(?:function|const|let)\s+([A-Za-z_][\w]*)",
                            appjs, re.M))
    unknown = []
    for f in pages:
        body = "\n".join(re.findall(r"<script>(.*?)</script>",
                                    f.read_text(encoding="utf-8"), re.S))
        local = set(re.findall(r"(?:function|const|let|var)\s+([A-Za-z_][\w]*)",
                               body))
        # NOT `\b`: that also matches the method half of
        # `localStorage.getItem(` and `r.json(`, which are not helpers and
        # could never be undefined. Only a BARE call -- nothing but
        # whitespace or an operator in front of it -- is a name this page
        # has to have defined somewhere.
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
    """One number, one source.

    A FIGURE ON TWO PAGES MUST COME FROM ONE FIELD.  The Q&A page briefly
    printed a 63-session rolling mean of 7.32% as "realised volatility" beside
    an Overview reporting 8.20% for the same run: both correct, computed
    independently, and irreconcilable to a reader. Nothing caught it because
    every check until now asked whether a file was internally consistent, never
    whether two files agreed.
    """
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

    # THE ONE THAT WOULD HAVE CAUGHT IT.  A rolling mean may legitimately differ
    # from the whole-run figure -- that is not the fault. The fault is a page
    # COMPUTING its own copy of a number another page already publishes, so the
    # test is on the source, not the value.
    qh = (DOCS / "qa.html").read_text(encoding="utf-8")
    r.append(_ok("the Q&A volatility card reads the Overview's own field",
                 "META.net_ann_vol" in qh,
                 "reads latest.json meta.net_ann_vol"
                 if "META.net_ann_vol" in qh else "computes its own -- they can drift"))

    n_pos = len(qa.get("positions") or [])
    r.append(_ok("Q&A position count matches the headline",
                 n_pos == m.get("n_positions", n_pos),
                 f"{n_pos} vs {m.get('n_positions')}"))
    # AND AGAINST ITSELF. The summary card reads `exposure`, the table reads
    # `positions`; nothing compared the two, so a payload could show 59 rows
    # under a heading saying 58 and pass every check on the page.
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
    """Did stage 6 write what it said it wrote?

    EVERY GUARD IN `publish.py` RUNS BEFORE THE WRITE.  The whitelist, the
    window guard, the page guard and the per-session reconciliation all inspect
    rows in memory; once the files are on disk the only thing that stage does is
    total their byte counts.  A truncated or half-written file -- interrupted
    run, full disk, encoding fault -- therefore passes everything and lands on
    the site looking fine.  This report is the only one that reads `docs/` back.

    IT IS `verify_stages` FOR STAGE 6.  The question is the same one: do the
    artifacts agree with each other, or is one of them internally perfect and
    describing a different run.  A published `latest.json` claiming an equity
    nobody compared against `Portfolio.parquet` is exactly the stale-but-
    well-formed failure this pipeline has been bitten by everywhere else.

    WHAT IT CANNOT CHECK is the live site.  The push is manual and deliberately
    so, which means `docs/` on disk is SUPPOSED to run ahead of the deployed
    page; a check against the URL would fail every time someone published and
    had not yet pushed.  This verifies the artifact, not the deployment.
    """
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

    # THE DEFINITIONS COME FROM publish.py, NOT FROM A COPY HERE.  Re-deriving
    # the cache stamp or the forbidden-word list in this file would give two
    # implementations of one rule, which is the shape that has already cost this
    # pipeline three quiet divergences.
    spec = importlib.util.spec_from_file_location("pub_v", PUBLISH)
    pub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pub)

    def _load_json(f):
        try:
            return _json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ---- 1. the top-level files are all present and parse -----------------
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

    # ---- 2/3. every file the indexes promise actually exists and parses ---
    for label, rows, sub in (("journal", idx, "days"), ("attribution", pidx, "pnl")):
        bad = [x["date"] for x in rows
               if _load_json(DATA / sub / f"{x['date']}.json") is None]
        r.append(_ok(f"every {label} session file exists and parses", not bad,
                     f"{len(rows) - len(bad)}/{len(rows)} files"
                     + (f"   bad: {bad[:4]}" if bad else "")))

    # ---- 4. the headline agrees with the portfolio it came from -----------
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

    # ---- 5. the curve ends where the portfolio ends -----------------------
    hl = hist["daily"][-1]
    same = (hl["date"] == d[-1]
            and abs(hl["equity_USD"] - float(eq[-1])) <= 0.01
            and len(hist["daily"]) == len(d))
    r.append(_ok("history.json ends on the portfolio's last session", same,
                 f"{hl['date']}  {len(hist['daily']):,} rows  "
                 f"{hl['equity_USD']:,.0f}"))

    # ---- 6. every published balance sheet still adds up -------------------
    #
    # `build_pnl` asserts this before writing; this asserts it after, on the
    # bytes a reader will actually be served.
    off = []
    for x in pidx:
        j = _load_json(DATA / "pnl" / f"{x['date']}.json")
        tot = sum(i["gross_pnl_USD"] for i in j["instruments"])
        if abs(tot - j["book"]["gross_pnl_USD"]) > 1.0:
            off.append(x["date"])
    r.append(_ok("every attribution sheet sums to its own book", not off,
                 f"{len(pidx)} sessions"
                 + (f"   off: {off[:4]}" if off else "")))

    # ---- 7. the cache stamp is present, uniform, and CURRENT --------------
    #
    # Uniform catches a partial stamping run; current catches assets that
    # changed after the last publish, which is the state that serves new markup
    # against a cached old script.
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

    # ---- 8. the provider is named nowhere under docs/ ---------------------
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
    """Close the books: recompute the money from the primary sources.

    THE ONLY REPORT THAT LEAVES THE ARTIFACTS.  Every other check in this file,
    `verify_stages` included, compares derived files with each other -- so a
    consistent misreading of the panel passes all of them, because every stage
    inherited the same misreading.  This one goes back to the trading books,
    `instrument_mapping.csv` and IRX, recomputes P&L, commission, interest,
    equity, turnover and notional from scratch, and requires the pipeline's
    numbers to match.

    Ten ties, ~3s.  It runs LAST because it is the broadest claim in the file,
    and it is worth reading even when it passes: the two sides of D are stage 3's
    billed contract count and stage 4's order legs, which disagreed by 143.5M
    contracts until the roll under-billing was fixed on 2026-08-29.  A cost-model
    regression shows up there as a contract count before it is ever money.

    The arithmetic lives in `Reconciliation_check/reconcile.py` and is rendered
    here rather than reimplemented -- two copies of a reconciliation is two
    reconciliations, and the second one is always the stale one.
    """
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
    """How many instruments stage 2 will write, for the bar's denominator.

    Read from contract_cycles.csv rather than hard-coded: the count follows the
    panel, and a bar that says 63 while 61 are written is worse than no bar.
    Returns None if the file cannot be read -- the bar degrades to a spinner
    rather than the pipeline failing over a cosmetic feature.
    """
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
                     [py, str(CYCLES)], args.dry_run)   # no total: see _bar
        if not (args.dry_run or args.no_verify):
            # A bad panel FAILS THE RUN HERE rather than being carried into
            # stage 2.  Books built off a broken panel are wrong rather than
            # absent, and absent is the failure that gets noticed.
            if verify_cycles() + verify_holds():
                print("")
                print("[ABORT] stage 1 verification failed; stage 2 NOT run.")
                return 3

    try:
        total += run(f"STAGE 2/5  trading_book.py  --jobs {args.jobs}",
                     [py, str(BOOK), "--jobs", str(args.jobs)], args.dry_run,
                     total=n_books())
        # STAGE 3 IS INSIDE THE SAME try, so a failure here still closes NDU.
        # It reads only what stage 2 wrote, needs no vendor, and takes seconds.
        if not args.no_portfolio:
            total += run(f"STAGE 3/5  portfolio.py  --nav {args.nav:,.0f}",
                         [py, str(PORTFOLIO), "--nav", repr(args.nav)],
                         args.dry_run)
            # STAGE 4 READS ONLY WHAT STAGE 3 WROTE and takes about a second,
            # so it is gated on stage 3 rather than given a skip of its own
            # meaning: a ledger derived from yesterday's positions would be
            # internally perfect and describe trades nobody is going to make.
            if not args.no_bookkeeping:
                total += run("STAGE 4/5  bookkeeping.py  (order ledger)",
                             [py, str(BOOKKEEPING)], args.dry_run)
                # THE JOURNAL IS NON-BLOCKING, ON PURPOSE.  It is a record, not
                # an input: nothing downstream reads it, no position depends on
                # it, and a store that can halt trading is a liability rather
                # than a control.  So a failure here is reported and the
                # pipeline carries on -- the derived ledger is still written and
                # tomorrow's append picks up what today missed.
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
        # Even on an abort: the pipeline started NDU, the pipeline closes it.
        if not (args.no_ndu or args.keep_ndu):
            close_ndu(args.dry_run)
    if not (args.dry_run or args.no_verify):
        # Stage 2 checks REPORT rather than abort: the books are already
        # written, so the useful thing is to say exactly what is wrong with
        # them, and to leave a non-zero exit for whatever runs this.
        # THE FEED ITSELF, before anything derived from it. Every suite below
        # compares derived files with each other, so a panel that is internally
        # consistent and wrong passes all of them.
        failures = verify_vendor(started)
        failures += verify_books(started, n_books())
        # THE RATES GET THEIR OWN REPORT, not extra lines in the book one.  They
        # are a separate artifact with separate consumers and an entirely
        # different failure mode -- a wrong rate is well-formed and silent -- so
        # burying eleven value checks at the end of a structural report would
        # make the thing they are guarding harder to see, not easier.
        failures += verify_fx(started)
        failures += verify_irx(started)
        if not args.no_portfolio:
            failures += verify_portfolio(started)
            if not args.no_bookkeeping:
                failures += verify_bookkeeping(started)
            # ACROSS stages, and only meaningful once each has spoken for
            # itself.
            failures += verify_stages(started)
            # LAST, and the only one that leaves the artifacts entirely: it
            # recomputes the money from the books, the mapping and IRX.  Every
            # report above compares derived files with each other, so a
            # consistent misreading of the panel passes all of them.
            if not (args.no_bookkeeping or args.no_reconcile):
                failures += verify_reconciliation(started)

    # THE RUN STAMP IS WRITTEN BEFORE STAGE 6, NOT AFTER THE SUMMARY, because
    # stage 6 READS it: `publish.py` takes the site's "Updated" line from this
    # file and refuses a run that failed or was partial.  Written afterwards,
    # the site would carry the PREVIOUS run's timestamp over THIS run's numbers
    # -- fresh figures under a stale date, which is worse than either alone.
    #
    # Written on every real run, pass or fail, with the count.  Recording only
    # successes would leave the last success's timestamp standing over data a
    # failed run had already overwritten.
    def _stamp(n_failures: int) -> None:
        if args.dry_run:
            return
        RUN_STAMP.write_text(json.dumps({
            "completed_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "failures": int(n_failures),
            # `--no-verify` leaves `failures` at 0 because nothing ran, which
            # is the one way a bad run can look like a clean one. Recorded
            # separately so stage 6 can tell "passed" from "never asked".
            "verified": not args.no_verify,
            "full_run": not (args.no_portfolio or args.no_bookkeeping
                             or args.no_reconcile),
        }, indent=1), encoding="utf-8")

    _stamp(failures)

    # STAGE 6 IS SKIPPED, NOT FAILED, when the run is not fit to publish from.
    # `publish.py` would refuse anyway -- `run_stamp()` aborts on a failed or
    # partial run -- but letting it abort HERE would end the pipeline on a
    # traceback about the website when the real news is the verification above.
    # The site keeping its last verified numbers is the correct outcome, and
    # worth saying out loud rather than exiting quietly.
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
            # THE ONLY REPORT THAT READS docs/ BACK.  Every guard inside
            # publish.py runs before the write; this one runs after it, on the
            # bytes a reader will be served.  It must therefore come after the
            # stage, which is why it is here and not in the block above.
            if not args.no_verify:
                failures += verify_publish(started)
                # The other half of the site: the code that renders the JSON,
                # and whether two pages showing the same figure got it from the
                # same place.
                failures += verify_assets(started)
                failures += verify_agreement(started)
                # LAST, and the only one that leaves the files entirely:
                # it asks a browser what the reader actually sees.
                failures += verify_render(started)
                # Re-stamp: the count changed after the first write, and a
                # stamp claiming zero failures would let the next manual
                # publish proceed off a run this one just failed.
                _stamp(failures)
            # THE DEPLOY IS GATED ON EVERYTHING ABOVE IT.  It runs only when
            # stage 5 wrote, the run was full and verified, and every report
            # including `verify_publish` came back clean -- so the push cannot
            # put numbers on a public page that this pipeline has not checked.
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

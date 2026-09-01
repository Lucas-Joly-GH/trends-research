import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk

if getattr(sys, "frozen", False):
    BASE = pathlib.Path(sys.executable).resolve().parent
else:
    BASE = pathlib.Path(__file__).resolve().parent
REPO = BASE.parent
PY = REPO / ".venv" / "Scripts" / "python.exe"
UPDATE = BASE / "Update.py"
LOGDIR = BASE / "logs"

PHASES = [
    (re.compile(r"^\s*NDU\s+\(start"), "Starting the data updater", 0.0, 0.12),
    (re.compile(r"^\s*STAGE 1/5"), "Stage 1 — roll cycles", 0.12, 0.70),
    (re.compile(r"^\s*VERIFY\s+stage 1"), "Checking stage 1", 0.70, 0.74),
    (re.compile(r"^\s*STAGE 2/5"), "Stage 2 — trading books", 0.74, 0.80),
    (re.compile(r"^\s*STAGE 3/5"), "Stage 3 — portfolio", 0.80, 0.84),
    (re.compile(r"^\s*STAGE 4/5"), "Stage 4 — order ledger", 0.84, 0.87),
    (re.compile(r"^\s*STAGE 4b"), "Stage 4b — journal", 0.87, 0.89),
    (re.compile(r"^\s*VERIFY\s+vendor bars"), "Verifying the books", 0.89, 0.94),
    (re.compile(r"^\s*STAGE 5/5"), "Stage 5 — publishing", 0.94, 0.97),
    (re.compile(r"^\s*VERIFY\s+publication"), "Checking the site", 0.97, 0.99),
    (re.compile(r"^\s*DEPLOY"), "Deploying", 0.99, 1.00),
]
SUB = re.compile(r"^\s*\[\s*(\d+)/\s*(\d+)\]")
DONE = re.compile(r"pipeline complete in (.+)$")
SUITE = re.compile(r"^\s*(\d+)/(\d+) passed")
# La confirmation de mise en ligne, telle que Update.py la journalise.
LIVE = re.compile(r"^\s*\[LIVE\]\s*(.+?)\s*$")
BAD = re.compile(r"\[FAIL\]|\[ABORT\]|Traceback")
# Les marches qui n'ont pas avance, et pourquoi. Update.py imprime cette
# ligne apres le releve du panel ; la fenetre la retient pour le resume,
# sinon elle defile et personne ne la voit.
HOLD = re.compile(r"^\s*\[HOLD\]\s*(.+?)\s*$")


class App:
    def __init__(self, root):
        self.root = root
        root.title("trends-research — daily run")
        root.geometry("620x300")
        root.minsize(560, 280)
        self.q = queue.Queue()
        self.rc = None
        self.phase_i = -1
        self.sub_seen = 0
        self.checks = 0
        self.failed_checks = 0
        self.live = ""
        self.summary = ""
        self.held = ""
        self.logpath = None
        self.proc = None

        pad = dict(padx=18, fill="x")
        tk.Label(root, text="Daily pipeline", font=("Segoe UI", 15)).pack(
            anchor="w", pady=(16, 0), **pad)
        self.phase = tk.Label(root, text="Starting…", font=("Segoe UI", 10),
                              anchor="w", fg="#444")
        self.phase.pack(anchor="w", pady=(2, 8), **pad)

        self.bar = ttk.Progressbar(root, mode="determinate", maximum=1000)
        self.bar.pack(pady=(0, 6), **pad)

        self.detail = tk.Label(root, text="", font=("Consolas", 8), anchor="w",
                               fg="#777", justify="left")
        self.detail.pack(anchor="w", **pad)

        self.result = tk.Label(root, text="", font=("Segoe UI", 20, "bold"))
        self.result.pack(pady=(14, 0))
        self.note = tk.Label(root, text="", font=("Segoe UI", 9), fg="#555",
                             justify="center")
        self.note.pack()

        row = tk.Frame(root)
        row.pack(side="bottom", pady=12)
        self.logbtn = tk.Button(row, text="Open log", width=12,
                                state="disabled", command=self.open_log)
        self.logbtn.pack(side="left", padx=6)
        self.closebtn = tk.Button(row, text="Cancel", width=12,
                                  command=self.on_close)
        self.closebtn.pack(side="left", padx=6)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        threading.Thread(target=self.run, daemon=True).start()
        root.after(80, self.pump)

    def run(self):
        if not PY.exists():
            self.q.put(("fatal", f"No interpreter at {PY}"))
            return
        LOGDIR.mkdir(exist_ok=True)
        import datetime
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.logpath = LOGDIR / f"run_{stamp}.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if hasattr(
            subprocess, "CREATE_NO_WINDOW") else 0
        try:
            self.proc = subprocess.Popen(
                [str(PY), str(UPDATE)], cwd=str(REPO), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=flags)
        except Exception as e:
            self.q.put(("fatal", str(e)))
            return
        with self.logpath.open("w", encoding="utf-8", buffering=1) as fh:
            for line in self.proc.stdout:
                fh.write(line)
                self.q.put(("line", line.rstrip("\n")))
        self.q.put(("exit", self.proc.wait()))

    def pump(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "line":
                    self.on_line(payload)
                elif kind == "exit":
                    self.finish(payload)
                    return
                elif kind == "fatal":
                    self.finish(-1, payload)
                    return
        except queue.Empty:
            pass
        self.root.after(80, self.pump)

    def on_line(self, line):
        for i, (pat, label, lo, hi) in enumerate(PHASES):
            if pat.search(line) and i > self.phase_i:
                self.phase_i = i
                self.sub_seen = 0
                self.phase.config(text=label)
                self.bar["value"] = lo * 1000
                break
        m = SUB.match(line)
        if m and 0 <= self.phase_i < len(PHASES):
            lo, hi = PHASES[self.phase_i][2], PHASES[self.phase_i][3]
            self.sub_seen += 1
            # Stage 1 walks the universe twice, so the denominator is doubled.
            total = int(m.group(2)) * (2 if self.phase_i == 1 else 1)
            frac = min(self.sub_seen / total, 1.0)
            self.bar["value"] = (lo + (hi - lo) * frac) * 1000
        s = SUITE.match(line)
        if s:
            self.checks += int(s.group(2))
            self.failed_checks += int(s.group(2)) - int(s.group(1))
        h = HOLD.match(line)
        if h:
            self.held = h.group(1)
        v = LIVE.match(line)
        if v:
            self.live = v.group(1)
        d = DONE.search(line)
        if d:
            self.summary = d.group(1).strip()
        txt = line.strip()
        if txt:
            self.detail.config(text=txt[:96])

    def finish(self, rc, fatal=""):
        self.rc = rc
        ok = rc == 0
        # SUCCESS NE S'AFFICHE QUE SI LE SITE SERT DEJA CETTE EXECUTION.
        # Avant, la fenetre disait SUCCESS des que la poussee etait partie :
        # on refermait, on ouvrait le site, et il montrait encore la veille.
        # Le code 3 dit « rien n'a echoue, mais ce n'est pas encore en ligne »
        # -- ni vert ni rouge, parce que ce n'est ni l'un ni l'autre et qu'il
        # n'y a rien a corriger, seulement a attendre.
        pending = rc == 3
        self.bar["value"] = 1000
        self.phase.config(text="")
        self.detail.config(text="")
        self.result.config(
            text="SUCCESS" if ok else "NOT LIVE YET" if pending else "FAILED",
            fg="#176b45" if ok else "#8a6d1f" if pending else "#9b2226")
        if pending:
            bits = []
            if self.summary:
                bits.append(f"completed in {self.summary}")
            if self.checks:
                bits.append(f"{self.checks} checks passed")
            txt = "   ·   ".join(bits)
            txt += chr(10) + "Everything ran and the push went out, but the "
            txt += "site was not serving it yet."
            if self.live:
                txt += chr(10) + self.live
            if self.held:
                txt += chr(10) + self.held
            self.note.config(text=txt)
            if self.logpath and self.logpath.exists():
                self.logbtn.config(state="normal")
            self.closebtn.config(text="Close")
            return
        if fatal:
            self.note.config(text=fatal)
        elif ok:
            bits = []
            if self.summary:
                bits.append(f"completed in {self.summary}")
            if self.checks:
                bits.append(f"{self.checks} checks passed")
            txt = "   ·   ".join(bits)
            if self.live:
                txt += chr(10) + self.live
            if self.held:
                txt += chr(10) + self.held
            self.note.config(text=txt)
        else:
            self.note.config(
                text=f"exit code {rc}"
                     + (f"   ·   {self.failed_checks} check(s) failed"
                        if self.failed_checks else "")
                     + ((chr(10) + self.held) if self.held else "")
                     + "\nOpen the log to see what went wrong.")
        if self.logpath and self.logpath.exists():
            self.logbtn.config(state="normal")
        self.closebtn.config(text="Close")

    def open_log(self):
        if self.logpath and self.logpath.exists():
            os.startfile(str(self.logpath))

    def on_close(self):
        if self.rc is None and self.proc and self.proc.poll() is None:
            self.proc.terminate()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.3)
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

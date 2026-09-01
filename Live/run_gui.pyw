"""La fenetre du pipeline quotidien.

TROIS CHOSES QUE CETTE FENETRE DOIT FAIRE, dans cet ordre : dire ou en est
l'execution, dire ce qu'elle fait, dire comment elle a fini. Le reste est
decoration, et se juge a ce qu'il retire plutot qu'a ce qu'il ajoute.

LA BARRE EST CALIBREE SUR DES MESURES, PAS SUR DES INTUITIONS.  L'ancienne
version donnait a l'etape 1 la portion 0,12 -> 0,70 de la barre. Depouillement
de treize executions : l'etape 1 prend 85 % du temps (215 s sur 252 s). La
barre rampait donc jusqu'a 70 % pendant trois minutes et demie, puis avalait
ses trente derniers pour cent en trente-cinq secondes -- le defaut classique
de la barre qui ment, et qui apprend a ne plus la regarder.

Les poids viennent maintenant de `logs/timings.json`, reecrit apres chaque
execution a partir du temps reellement passe dans chaque etape, chronometre
ici. La barre se recalibre donc seule : si le poste change ou si l'univers
grandit, elle suit sans qu'on y touche.
"""
import json
import os
import pathlib
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
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
TIMINGS = LOGDIR / "timings.json"
SITE = "https://lucas-joly-gh.github.io/trends-research/"

# LA PALETTE EST CELLE DU THEME SOMBRE DU SITE, recopiee.  L'outil qui publie
# et la chose publiee se ressemblent, ce qui n'est pas qu'un agrement : on
# passe de l'un a l'autre toute la journee, et deux gris differents pour le
# meme role donnent l'impression de deux logiciels sans rapport.
PAGE = "#0b0b0b"
PANEL = "#131313"
INK = "#bcbcbc"
MUTED = "#8a8a8a"
FAINT = "#565656"
ACCENT = "#e6e6e6"
POS = "#35c46b"
NEG = "#ff5c5c"
WARN = "#d9a441"
INFO = "#5cc8d8"
RULE = "#2a2a2a"

MONO = ("Consolas", 9)
UI = ("Segoe UI", 10)

# Les etapes, avec leur poids EN SECONDES.  Repli du premier lancement
# seulement : `timings.json` prend le relais des qu'il existe.
#
# CES VALEURS SONT MESUREES A LA PENDULE, ET LA PREMIERE SERIE NE L'ETAIT PAS.
# Elle etait tiree des durees que les journaux impriment (« ok (210s) »), qui
# ne couvrent que les cinq etapes numerotees : le demarrage du fournisseur de
# donnees, les verifications et le deploiement n'y figuraient pas, et je les
# avais estimes. Mal, d'un facteur dix pour certains -- 6 s devines contre 70 s
# reels pour NDU, 6 contre 46 pour la verification du site, 60 contre 104 pour
# le deploiement, qui attend desormais que Pages serve la publication. Ces
# quatre-la font a elles seules la moitie de l'attente. Total suppose : 5 min ;
# reel : 8 min 11 s.
#
# UN SEUL RELEVE, donc, et assume comme tel : il vaut infiniment mieux que des
# suppositions, et la mediane glissante le corrige des la deuxieme execution.
# `deploy` est le plus variable des onze -- il depend de la latence du CDN de
# GitHub, observee entre 56 s et 104 s.
#
# Le dernier champ dit combien de fois l'etape parcourt l'univers : l'etape 1
# le fait deux fois, donc ses sous-lignes « [n/63] » vont jusqu'a 126.
PHASES = [
    ("ndu",       re.compile(r"^\s*NDU\s+\(start"),        "Data updater",      70, 1),
    ("STAGE 1/5", re.compile(r"^\s*STAGE 1/5"),            "Roll cycles",      209, 2),
    ("v1",        re.compile(r"^\s*VERIFY\s+stage 1"),     "Checking stage 1",   6, 1),
    ("STAGE 2/5", re.compile(r"^\s*STAGE 2/5"),            "Trading books",     10, 1),
    ("STAGE 3/5", re.compile(r"^\s*STAGE 3/5"),            "Portfolio",          5, 1),
    ("STAGE 4/5", re.compile(r"^\s*STAGE 4/5"),            "Order ledger",       2, 1),
    ("STAGE 4b",  re.compile(r"^\s*STAGE 4b"),             "Journal",            3, 1),
    ("vbars",     re.compile(r"^\s*VERIFY\s+vendor bars"), "Verifying books",   21, 1),
    ("STAGE 5/5", re.compile(r"^\s*STAGE 5/5"),            "Publishing",        16, 1),
    ("vpub",      re.compile(r"^\s*VERIFY\s+publication"), "Checking the site", 46, 1),
    ("deploy",    re.compile(r"^\s*DEPLOY"),               "Deploying",        104, 1),
]

SUB = re.compile(r"^\s*\[\s*(\d+)/\s*(\d+)\]")
DONE = re.compile(r"pipeline complete in (.+)$")
SUITE = re.compile(r"^\s*(\d+)/(\d+) passed")
LIVE = re.compile(r"^\s*\[LIVE\]\s*(.+?)\s*$")
HOLD = re.compile(r"^\s*\[HOLD\]\s*(.+?)\s*$")
# LES DUREES SONT CHRONOMETREES ICI, pas lues dans le flux. Update.py
# n'annonce « ok (210s) » que pour les cinq etapes numerotees : les
# verifications, NDU et surtout le deploiement -- qui attend desormais
# que le site serve la publication, soit pres d'une minute -- n'impriment
# rien de tel. Se fier au texte laisserait donc un cinquieme de la barre
# a jamais non calibre. Le temps entre deux changements d'etape, lui,
# existe pour toutes, et c'est de toute facon celui qu'on attend devant
# l'ecran.

# Le classement d'une ligne pour la coloration, dans l'ordre ou il faut
# l'essayer : le premier motif qui matche gagne. L'echec passe avant tout le
# reste, parce qu'une ligne d'echec contient souvent le mot « OK » d'a cote.
TAGS = [
    ("bad",  re.compile(r"\[FAIL\]|\[ABORT\]|Traceback|VERIFICATION FAILURE")),
    ("warn", re.compile(r"\[WARN\]|\[NOTE\]|\[HOLD\]|\[SKIP\]")),
    ("info", re.compile(r"\[LIVE\]|\[WATCH\]")),
    ("good", re.compile(r"\[OK\s*\]")),
    ("head", re.compile(r"^\s*(STAGE|VERIFY|DEPLOY|NDU)\b")),
    ("rule", re.compile(r"^={10,}")),
    ("dim",  re.compile(r"^\s*\.\.\.")),
]


def load_weights():
    """Les poids d'etape : mesures si on en a, valeurs de repli sinon."""
    base = {k: float(w) for k, _, _, w, _ in PHASES}
    try:
        seen = json.loads(TIMINGS.read_text(encoding="utf-8"))
        for k, vals in seen.items():
            if k in base and vals:
                base[k] = float(statistics.median(vals))
    except Exception:
        pass
    return base


def save_timings(observed):
    """Ajouter les durees de cette execution, en gardant les dix dernieres.

    Une mediane sur dix executions absorbe le coup de froid occasionnel --
    un cache vide, un reseau lent -- sans mettre un mois a suivre un vrai
    changement de regime. Une calibration ratee n'est pas une panne : on
    n'en fait jamais une erreur visible.
    """
    if not observed:
        return
    try:
        LOGDIR.mkdir(exist_ok=True)
        try:
            hist = json.loads(TIMINGS.read_text(encoding="utf-8"))
        except Exception:
            hist = {}
        for k, v in observed.items():
            hist.setdefault(k, []).append(round(float(v), 1))
            hist[k] = hist[k][-10:]
        TIMINGS.write_text(json.dumps(hist, indent=1), encoding="utf-8")
    except Exception:
        pass


# Marque la place, dans la note, de la phrase qui dira ce que le panneau
# montre -- decidee plus tard, quand on sait si un marqueur d'echec existe.
FOCUS_PLACEHOLDER = "<focus>"


def human(s):
    s = max(0, int(round(s)))
    return f"{s}s" if s < 60 else f"{s // 60}m {s % 60:02d}s"


ICON_B64 = (
    "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"
)


def window_icon(root):
    """Poser l'icone du site sur la fenetre, a la place de la plume de Tk.

    EN BASE64 PLUTOT QU'EN FICHIER JOINT.  Un .ico embarque avec --add-data
    demanderait de resoudre son chemin, qui differe entre le script et le
    binaire gele : un chemin de plus a se tromper, pour une image de 2 Ko.
    Ici l'icone voyage dans le code et ne peut pas manquer a l'appel.

    La reference est gardee sur `root` : Tk ne retient pas ses PhotoImage,
    et une image collectee laisse une icone vide sans rien signaler.
    """
    try:
        img = tk.PhotoImage(data=ICON_B64)
        root.iconphoto(True, img)
        root._icon = img
    except Exception:
        pass

def dark_titlebar(root):
    """Assombrir la barre de titre, que Tk ne peint pas.

    La barre de titre appartient au gestionnaire de fenetres, pas au
    programme : un bandeau blanc restait donc pose sur une application
    entierement sombre. C'est DWM qui l'expose, et il faut le demander en
    deux temps parce que Windows a change d'API en cours de route --
    l'attribut 20 depuis la 2004, l'attribut 19 avant. On tente les deux et
    on garde celui qui repond.

    Sur Windows 11 on va plus loin : 34/35/36 posent la couleur exacte du
    cadre, du fond et du titre, ce qui evite le gris standard du mode sombre
    a cote de notre propre noir. Les versions plus anciennes refusent ces
    attributs, sans consequence : le mode sombre generique a deja fait
    l'essentiel.

    Rien ici ne peut faire echouer le lancement -- c'est de la peinture.
    """
    try:
        import ctypes
        root.update_idletasks()          # le handle n'existe qu'apres
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) \
            or root.winfo_id()
        dwm = ctypes.windll.dwmapi
        on = ctypes.c_int(1)
        for attr in (20, 19):            # USE_IMMERSIVE_DARK_MODE
            if dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(on),
                                         ctypes.sizeof(on)) == 0:
                break
        # COLORREF est 0x00BBGGRR, l'inverse de l'ordre habituel.
        rgb = lambda h: ctypes.c_int(int(h[5:7] + h[3:5] + h[1:3], 16))
        for attr, col in ((34, PAGE), (35, PAGE), (36, ACCENT)):
            v = rgb(col)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v),
                                      ctypes.sizeof(v))
    except Exception:
        pass


class App:
    def __init__(self, root):
        self.root = root
        root.title("trends-research — daily run")
        root.geometry("980x680")
        root.minsize(760, 520)
        root.configure(bg=PAGE)

        self.q = queue.Queue()
        self.rc = None
        self.phase_i = -1
        self.sub_seen = 0
        self.checks = 0
        self.failed_checks = 0
        self.live = ""
        self.summary = ""
        self.held = ""
        # Ce que le panneau montre finalement : ecrit apres coup, parce que la
        # note se compose avant qu'on sache si un marqueur d'echec existe. La
        # note annoncait « filtered to the failures » meme quand rien n'avait
        # ete filtre, ce qui laissait chercher un filtre absent.
        self.focus_note = ""
        self.logpath = None
        self.proc = None
        self.t0 = time.time()
        self.lines = []           # tout le flux, pour le filtre et la copie
        self.follow = True
        # Vrai pendant NOS PROPRES defilements. `see("end")` declenche le meme
        # rappel qu'un coup de molette : sans ce drapeau la fenetre prenait son
        # auto-defilement pour un geste de l'utilisateur et coupait le suivi des
        # la premiere ligne. « paused » s'affichait sans que personne n'ait
        # touche a rien, et le flux se figeait en haut du panneau.
        self._auto = False
        self.observed = {}        # durees mesurees pendant CETTE execution
        self.phase_t0 = self.t0

        self.weights = load_weights()
        self.spans = self._spans()

        self._style()
        self._build()
        window_icon(root)
        dark_titlebar(root)

        threading.Thread(target=self.run, daemon=True).start()
        root.after(80, self.pump)
        root.after(250, self.tick)

    # ----------------------------------------------------------- calibration
    def _spans(self):
        """Convertir les poids en secondes en bornes [lo, hi] sur la barre."""
        tot = sum(self.weights.get(k, w) for k, _, _, w, _ in PHASES) or 1.0
        out, acc = [], 0.0
        for k, _, _, w, _ in PHASES:
            d = self.weights.get(k, w)
            out.append((acc / tot, (acc + d) / tot))
            acc += d
        return out

    def total_estimate(self):
        return sum(self.weights.get(k, w) for k, _, _, w, _ in PHASES)

    # ----------------------------------------------------------------- style
    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")       # le seul theme ttk vraiment recolorable
        except tk.TclError:
            pass
        # Une barre pleine et verte au-dessus du mot FAILED est un contresens :
        # c'est le plus grand objet de la fenetre, et il dirait le contraire du
        # verdict. Une variante par issue, posee a la fin.
        for name, col in (("dark", POS), ("bad", NEG), ("warn", WARN)):
            st.configure(f"{name}.Horizontal.TProgressbar",
                         troughcolor=PANEL, background=col, bordercolor=RULE,
                         lightcolor=col, darkcolor=col, thickness=10)
        # L'ascenseur reste un tk.Scrollbar et non un ttk : sous « clam » ses
        # deux fleches gardent le gris clair du systeme quoi qu'on configure,
        # et deux carres blancs dans un coin sombre se voient plus que tout le
        # reste. Le widget classique accepte ses couleurs directement.

    def _lab(self, parent, text, font=UI, fg=INK, **kw):
        return tk.Label(parent, text=text, font=font, fg=fg, bg=parent["bg"],
                        anchor="w", justify="left", **kw)

    def _build(self):
        r = self.root
        head = tk.Frame(r, bg=PAGE)
        head.pack(fill="x", padx=20, pady=(16, 0))
        self._lab(head, "Daily pipeline", ("Segoe UI", 16)).pack(side="left")
        self.clock = self._lab(head, "", ("Consolas", 10), MUTED)
        self.clock.pack(side="right")

        self.phase = self._lab(r, "Starting…", ("Segoe UI", 10), MUTED)
        self.phase.pack(fill="x", padx=20, pady=(4, 6))

        self.bar = ttk.Progressbar(r, mode="determinate", maximum=1000,
                                   style="dark.Horizontal.TProgressbar")
        self.bar.pack(fill="x", padx=20)

        # LA LISTE DES ETAPES EST LA VRAIE GRANULARITE.  Une barre dit « 62 % »,
        # ce qui ne se verifie pas ; une liste dit quelle etape tourne, lesquelles
        # sont finies et en combien de temps -- elle rend donc la barre
        # controlable au lieu d'etre a croire sur parole.
        strip = tk.Frame(r, bg=PAGE)
        strip.pack(fill="x", padx=20, pady=(10, 4))
        self.steps = []
        for i, (_, _, label, _, _) in enumerate(PHASES):
            cell = tk.Frame(strip, bg=PAGE)
            cell.grid(row=i // 6, column=i % 6, sticky="w", padx=(0, 14), pady=1)
            dot = tk.Label(cell, text="·", font=("Consolas", 11), fg=FAINT,
                           bg=PAGE, width=2)
            dot.pack(side="left")
            txt = tk.Label(cell, text=label, font=("Segoe UI", 8), fg=FAINT,
                           bg=PAGE)
            txt.pack(side="left")
            self.steps.append((dot, txt))

        # -------------------------------------------------------- le terminal
        tools = tk.Frame(r, bg=PAGE)
        tools.pack(fill="x", padx=20, pady=(8, 2))
        self._lab(tools, "Output", ("Segoe UI", 9), MUTED).pack(side="left")
        self.followbtn = tk.Button(
            tools, text="following", font=("Segoe UI", 8), bg=PANEL, fg=POS,
            relief="flat", bd=0, padx=8, activebackground=RULE,
            activeforeground=ACCENT, command=self.toggle_follow)
        self.followbtn.pack(side="right")
        self.filter = tk.Entry(tools, font=MONO, bg=PANEL, fg=INK, bd=0,
                               insertbackground=ACCENT, width=26,
                               highlightthickness=1, highlightbackground=RULE,
                               highlightcolor=MUTED)
        self.filter.pack(side="right", padx=8, ipady=3)
        self.filter.bind("<KeyRelease>", lambda _e: self.rerender())
        self._lab(tools, "filter", ("Segoe UI", 8), FAINT).pack(side="right")

        wrap = tk.Frame(r, bg=RULE)
        wrap.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        self.term = tk.Text(wrap, bg=PANEL, fg=INK, font=MONO, bd=0,
                            padx=10, pady=8, wrap="none", state="disabled",
                            insertbackground=ACCENT, selectbackground=RULE,
                            selectforeground=ACCENT, height=12)
        self.sb = tk.Scrollbar(wrap, orient="vertical", command=self.on_scroll,
                               bg=PANEL, troughcolor=PAGE, activebackground=MUTED,
                               highlightthickness=0, bd=0, width=12,
                               relief="flat", elementborderwidth=0)
        self.term.configure(yscrollcommand=self.on_yview)
        self.term.pack(side="left", fill="both", expand=True, padx=1, pady=1)
        self.sb.pack(side="right", fill="y")
        for tag, colour in (("good", POS), ("bad", NEG), ("warn", WARN),
                            ("info", INFO), ("head", ACCENT), ("rule", RULE),
                            ("dim", FAINT), ("plain", INK)):
            self.term.tag_configure(tag, foreground=colour)
        self.term.tag_configure("head", font=("Consolas", 9, "bold"))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>", "<Prior>",
                    "<Next>", "<Up>", "<Down>", "<Home>", "<End>"):
            self.term.bind(seq, self.user_scrolled, add="+")

        # ----------------------------------------------------------- le pied
        foot = tk.Frame(r, bg=PAGE)
        foot.pack(fill="x", padx=20, pady=(0, 14))
        self.result = tk.Label(foot, text="", font=("Segoe UI", 15, "bold"),
                               bg=PAGE, fg=INK)
        self.result.pack(side="left")
        self.note = tk.Label(foot, text="", font=("Segoe UI", 9), fg=MUTED,
                             bg=PAGE, justify="left", anchor="w")
        self.note.pack(side="left", padx=14)

        btns = tk.Frame(foot, bg=PAGE)
        btns.pack(side="right")
        self.closebtn = self._btn(btns, "Cancel", self.on_close)
        self.logbtn = self._btn(btns, "Open log", self.open_log, "disabled")
        self.copybtn = self._btn(btns, "Copy output", self.copy_out)
        self.sitebtn = self._btn(btns, "Open site", self.open_site)
        r.protocol("WM_DELETE_WINDOW", self.on_close)

    def _btn(self, parent, text, cmd, state="normal"):
        b = tk.Button(parent, text=text, font=("Segoe UI", 9), bg=PANEL,
                      fg=INK, relief="flat", bd=0, padx=12, pady=4,
                      activebackground=RULE, activeforeground=ACCENT,
                      disabledforeground=FAINT, state=state, command=cmd)
        b.pack(side="right", padx=4)
        return b

    # ------------------------------------------------------------ defilement
    # LE SUIVI SE DECIDE SUR LE GESTE, PAS SUR LA POSITION.  Premiere version :
    # `on_yview` comparait la position a 1.0 et coupait le suivi des qu'on
    # n'etait plus tout en bas. Mais Tk appelle `yscrollcommand` au repos, APRES
    # que `see("end")` a rendu la main -- notre propre defilement revenait donc
    # par le meme chemin qu'un coup de molette, et le suivi se coupait tout seul
    # a la premiere ligne. Un drapeau « c'est moi qui defile » ne repare rien :
    # il est deja retombe quand le rappel arrive.
    #
    # On ecoute donc les gestes eux-memes -- molette, ascenseur, touches de
    # navigation -- et on relit la position APRES coup, au repos, quand elle est
    # a jour. Se decrocher en remontant, se raccrocher en revenant en bas.
    def on_yview(self, lo, hi):
        self.sb.set(lo, hi)

    def on_scroll(self, *args):
        self.term.yview(*args)
        self.user_scrolled()

    def user_scrolled(self, _e=None):
        self.root.after_idle(self._settle)

    def _settle(self):
        at_end = self.term.yview()[1] >= 0.999
        if at_end != self.follow:
            self.follow = at_end
            self._paint_follow()

    def toggle_follow(self):
        self.follow = not self.follow
        if self.follow:
            self.see_end()
        self._paint_follow()

    def _paint_follow(self):
        self.followbtn.config(text="following" if self.follow else "paused",
                              fg=POS if self.follow else WARN)

    # -------------------------------------------------------------- processus
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

    def tick(self):
        if self.rc is not None:
            return
        el = time.time() - self.t0
        p = self.bar["value"] / 1000.0
        # L'ESTIMATION SE CORRIGE SUR LE REEL.  Tant qu'on a trop peu avance
        # pour extrapoler, on affiche la duree calibree ; passe 8 %, le rythme
        # observe pendant CETTE execution est un meilleur juge que la mediane
        # historique -- lui seul sait que le cache est froid aujourd'hui.
        eta = el * (1 - p) / p if p > 0.08 else max(
            0.0, self.total_estimate() - el)
        self.clock.config(text=f"{human(el)} elapsed   ·   ~{human(eta)} left")
        self.root.after(250, self.tick)

    # ----------------------------------------------------------------lecture
    def on_line(self, line):
        self.lines.append(line)
        if self.passes_filter(line):
            self.append(line)

        for i, (_, pat, label, _, _) in enumerate(PHASES):
            if pat.search(line) and i > self.phase_i:
                now = time.time()
                if 0 <= self.phase_i < len(PHASES):
                    self.observed[PHASES[self.phase_i][0]] = now - self.phase_t0
                self.phase_t0 = now
                self.mark_done(self.phase_i)
                self.phase_i = i
                self.sub_seen = 0
                self.phase.config(text=label)
                self.bar["value"] = self.spans[i][0] * 1000
                dot, txt = self.steps[i]
                dot.config(text="▸", fg=ACCENT)
                txt.config(fg=ACCENT)
                break

        m = SUB.match(line)
        if m and 0 <= self.phase_i < len(PHASES):
            lo, hi = self.spans[self.phase_i]
            self.sub_seen += 1
            total = int(m.group(2)) * PHASES[self.phase_i][4]
            frac = min(self.sub_seen / max(total, 1), 1.0)
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

    def mark_done(self, i, ok=True):
        if not (0 <= i < len(self.steps)):
            return
        dot, txt = self.steps[i]
        secs = self.observed.get(PHASES[i][0])
        dot.config(text="✓" if ok else "×", fg=POS if ok else NEG)
        txt.config(fg=MUTED if ok else NEG,
                   text=PHASES[i][2] + (f"  {secs:.0f}s" if secs else ""))

    # --------------------------------------------------------------- terminal
    def classify(self, line):
        for tag, pat in TAGS:
            if pat.search(line):
                return tag
        return "plain"

    def passes_filter(self, line):
        f = self.filter.get().strip().lower()
        return not f or f in line.lower()

    def see_end(self):
        self._auto = True
        try:
            self.term.see("end")
        finally:
            self._auto = False

    def append(self, line):
        self.term.config(state="normal")
        self.term.insert("end", line + "\n", self.classify(line))
        self.term.config(state="disabled")
        if self.follow:
            self.see_end()

    def rerender(self):
        self.term.config(state="normal")
        self.term.delete("1.0", "end")
        for l in self.lines:
            if self.passes_filter(l):
                self.term.insert("end", l + "\n", self.classify(l))
        self.term.config(state="disabled")
        if self.follow:
            self.see_end()

    def copy_out(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append("\n".join(self.lines))
            self.copybtn.config(text="copied")
            self.root.after(1200,
                            lambda: self.copybtn.config(text="Copy output"))
        except tk.TclError:
            pass

    def open_site(self):
        import webbrowser
        webbrowser.open(SITE)

    # -------------------------------------------------------------------- fin
    def finish(self, rc, fatal=""):
        self.rc = rc
        ok = rc == 0
        # SUCCESS NE S'AFFICHE QUE SI LE SITE SERT DEJA CETTE EXECUTION.
        # Le code 3 dit « rien n'a echoue, mais ce n'est pas encore en ligne »
        # -- ni vert ni rouge, parce que ce n'est ni l'un ni l'autre et qu'il
        # n'y a rien a corriger, seulement a attendre.
        pending = rc == 3
        # La derniere etape n'a pas de suivante pour la clore : on la ferme ici,
        # sans quoi le deploiement -- justement celui qu'on veut mesurer --
        # manquerait a chaque calibration.
        if 0 <= self.phase_i < len(PHASES):
            self.observed[PHASES[self.phase_i][0]] = time.time() - self.phase_t0
        self.mark_done(self.phase_i, ok or pending)
        self.bar.config(style=("dark" if ok else "warn" if pending else "bad")
                              + ".Horizontal.TProgressbar")
        self.bar["value"] = 1000
        self.phase.config(text="")
        self.clock.config(text=f"{human(time.time() - self.t0)} total")
        if ok or pending:
            save_timings(self.observed)

        self.result.config(
            text="SUCCESS" if ok else "NOT LIVE YET" if pending else "FAILED",
            fg=POS if ok else WARN if pending else NEG)

        bits = []
        if self.summary:
            bits.append(f"completed in {self.summary}")
        if self.checks:
            bits.append(f"{self.checks} checks passed")
        if fatal:
            txt = fatal
        elif pending:
            txt = "   ·   ".join(bits) + chr(10) \
                + "Everything ran and the push went out; the site was not " \
                  "serving it yet."
        elif ok:
            txt = "   ·   ".join(bits)
        else:
            txt = f"exit code {rc}" + (
                f"   ·   {self.failed_checks} check(s) failed"
                if self.failed_checks else "") + chr(10) \
                + FOCUS_PLACEHOLDER
        if self.live and not fatal:
            txt += chr(10) + self.live
        if self.held and not fatal:
            txt += chr(10) + self.held
        self.note.config(text=txt)

        if not ok and not pending and self.lines:
            # Amener l'echec sous les yeux plutot que de demander de le
            # chercher : c'est la seule raison pour laquelle on revient a ce
            # panneau apres coup.
            #
            # MAIS ON NE FILTRE QUE SUR CE QUI EXISTE. Poser « FAIL » quand
            # aucune ligne ne le contient -- un plantage avant les controles,
            # une sortie non nulle sans verdict -- vidait entierement le
            # panneau : on lisait FAILED au-dessus d'un cadre noir, et le
            # premier reflexe etait de croire la sortie perdue. Le pire ecran
            # possible, precisement au pire moment.
            # « error » figurait dans cette liste et a ete retire : il
            # attrapait les [SKIP] anodins d'une dependance optionnelle et
            # presentait donc, sous le mot FAILED, sept lignes sans rapport
            # avec la panne. Un filtre qui montre la mauvaise chose est pire
            # qu'aucun filtre -- a defaut de marqueur franc, on montre tout et
            # on descend a la fin, ou la cause se trouve presque toujours.
            for probe in ("FAIL", "ABORT", "Traceback"):
                if any(probe.lower() in l.lower() for l in self.lines):
                    self.filter.delete(0, "end")
                    self.filter.insert(0, probe)
                    self.rerender()
                    self.focus_note = f"Output filtered to {probe}."
                    break
            else:
                self.term.see("end")
                self.focus_note = ("No failure marker in the output; showing "
                                   "all of it, at the end.")
            self.note.config(text=self.note.cget("text").replace(
                FOCUS_PLACEHOLDER, self.focus_note))

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

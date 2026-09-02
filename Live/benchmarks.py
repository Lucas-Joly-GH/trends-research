"""Le classement des strategies etudiees.

Une comparaison n'a de sens que si tout le monde est mesure au meme metre.
Le site publie deja quatre series -- le livre, l'achat-conservation sur le
meme univers, le S&P 500 et le melange hors actions -- et il les mesure sur
les rendements EXCEDENTAIRES QUOTIDIENS de 1998 a 2025 : ecart-type annualise
par racine de 256, Sharpe = moyenne x 256 / vol, et perte maximale lue sur
une valeur liquidative qui, elle, porte les interets. Recalculer tout cela
autrement -- en mensuel, par exemple -- donnerait des chiffres voisins mais
differents, et le logiciel contredirait le site sans que personne comprenne
pourquoi. On reprend donc exactement la meme mesure, et pour les quatre
series du site on lit meme directement les statistiques publiees.

Ajouter une strategie : deposer un fichier dans Live/Benchmarks/, ou le
fabriquer depuis une simulation de portefeuille avec

    python benchmarks.py --add Portefeuille.parquet --key ewmac_rapide
                         --name "EWMAC 16-64 seul"

ce qui applique la meme mesure aux rendements de cette simulation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# Une fois embarque dans l'executable, `__file__` pointe vers le dossier
# temporaire ou PyInstaller deplie le code, et non vers Live/ : le registre et
# expectations.json seraient introuvables. On se cale donc sur l'executable,
# comme le fait la fenetre.
if getattr(sys, "frozen", False):
    HERE = Path(sys.executable).resolve().parent
else:
    HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EXPECT = REPO / "docs" / "data" / "expectations.json"
REGISTER = HERE / "Benchmarks"

# Le site normalise a 10 % de volatilite ex-post pour comparer des series de
# risques differents ; on garde la meme cible pour rester lisible d'un ecran
# a l'autre.
VN_TARGET = 0.10
YEAR = 256


# --------------------------------------------------------------- la mesure
def stat(excess, rf, k: float = 1.0) -> dict:
    """Les statistiques du site, sur des rendements quotidiens excedentaires.

    `excess` est net du taux sans risque, `rf` est ce taux : la volatilite et
    le Sharpe se lisent sur l'excedent seul, mais la valeur liquidative -- et
    donc la perte maximale -- porte les interets, parce que c'est bien ce que
    le compte aurait fait. `k` sert a la version normalisee en volatilite :
    on ne met a l'echelle que la partie risquee, jamais les interets.

    La premiere seance est une amorce : elle pose la valeur de depart et ne
    porte pas de rendement. On l'ecarte, comme le fait la publication. La
    fonction `stats` du portefeuille, elle, la garde ; sur un releve qui
    commence a l'inception les deux ne tombent donc pas exactement au meme
    Sharpe. C'est la convention des quatre series du site qu'on retient ici,
    puisque c'est a elles que le classement compare.
    """
    import numpy as np

    r = np.asarray(excess, dtype=float)[1:] * k
    f = np.asarray(rf, dtype=float)[1:]
    if r.size == 0:
        return {"total": 0.0, "vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    vol = float(r.std(ddof=0) * math.sqrt(YEAR))
    nav = np.cumprod(1.0 + r + f)
    return {"total": round(float(nav[-1] - 1.0), 6),
            "vol": round(vol, 6),
            "sharpe": round(float(r.mean() * YEAR / vol) if vol else 0.0, 4),
            "max_dd": round(float((nav / np.maximum.accumulate(nav)
                                   - 1.0).min()), 6)}


def vn_factor(excess) -> float:
    """Le facteur qui amene la serie a VN_TARGET de volatilite ex-post."""
    import numpy as np
    sd = float(np.asarray(excess, dtype=float).std(ddof=0) * math.sqrt(YEAR))
    return VN_TARGET / sd if sd else 1.0


def derive(st: dict, months: int) -> dict:
    """Ce qui se deduit des statistiques de base, sans nouvelle hypothese."""
    years = months / 12.0 if months else 0.0
    total = float(st.get("total", 0.0))
    cagr = ((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    dd = abs(float(st.get("max_dd", 0.0)))
    return {"cagr": cagr, "calmar": (cagr / dd) if dd else 0.0}


# ------------------------------------------------------------- le registre
def _site_entries() -> list:
    """Les quatre series du site, avec leurs statistiques telles que publiees.

    On ne les recalcule pas : le site les a deja etablies sur le quotidien,
    que le JSON ne publie pas. Les relire garantit que les deux surfaces
    disent le meme chiffre.
    """
    if not EXPECT.is_file():
        return []
    d = json.loads(EXPECT.read_text(encoding="utf-8"))
    if "hist_bench_stats" not in d:
        return []
    vn = {r["key"]: r for r in d.get("hist_bench_vn_stats", [])}
    curve = {}
    for row in d.get("hist_bench", []):
        for k, v in row.items():
            if k != "month":
                curve.setdefault(k, []).append((row["month"], v))
    months = len(d.get("hist_bench", []))
    out = []
    for r in d["hist_bench_stats"]:
        k = r["key"]
        out.append({
            "key": k, "name": r["name"], "source": "site",
            "from": d.get("hist_bench_from", ""), "to": d.get("hist_bench_to", ""),
            "months": months, "sessions": d.get("hist_bench_n", 0),
            "stats": {q: r[q] for q in ("total", "vol", "sharpe", "max_dd")},
            "vn": {q: vn.get(k, {}).get(q) for q in
                   ("total", "vol", "sharpe", "max_dd", "k")},
            "monthly": curve.get(k, []),
            "note": "publie sur le site",
        })
    return out


def _custom_entries() -> list:
    """Les strategies ajoutees a la main, une par fichier."""
    out = []
    if not REGISTER.is_dir():
        return out
    for p in sorted(REGISTER.glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [!] {p.name} illisible : {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        missing = [q for q in ("key", "name", "stats") if q not in d]
        if missing:
            print(f"  [!] {p.name} incomplet : il manque {', '.join(missing)}",
                  file=sys.stderr)
            continue
        d.setdefault("source", "ajout")
        d.setdefault("monthly", [])
        d.setdefault("months", len(d["monthly"]))
        d.setdefault("vn", {})
        d.setdefault("note", "")
        d.setdefault("from", "")
        d.setdefault("to", "")
        d["file"] = p.name
        out.append(d)
    return out


def entries() -> list:
    """Tout le registre, chaque entree portant deja ses metriques derivees."""
    rows = _site_entries() + _custom_entries()
    seen = {}
    for e in rows:
        e.update(derive(e["stats"], e.get("months", 0)))
        if e["key"] in seen:
            print(f"  [!] cle en double : {e['key']} ({e.get('file', 'site')})",
                  file=sys.stderr)
        seen[e["key"]] = True
    return rows


SORTS = {
    "sharpe":  ("Sharpe",       lambda e: e["stats"]["sharpe"],  True),
    "total":   ("Total",        lambda e: e["stats"]["total"],   True),
    "cagr":    ("TCAC",         lambda e: e["cagr"],             True),
    "vol":     ("Volatilite",   lambda e: e["stats"]["vol"],     False),
    "max_dd":  ("Perte max",    lambda e: e["stats"]["max_dd"],  True),
    "calmar":  ("Calmar",       lambda e: e["calmar"],           True),
    "name":    ("Nom",          lambda e: e["name"].lower(),     False),
}


def rank(rows: list, by: str = "sharpe") -> list:
    """Classe, et pose le rang dans chaque ligne."""
    if by not in SORTS:
        by = "sharpe"
    _, key, desc = SORTS[by]
    out = sorted(rows, key=key, reverse=desc)
    for i, e in enumerate(out, 1):
        e["rank"] = i
    return out


# --------------------------------------------- fabriquer une entree nouvelle
def from_portfolio(parquet: Path, key: str, name: str, note: str = "",
                   start: str = "", end: str = "") -> dict:
    """Mesure une simulation de portefeuille au metre du site.

    On prend `net_ret` comme rendement excedentaire et `rf_accrual_next`
    comme taux, exactement comme le fait la publication : c'est ce qui rend
    la nouvelle ligne comparable aux quatre autres plutot que simplement
    voisine.

    Attention au fichier qu'on lui donne : Portfolio.parquet tel que le
    pipeline le laisse ne contient que la fenetre vivante, la publication
    resimulant l'historique en memoire sans jamais l'ecrire. Mesurer 1998
    dessus ne renvoie pas une erreur mais des zeros, ce qui serait pire ;
    d'ou le controle plus bas.
    """
    import numpy as np
    import polars as pl

    d = pl.read_parquet(parquet).sort("date")
    need = {"date", "net_ret"}
    if not need <= set(d.columns):
        raise SystemExit(f"[ABORT] {parquet.name} : il manque "
                         f"{', '.join(sorted(need - set(d.columns)))}")
    dates = [str(x) for x in d.get_column("date").to_list()]
    if start:
        keep = [i for i, x in enumerate(dates) if x >= start]
    else:
        keep = list(range(len(dates)))
    if end:
        keep = [i for i in keep if dates[i] <= end]
    if len(keep) < 2:
        raise SystemExit(f"[ABORT] {parquet.name} : {len(keep)} seance(s) "
                         f"dans la fenetre demandee")

    ret = d.get_column("net_ret").to_numpy()
    # Sans fenetre explicite, on se cale sur ce que le fichier porte vraiment.
    # Une simulation ecrit une ligne par seance depuis 1978 mais ne remplit
    # les rendements qu'a partir de son propre depart : prendre le fichier
    # entier noierait la volatilite sous des milliers de zeros.
    if not (start or end):
        nz = [i for i in keep if float(ret[i] or 0.0) != 0.0]
        if nz:
            # On remonte d'une seance : c'est l'amorce, celle que `stat`
            # ecarte. Partir directement du premier rendement non nul le
            # ferait passer pour l'amorce et le perdrait.
            first = max(nz[0] - 1, keep[0])
            keep = [i for i in keep if first <= i <= nz[-1]]
    live = sum(1 for i in keep if float(ret[i] or 0.0) != 0.0)
    if live < 20:
        span = f"{dates[keep[0]]} a {dates[keep[-1]]}"
        nz = [dates[i] for i in range(len(dates))
              if float(ret[i] or 0.0) != 0.0]
        has = f"{nz[0]} a {nz[-1]}" if nz else "aucune"
        raise SystemExit(
            f"[ABORT] {parquet.name} : {live} rendement(s) non nul(s) sur "
            f"{len(keep)} seances demandees ({span}).\n"
            f"          Le fichier ne porte des rendements que sur {has}.\n"
            f"          Une simulation ne garde l'historique que si elle est "
            f"lancee dessus :\n"
            f"          python 3_Portfolio/portfolio.py --start-date 1990-01-02")
    rfa = (d.get_column("rf_accrual_next").to_numpy()
           if "rf_accrual_next" in d.columns else np.zeros(len(dates)))
    # Le taux acquis A l'entree du jour est celui de la veille (cf. publish).
    ex = np.array([0.0] + [float(ret[i] or 0.0) for i in keep[1:]])
    rf = np.array([0.0] + [float(rfa[i - 1] or 0.0) for i in keep[1:]])
    dts = [dates[i] for i in keep]

    st = stat(ex, rf)
    k = vn_factor(ex)
    vst = stat(ex, rf, k)
    vst["k"] = round(k, 4)

    nav = np.cumprod(1.0 + ex + rf)
    last = {}
    for i, x in enumerate(dts):
        last[x[:7]] = i
    monthly = [[m, round(float(nav[last[m]]), 6)] for m in sorted(last)]

    return {"key": key, "name": name, "note": note, "source": "ajout",
            "from": dts[0][:7], "to": dts[-1][:7], "months": len(monthly),
            "sessions": len(dts), "stats": st, "vn": vst, "monthly": monthly}


# -------------------------------------------------------------------- la CLI
def flag_windows(rows: list) -> tuple:
    """Marque les entrees qui ne couvrent pas la meme periode que les autres.

    Classer un releve vivant de 173 seances a cote d'un historique de 28 ans
    revient a comparer deux choses differentes ; on ne l'interdit pas -- c'est
    parfois exactement ce qu'on veut voir -- mais on refuse de le cacher.

    Renvoie les deux bornes de la periode majoritaire sans les mettre en
    phrase : la ligne de commande parle francais, la fenetre anglais, et
    fabriquer le libelle ici melangerait les deux.
    """
    spans = {}
    for e in rows:
        spans.setdefault((e.get("from", ""), e.get("to", "")), []).append(e)
    if not spans:
        return ("", "")
    main = max(spans, key=lambda k: len(spans[k]))
    for k, group in spans.items():
        for e in group:
            e["off_window"] = (k != main)
    return main


def _fmt(e: dict) -> str:
    s, v = e["stats"], e.get("vn") or {}
    vt = v.get("total")
    return ("  %2d %s %-32s %9.2fx %7.2f%% %7.2f%% %7.2f %8.1f%% %7.2f %9s"
            % (e["rank"], "*" if e.get("off_window") else " ", e["name"][:32],
               1.0 + s["total"], 100 * e["cagr"], 100 * s["vol"],
               s["sharpe"], 100 * s["max_dd"], e["calmar"],
               ("%.2fx" % (1.0 + vt)) if vt is not None else "-"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--by", default="sharpe", choices=sorted(SORTS),
                    help="colonne de classement")
    ap.add_argument("--json", action="store_true",
                    help="sortie machine plutot que tableau")
    ap.add_argument("--add", default=None, metavar="PARQUET",
                    help="mesure une simulation et l'inscrit au registre")
    ap.add_argument("--key", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--start", default="", help="par defaut : tout le fichier")
    ap.add_argument("--end", default="")
    ap.add_argument("--like-site", action="store_true",
                    help="cadre sur 1998-2025, la fenetre des quatre series "
                         "du site")
    a = ap.parse_args()
    if a.like_site:
        a.start, a.end = "1998-01-01", "2025-12-31"

    if a.add:
        if not (a.key and a.name):
            print("[ABORT] --add demande aussi --key et --name")
            return 2
        e = from_portfolio(Path(a.add), a.key, a.name, a.note, a.start, a.end)
        REGISTER.mkdir(parents=True, exist_ok=True)
        out = REGISTER / f"{a.key}.json"
        out.write_text(json.dumps(e, indent=2), encoding="utf-8")
        print(f"  inscrit : {out}")
        print(f"  {e['sessions']:,} seances, {e['from']} a {e['to']}")
        print(f"  Sharpe {e['stats']['sharpe']}, "
              f"vol {100 * e['stats']['vol']:.2f}%, "
              f"perte max {100 * e['stats']['max_dd']:.1f}%")
        return 0

    rows = rank(entries(), a.by)
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("  registre vide : le site n'a pas encore publie "
              "expectations.json")
        return 1
    label = SORTS[a.by][0]
    w0, w1 = flag_windows(rows)
    print(f"\n  CLASSEMENT  ({len(rows)} strategies, par {label})")
    print(f"  fenetre {w0} a {w1}, rendements excedentaires quotidiens")
    print("  %2s %s %-32s %10s %8s %8s %7s %9s %7s %9s"
          % ("#", " ", "strategie", "total", "TCAC", "vol", "Sharpe",
             "perte max", "Calmar", "vn 10%"))
    print("  " + "-" * 106)
    for e in rows:
        print(_fmt(e))
    off = [e for e in rows if e.get("off_window")]
    if off:
        print(f"\n  * mesure sur une autre periode, donc pas directement "
              f"comparable :")
        for e in off:
            print(f"      {e['name']} : {e.get('from', '?')} a "
                  f"{e.get('to', '?')}, {e.get('sessions', 0):,} seances")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

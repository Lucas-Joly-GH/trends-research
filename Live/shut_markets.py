"""Quels instruments n'ont pas avance, et pourquoi.

LE PROBLEME QUE CE MODULE RESOUT. Le panel avance des qu'un quorum
d'instruments a cote une nouvelle seance. Les marches qui n'ont PAS cote ce
jour-la restent en arriere -- legitimement, leur bourse etait fermee. Six
verifications supposaient pourtant que chaque marche cote chaque seance, et
tombaient donc toutes ensemble quatre fois par an : lundi de Paques, premier
et dernier lundi de mai, lundi d'aout, lendemain de Noel. Onze fois depuis
2024 pour la seule place de Londres.

DEUX CAUSES, PAS UNE. Un instrument peut manquer la seance parce que
  * sa bourse etait FERMEE ce jour-la (jour ferie local), ou
  * sa seance se termine plus tard que l'heure ou le pipeline a tourne, ou
    le fournisseur n'a pas encore publie (decalage horaire).

On les distingue sans calendrier externe, par l'histoire de l'instrument
lui-meme : un marche qui a deja rate des seances que le panel avait est un
marche qui suit son propre calendrier ; un marche qui n'en a jamais rate une
seule et qui manque soudain la derniere est un marche dont la donnee n'est
pas encore arrivee.

Le seuil est volontairement bas -- deux absences anterieures suffisent. Une
absence isolee reste traitee comme un retard de donnee, donc comme une
anomalie a signaler, ce qui est le comportement prudent.
"""
from __future__ import annotations

import csv
import pathlib

HOLIDAY = "fermé"
LATE = "retard"
AHEAD = "avance"

# Deux absences anterieures suffisent a etablir qu'un marche suit son propre
# calendrier. En dessous, on prefere crier au retard de donnee.
_GAPS_FOR_CALENDAR = 2


def _sessions(path: pathlib.Path, col: str = "Continuous_C") -> list[str]:
    """Les seances ou l'instrument a reellement cote."""
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get(col) not in ("", "None", None):
                out.append(r["date"])
    return out


def survey(book_dir: str | pathlib.Path, lookback: int = 500) -> dict:
    """Etat de chaque instrument par rapport a la seance la plus recente.

    Renvoie ``{"as_of": str, "shut": {inst: (derniere, motif)}}`` ou le motif
    vaut HOLIDAY, LATE ou AHEAD. Un instrument AHEAD a cote une seance PLUS
    RECENTE que le panel : sa place ferme plus tot dans la journee, et le
    quorum le retient volontairement pour ne pas publier une coupe ou deux
    instruments sur soixante-trois ont un jour d'avance.
    """
    book_dir = pathlib.Path(book_dir)
    per = {}
    for f in sorted(book_dir.glob("*.csv")):
        s = _sessions(f)
        if s:
            per[f.stem] = s
    if not per:
        return {"as_of": "", "shut": {}}

    as_of = max(s[-1] for s in per.values())
    # Le calendrier de reference : les seances vues par le panel entier.
    panel = sorted({d for s in per.values() for d in s})[-lookback:]

    shut = {}
    for inst, s in per.items():
        if s[-1] == as_of:
            continue
        if s[-1] > as_of:                       # impossible ici, garde-fou
            shut[inst] = (s[-1], AHEAD)
            continue
        own = set(s)
        # Combien de seances du panel cet instrument a-t-il deja ratees,
        # avant celle-ci ? C'est la signature d'un calendrier propre.
        gaps = sum(1 for d in panel if d not in own and d < as_of)
        shut[inst] = (s[-1], HOLIDAY if gaps >= _GAPS_FOR_CALENDAR else LATE)
    return {"as_of": as_of, "shut": shut}


def pending_from_shut(pending_csv: str | pathlib.Path,
                      shut: dict, as_of: str) -> float:
    """Commission decidee mais pas encore prelevee faute de seance.

    Un ordre decide avant ``as_of`` sur un marche ferme n'a pas pu s'executer :
    il reste en attente et son cout n'apparait ni au releve ni dans l'equity.
    Les rapprochements doivent en tenir compte plutot que de crier a l'ecart.
    """
    p = pathlib.Path(pending_csv)
    if not p.is_file():
        return 0.0
    tot = 0.0
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("instrument") in shut and r.get("decision_date", "") < as_of:
                try:
                    tot += float(r.get("commission_USD") or 0.0)
                except ValueError:
                    pass
    return tot


def describe(state: dict) -> str:
    """Une ligne pour le journal et pour la fenetre du .exe."""
    shut = state.get("shut") or {}
    if not shut:
        return ""
    order = ((HOLIDAY, "marché fermé"), (LATE, "données en retard"),
             (AHEAD, "décalage horaire"))
    bits = []
    for tag, label in order:
        who = sorted(k for k, (_d, m) in shut.items() if m == tag)
        if who:
            bits.append(f"{label}: {', '.join(who)}")
    return "  |  ".join(bits)

"""L'ecusson a colonnes, reconstruit en vectoriel puis inverse.

PAS UN CALQUE : la reference est un bitmap, et un bitmap agrandi bave,
reduit se bouche, et ne se recolore pas proprement. On redessine donc la
geometrie, ce qui donne un trace net a n'importe quelle taille.

CE QUI A RATE DEUX FOIS. J'ai d'abord traite les fentes comme des rayons
partant d'un foyer unique. Un rayon est une droite : la fente exterieure
part donc en biais des le sommet et traverse tout le blason, d'ou le
coquillage. Or dans la reference les colonnes sont VERTICALES en haut et
ne se rabattent vers le centre que dans le dernier tiers. Ce n'est pas un
eventail, c'est un faisceau qui se resserre en bas -- deux regimes, pas
un. On modelise donc le coude explicitement.
"""
import pathlib
import sys

# DEUX DECLINAISONS DU MEME BLASON, et ce n'est pas un caprice : sept
# colonnes ne peuvent pas se resoudre dans seize pixels. Les fentes s'y
# bouchent et la marque devient une pastille grise. La version a cinq
# colonnes garde la silhouette, l'ecartement et l'arche -- elle se lit
# comme la meme chose -- mais ses fentes sont assez larges pour survivre a
# la reduction. Sept partout ou il y a la place, cinq dans l'onglet.
#
# L'empan reste identique dans les deux cas : c'est ce qui fait qu'on les
# reconnait comme une seule marque et non comme deux.
SPAN = 83.5


def geometry(n, gap_ratio):
    """Entraxe et largeur de colonne pour n colonnes a empan constant."""
    barw = SPAN / (n + (n - 1) * gap_ratio)
    return barw * (1 + gap_ratio), barw


N = 7
PITCH, BARW = geometry(N, 0.47)
TOP_C = 9.0                # sommet de la colonne centrale
STEP = 5.6                 # descente du sommet par colonne ecartee
BEND = 57.0                # ou les colonnes commencent a se rabattre
# LE DESSIN DOIT TENIR DANS SES 100 UNITES : en faisant plonger le flanc
# trop bas, le blason se faisait couper par le bord de la vignette et non
# par l'arc.
PULL = 0.72                # ce que les FENTES rattrapent au bas du coude
BOT = 99.0                 # ou les fentes s'arretent, sous l'arc
# Le flanc se rabat MOINS vite que les fentes : c'est ce qui laisse a la
# colonne exterieure une largeur au lieu de la reduire a un fil.
OUT_PULL = 0.66
# L'ARC EST LA BASE, PAS UNE ENCOCHE.  Premiere lecture, fausse : j'en
# avais fait un petit trou perce dans un fond plat, si bien que les
# colonnes centrales s'y terminaient et les exterieures sur le bord droit
# -- des pages posees sur deux appuis differents. Ce qu'il faut, c'est un
# seul demi-cercle sur lequel TOUTES viennent se relier, comme un cahier
# sur sa reliure.
#
# Son rayon n'est donc pas un reglage : il vaut exactement la demi-largeur
# du faisceau une fois resserre, pour que l'arc rejoigne les deux branches
# exterieures a leur extremite. Le calculer plutot que le choisir garantit
# que la reliure reste juste si l'ecartement ou le coude changent.
BOT_LINE = 94.0            # la ligne du bas EST le centre de l'arc


def cx(i):
    return 50.0 + (i - (N - 1) / 2) * PITCH


def bar_top(i):
    return TOP_C + abs(i - (N - 1) / 2) * STEP


def converge(x, pull=None):
    """Ou se trouve l'abscisse x une fois le faisceau resserre."""
    return 50.0 + (x - 50.0) * (1.0 - (PULL if pull is None else pull))


def arc_r():
    """Le rayon de la reliure : la demi-largeur du faisceau resserre."""
    return 50.0 - converge(cx(0) - BARW / 2, OUT_PULL)


def outline():
    """Fronton en gradins, flancs verticaux, puis la pointe de l'ecu."""
    pts = []
    for i in range(N):
        t = bar_top(i)
        pts.append((cx(i) - BARW / 2, t))
        pts.append((cx(i) + BARW / 2, t))
    xr, xl = cx(N - 1) + BARW / 2, cx(0) - BARW / 2
    pts += [(xr, BEND), (converge(xr, OUT_PULL), BOT_LINE),
            (converge(xl, OUT_PULL), BOT_LINE), (xl, BEND)]
    return pts


def slot(j):
    """Une fente : verticale jusqu'au coude, rabattue vers le centre
    ensuite. Elle demarre au-dessus du fronton et finit sous la pointe --
    ce qui deborde est coupe par le contour, et evite d'avoir a calculer
    ou exactement chaque fente rencontre le bord."""
    l = cx(j) + BARW / 2
    r = cx(j + 1) - BARW / 2
    return [(l, -6.0), (r, -6.0),
            (r, BEND), (converge(r), BOT),
            (converge(l), BOT), (l, BEND)]


def configure(n, gap_ratio, step):
    """Basculer les constantes du module sur une declinaison."""
    global N, PITCH, BARW, STEP
    N, STEP = n, step
    PITCH, BARW = geometry(n, gap_ratio)


# UNE SEULE DECLINAISON. Cinq colonnes tiennent a 16 px et restent
# franches en grand : garder une seconde geometrie pour les grandes tailles
# n'achetait qu'une difference que personne ne verrait, au prix de deux
# dessins a maintenir accordes.

# La declinaison retenue : cinq colonnes, une fente valant 0,62 colonne,
# un gradin de 7,6. Les autres essais ont ete ecartes -- cinq tiennent a
# 16 px et restent franches en grand.
configure(5, 0.62, 7.6)

"""La carte de partage : 1200x630, dessinee a partir des memes chiffres.

POURQUOI LA REGENERER A CHAQUE PUBLICATION plutot qu'une image fixe. Une
carte figee montrant une courbe et un Sharpe devient fausse des le
lendemain, et elle est vue par des gens qui ne cliqueront peut-etre
jamais -- c'est le seul endroit du site ou une valeur perimee ne peut
etre ni datee ni corrigee par le lecteur. Soit elle suit les donnees,
soit elle ne porte aucun chiffre. Elle les suit.

POURQUOI RIEN ICI NE PEUT FAIRE ECHOUER LA PUBLICATION. L'appelant
enveloppe cet appel : une vignette est un ornement, et un pipeline qui
refuse de publier une seance parce qu'une police manque aurait ses
priorites a l'envers. En cas d'echec la carte precedente reste en place,
ce qui est exactement le bon repli -- elle est deja proche.

La palette est celle du theme clair du site, recopiee et non deduite :
le CSS n'est pas lisible d'ici, et une carte qui derive lentement de
l'identite du site est pire qu'une carte qui casse franchement.
"""
from __future__ import annotations

PAGE = "#fdfdfc"
INK = "#14213d"
MUTED = "#5a6472"
RULE = "#c9ccd4"
POS = "#176b45"
NEG = "#9b2226"

W, H, DPI = 1200, 630, 100


def draw_crest(fig, plt):
    """Le blason du site, en haut a droite de la carte.

    DESSINE, PAS IMPORTE. Le meme blason existe en SVG dans docs/, mais le
    lire ici supposerait un moteur de rendu vectoriel dans le pipeline --
    une dependance de plus pour un ornement, et la carte a justement pour
    regle de ne jamais pouvoir faire echouer une publication. La geometrie
    est partagee avec les icones via crest.py ; seul le trace differe.

    En haut a DROITE : le titre et les chiffres occupent la gauche, et une
    marque posee la n'entrerait en concurrence avec rien.
    """
    from matplotlib.patches import Circle, Polygon
    import crest

    # De la grille 0..100 du blason vers les coordonnees de figure.
    size, x0, y0 = 0.132, 0.845, 0.655
    m = lambda p: (x0 + p[0] / 100.0 * size,
                   y0 + (100 - p[1]) / 100.0 * size * W / H)

    fig.add_artist(Polygon([m(p) for p in crest.outline()], closed=True,
                           facecolor=INK, edgecolor="none", zorder=3))
    # Les fentes et l'arc sont repeints dans la couleur du fond : sur une
    # carte au fond plein c'est exact, et cela evite un masque que
    # matplotlib rendrait mal a cette taille.
    for j in range(crest.N - 1):
        fig.add_artist(Polygon([m(q) for q in crest.slot(j)], closed=True,
                               facecolor=PAGE, edgecolor="none", zorder=4))
    c = m((50.0, crest.BOT_LINE))
    fig.add_artist(Circle(c, crest.arc_r() / 100.0 * size,
                          facecolor=PAGE, edgecolor="none", zorder=4))


def render(out_path, dates, equity, stats: dict, url: str) -> None:
    """Ecrit la carte. Lever une exception ici est sans consequence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    eq = [float(v) for v in equity]
    up = eq[-1] >= eq[0]

    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=PAGE)

    # La courbe occupe le bas de la carte : sur une vignette, le texte doit
    # etre lisible en premier et la forme ne sert qu'a dire « c'est une
    # serie temporelle ». L'inverse -- un grand graphique et un titre en
    # coin -- ne se lit pas a 300 px de large dans un fil.
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.42])
    ax.set_facecolor(PAGE)
    # L'aire descend jusqu'au BAS DU CADRE, pas jusqu'au minimum de la serie :
    # remplir jusqu'au minimum laisse une bande blanche sous la courbe, et sur
    # une vignette sans axes ni cadre cette bande ne se lit pas comme une
    # marge mais comme une image mal decoupee.
    lo, hi = min(eq), max(eq)
    pad = (hi - lo) * 0.12 or 1.0
    floor = lo - pad
    ax.plot(range(len(eq)), eq, color=INK, linewidth=2.0, solid_capstyle="round")
    ax.fill_between(range(len(eq)), eq, floor, color=POS if up else NEG,
                    alpha=0.07, linewidth=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.set_xlim(0, len(eq) - 1)
    ax.set_ylim(floor, hi + pad)

    def txt(x, y, s, size, color=INK, weight="normal", family="serif"):
        fig.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                 fontfamily=family, va="baseline")

    draw_crest(fig, plt)
    txt(0.055, 0.815, "Systematic futures", 44, INK, "bold")
    txt(0.055, 0.735, "2026 paper-trading run, published every session", 21, MUTED)

    # Le filet s'arrete AVANT le blason. A 0,945 il lui passait dessous :
    # invisible parce que le blason est au-dessus, mais il ressortait de
    # part et d'autre comme un trait qui traverse la marque.
    fig.add_artist(plt.Line2D([0.055, 0.805], [0.700, 0.700],
                              color=RULE, linewidth=1.2))

    # Quatre chiffres, pas huit : ce qui doit survivre a une vignette lue en
    # une seconde. Le libelle sous la valeur, parce que l'oeil descend.
    cells = [(stats["ret"], "Return, annualised"),
             (stats["sharpe"], "Sharpe"),
             (stats["dd"], "Deepest drawdown"),
             (stats["sessions"], "Sessions published")]
    for i, (v, lab) in enumerate(cells):
        x = 0.055 + i * 0.2266
        col = NEG if v.startswith("-") else INK
        txt(x, 0.560, v, 33, col, "bold")
        txt(x, 0.498, lab, 15, MUTED)

    txt(0.055, 0.436, url, 15, MUTED, family="monospace")
    fig.savefig(out_path, facecolor=PAGE, dpi=DPI)
    plt.close(fig)

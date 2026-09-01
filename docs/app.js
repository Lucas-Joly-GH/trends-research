// Shared helpers for every page.  No dependencies, no build step.

// CACHE BUSTING, FROM THIS SCRIPT'S OWN TAG.  GitHub Pages serves static assets
// with a ten-minute cache and neither `app.js` nor the JSON ever changes name,
// so a reader who visited earlier can get new HTML against a stale script --
// exactly the mix that renders a page which looks updated and behaves like the
// old one.  `publish.py` stamps `<script src="app.js?v=HASH">` with a hash of
// the script AND the data, so the value moves whenever either does; every data
// fetch then inherits it through `bust()`.  No stamp (opened from disk, or a
// page publish.py has not touched) degrades to the plain URL.
const BUILD = (() => {
  const src = document.currentScript && document.currentScript.src || "";
  const m = src.match(/[?&]v=([A-Za-z0-9]+)/);
  return m ? m[1] : "";
})();
const bust = u => BUILD ? u + (u.includes("?") ? "&" : "?") + "v=" + BUILD : u;

// LA REQUETE VERS latest.json EST PARTAGEE, PAS REPETEE.  Le bandeau de nav en
// a besoin sur toutes les pages, et l'accueil comme la page Q&A en ont besoin
// pour leur propre contenu : les deux partaient en meme temps, donc aucune ne
// pouvait servir de cache a l'autre et ces pages telechargeaient le fichier
// deux fois.  Une seule promesse, consommee autant de fois qu'on veut. Chaque
// consommateur attache son propre `.catch` : le bandeau tolere l'echec, une
// page privee de ses chiffres non.
const LATEST = fetch(bust("data/latest.json")).then(r => r.json());

// LOCALE IS PINNED, not inherited.  A French browser renders 1.35300 as
// "1,35300" and $108,996,447 as "108 996 447" -- correct locally, ambiguous to
// everyone else, and a comma decimal separator on a price table turns a
// checkable number into an argument.
const L = "en-US";
const el = id => document.getElementById(id);
const fmtMoney = v => v == null ? "—" :
  v.toLocaleString(L, {minimumFractionDigits: 0, maximumFractionDigits: 0});
const fmtPx = v => {
  if (v == null) return "—";
  const a = Math.abs(v);
  const d = a === 0 ? 2 : a < 10 ? 5 : a < 1000 ? 3 : 2;
  return v.toLocaleString(L, {minimumFractionDigits: d, maximumFractionDigits: d});
};
const pct = v => v == null ? "—" : (v * 100).toFixed(2) + "%";

// A PERFORMANCE FIGURE IS NEVER ROUNDED IN ITS OWN FAVOUR.  1.1651 rounds to
// 1.17 under the usual rule, which is half a hundredth of free credit at the
// boundary -- and the boundary is exactly where a reader checking the number
// against their own arithmetic will land on the other side of it. `Math.floor`
// is the right primitive rather than a special case for the sign: it rounds
// toward -infinity, which is the LESS flattering direction for a Sharpe whether
// it is positive or negative.
//
// The `toFixed(6)` is not decoration. `1.16 * 100` is 115.99999999999999 in
// binary floating point, and flooring that gives 1.15 -- a figure understated
// by a whole hundredth by the very code meant to shave half of one.
const floor2 = v => v == null ? "—"
  : (Math.floor(Number((v * 100).toFixed(6))) / 100).toFixed(2);

// ALIGNMENT BELONGS TO THE COLUMN, NOT TO ITS POSITION.  It used to be set by
// :first-child / :nth-child(2), which suited Instrument and Contract and left
// Side and Why right-aligned like numbers -- a four-letter word pushed to the
// right of a wide column detaches from its neighbour and reads as floating.
// Positional rules also cannot be right for both tables at once: Pending has
// six columns and Executed nine, in different orders.  `l:true` marks a text
// column; everything else is numeric and stays right.
// NEGATIVES ARE DETECTED FROM THE RENDERED TEXT, not from the underlying
// number.  A column added later gets the colour without anyone remembering to
// wire it, and a formatter that decides to print "(1,234)" or round -0.4 to
// "0.00" stays honest -- the cell is coloured if and only if a minus is on
// screen.  U+2014 (the em dash used for a shut market) deliberately does not
// match; U+2212 and the plain hyphen do.
const isNeg = v => /^[-−]/.test(String(v).trim());
// GREEN IS OPT-IN, RED IS NOT.  Any negative is worth catching wherever it
// appears, but a column that can only be positive -- Qty, Fill open, Commission
// -- learns nothing from being green, and colouring all of them is how a report
// turns into a trading terminal.  So `pos` only lands on a column whose
// definition says `signed: true`, meaning both signs genuinely occur in it.
const isPos = v => {
  const n = parseFloat(String(v).replace(/[^0-9.\-−]/g, "")
                                .replace("−", "-"));
  return isFinite(n) && n > 0;
};

// UN GRAPHIQUE SANS NOM N'EXISTE PAS POUR UN LECTEUR D'ECRAN.  Un <svg> nu est
// annonce comme « graphique », sans plus, et les huit de ce site le sont tous
// de la meme facon -- autant de trous identiques dans la lecture de la page.
//
// `aria-labelledby` PLUTOT QU'`aria-label` : la legende est ecrite APRES le
// trace (elle cite des chiffres que le trace vient de calculer), donc copier
// son texte au moment du dessin capturerait une chaine vide. Pointer dessus
// resout la lecture au moment ou l'utilisateur y arrive, pas avant. La
// legende recoit un identifiant si elle n'en a pas.
let _figN = 0;
function nameChart(svg) {
  const fig = svg.closest("figure");
  const cap = fig && fig.querySelector("figcaption");
  if (!cap) return;
  if (!cap.id) cap.id = "figcap" + (++_figN);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-labelledby", cap.id);
}

function table(node, rows, cols) {
  if (!rows.length) { node.innerHTML = "<tr><td>Nothing.</td></tr>"; return; }
  const klass = (c, v) => {
    const sign = v === undefined ? ""
               : isNeg(v) ? "neg"
               : (c.signed && isPos(v)) ? "pos" : "";
    const k = ((c.l ? "l " : "") + sign).trim();
    return k ? ` class="${k}"` : "";
  };
  node.innerHTML =
    "<thead><tr>" + cols.map(c => `<th scope="col"${klass(c)}>${c.label}</th>`)
                        .join("") + "</tr></thead>"
    + "<tbody>" + rows.map(r =>
        "<tr>" + cols.map(c => {
          const v = c.get(r);
          return `<td${klass(c, v)}>${v}</td>`;
        }).join("") + "</tr>"
      ).join("") + "</tbody>";
}

const C = {
  inst:   {label: "Instrument",     get: r => r.instrument, l: true},
  con:    {label: "Contract",       get: r => r.contract, l: true},
  side:   {label: "Side",           get: r => r.action, l: true},
  qty:    {label: "Qty",            get: r => fmtMoney(r.quantity)},
  kind:   {label: "Why",            get: r => r.kind, l: true},
  open:   {label: "Fill open",      get: r => fmtPx(r.fill_open)},
  close:  {label: "Decision close", get: r => fmtPx(r.decision_close)},
  // « Commission » sous-vendait ce qui est modelise : le cout porte un
  // tick plein de spread a l'aller comme au retour, en plus des frais.
  // L'appeler commission invitait le reproche d'ignorer le slippage,
  // alors qu'il est dedans.
  comm:   {label: "Trading cost $", get: r => fmtMoney(r.trading_cost_USD)},
  // "PnL $" read as the position's move for the session, which it is not: this
  // is proportional crystallisation.  A trade that adds to a position banks
  // nothing, one that cuts it by half banks half of the P&L accumulated so far,
  // a close banks the rest -- so an untouched winner shows 0 here all year.
  // "Realised" is the word that carries that, and the footers spell it out.
  real:   {label: "Realised P&L $",  get: r => fmtMoney(r.realised_pnl_USD),
           signed: true},
  carry:  {label: "Carried",        get: r => fmtMoney(r.carried_sessions)},
  reason: {label: "Reason",         get: r => r.reason, l: true},
};

// THE THEME TOGGLE.  The stored choice is applied by an inline script in the
// <head> so the page never paints in the wrong scheme; this only wires the
// button and keeps its label truthful.
//
// NOTHING IS REDRAWN WHEN THE THEME CHANGES, including the charts. That falls
// out of having moved SVG colour onto classes: the drawing code emits
// `class="series"` and the stylesheet decides what that means, so a theme
// switch is a repaint the browser does by itself. Had the colours stayed in
// presentation attributes, every chart would need rebuilding here.
const THEME_KEY = "theme";
const stored = () => { try { return localStorage.getItem(THEME_KEY); }
                       catch (e) { return null; } };

// The ACTIVE theme: the stored choice if there is one, light otherwise -- not
// simply `dataset.theme`, which is empty until someone chooses and would make
// the button offer to switch to the theme already on screen. The fallback is a
// constant now rather than a media query, because the stylesheet no longer
// consults the system either; the two have to agree or the button lies.
function activeTheme() {
  return document.documentElement.dataset.theme || "light";
}

function initTheme() {
  const btn = el("theme");
  if (!btn) return;
  const paint = () => {
    // The label names the DESTINATION, and `title` says so in words, because
    // a bare "Light" is equally readable as the current state or the target.
    const to = activeTheme() === "dark" ? "light" : "dark";
    btn.textContent = to === "light" ? "Light" : "Dark";
    btn.title = `Switch to the ${to} theme`;
    btn.setAttribute("aria-label", btn.title);
  };
  paint();
  btn.addEventListener("click", () => {
    const to = activeTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = to;
    try { localStorage.setItem(THEME_KEY, to); } catch (e) {}
    paint();
  });
}
addEventListener("DOMContentLoaded", initTheme);
if (document.readyState !== "loading") initTheme();

// STALENESS IS THE READER'S QUESTION, NOT THE PUBLISHER'S.  `Updated` tells
// you when the pipeline last ran, but only if you know what it SHOULD say --
// a date that stopped moving looks exactly like a date. This says it in words.
//
// Threshold is 3 days, which is what a weekend costs: on a Monday the newest
// session is legitimately Friday's, and on the Tuesday after a missed Monday
// run it is four days old and something is wrong. Judged against the READER'S
// clock, so a badly set clock can produce a false warning -- which is why the
// wording states the fact ("the newest session shown is X, N days ago") rather
// than making a claim about the pipeline.
function staleNote(lastDate) {
  if (!lastDate) return "";
  const then = Date.parse(lastDate + "T00:00:00Z");
  if (!isFinite(then)) return "";
  const days = Math.floor((Date.now() - then) / 86400000);
  if (days <= 3) return "";
  return `The newest session shown is ${lastDate}, ${days} days ago. `
       + (days > 10
          ? "This page has stopped updating."
          : "A session may be missing.");
}

// A LINE, AN AXIS AND FOUR LABELS.  Deliberately not a charting library: the
// series is 171 points, the page is meant to read as a printed report, and a
// dependency that renders one polyline is a dependency that can break the page.
// WIDTH IS MEASURED, NOT ASSUMED.  A fixed 960-unit viewBox is scaled to fit
// whatever box it lands in, and the axis text scales with it: in a 630px column
// the 10px labels rendered at 6.6px.  Taking W from the element's own pixel
// width makes one viewBox unit one CSS pixel, so 10 means 10 wherever the chart
// sits.  Hence the redraw on resize below -- W is only true for one width.
function lineChart(id, days, vals, lo, hi, hSpec, fmt, fillDown, fmtExact,
                   ref, split) {
  const svg = el(id);
  CHARTS[id] = [days, vals, lo, hi, hSpec, fmt, fillDown, fmtExact, ref, split];
  const W = Math.round((svg.parentNode || svg).getBoundingClientRect().width) || 640;
  // HEIGHT TRACKS WIDTH.  A constant height is one aspect ratio pretending to be
  // a design: the 170px drawdown panel reads fine at 630px wide and becomes a
  // 6.4:1 sliver at 1090px.  Ratio between bounds instead.
  // `extra` lets a page hand this chart leftover column height -- see
  // `balance()` on the overview.  Ratio first, then the donation.
  const H = Math.round(Math.min(hSpec.max, Math.max(hSpec.min, W * hSpec.r))
                       + (hSpec.extra || 0));
  const PL = 62, PR = 8, PT = 10, PB = 20;
  const x = i => PL + i * (W - PL - PR) / Math.max(vals.length - 1, 1);
  const pad = (hi - lo) * 0.06 || 1;
  const y0 = lo - pad, y1 = hi + pad;
  const y = v => PT + (y1 - v) * (H - PT - PB) / (y1 - y0);

  let g = "";
  // horizontal grid + value labels
  for (let k = 0; k <= 4; k++) {
    const v = y0 + (y1 - y0) * k / 4, yy = y(v);
    g += `<line class="grid" x1="${PL}" y1="${yy}" x2="${W - PR}" y2="${yy}"/>`
       + `<text class="axis" x="${PL - 6}" y="${yy + 3.5}" `
       + `text-anchor="end">${fmt(v)}</text>`;
  }
  // month ticks, from the first session of each month
  let seen = "";
  days.forEach((r, i) => {
    const mo = r.date.slice(0, 7);
    if (mo !== seen) {
      seen = mo;
      g += `<line class="tick" x1="${x(i)}" y1="${PT}" x2="${x(i)}" `
         + `y2="${H - PB}"/>`
         + `<text class="axis" x="${x(i)}" y="${H - 6}" `
         + `text-anchor="middle">${r.date.slice(5, 7)}</text>`;
    }
  });

  const pts = vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  // a baseline: the starting equity, or zero drawdown
  // `ref` OVERRIDES THE BASELINE, and the vol chart is why. Its default is the
  // FIRST VALUE, which on an equity curve is the starting capital and means
  // something; on a realised-volatility line it is just whatever the first
  // window happened to print, and a dashed rule there reads as the target it is
  // not. A caller with a real reference level passes one.
  const base = ref ? ref.v : (fillDown ? 0 : vals[0]);

  // TWO-TONE AREA, for a series that crosses its own baseline. The drawdown
  // panel never does -- it is zero or negative, so one colour says everything.
  // An equity curve does, and a single fill would colour a losing month the
  // same as a winning one. Drawn as the SAME polygon twice under opposing
  // clips, rather than as two stitched paths: stitching needs the crossing
  // points interpolated, and getting that subtly wrong leaves hairline gaps
  // exactly where the series meets the line.
  if (split && base >= y0 && base <= y1) {
    const yb = y(base).toFixed(1);
    const area = `${x(0).toFixed(1)},${yb} ${pts} `
               + `${x(vals.length - 1).toFixed(1)},${yb}`;
    g += `<clipPath id="cu-${id}"><rect x="${PL}" y="${PT}" `
       + `width="${W - PL - PR}" height="${Math.max(yb - PT, 0)}"/></clipPath>`
       + `<clipPath id="cd-${id}"><rect x="${PL}" y="${yb}" `
       + `width="${W - PL - PR}" height="${Math.max(H - PB - yb, 0)}"/></clipPath>`
       + `<polygon class="fill-up" clip-path="url(#cu-${id})" `
       + `points="${area}"/>`
       + `<polygon class="fill-dn" clip-path="url(#cd-${id})" `
       + `points="${area}"/>`;
  }
  if (fillDown) {
    g += `<polygon class="fill" points="${x(0)},${y(0)} ${pts} `
       + `${x(vals.length - 1)},${y(0)}"/>`;
  }
  g += `<polyline class="series" points="${pts}"/>`;
  if (base >= y0 && base <= y1) {
    g += `<line class="base" x1="${PL}" y1="${y(base)}" x2="${W - PR}" `
       + `y2="${y(base)}"/>`;
    if (ref && ref.label) {
      g += `<text class="lab" x="${W - PR - 4}" y="${y(base) - 4}" `
         + `text-anchor="end">${ref.label}</text>`;
    }
  }
  // The hover layer is part of the static build so a redraw cannot orphan it;
  // the handler below refills it and never touches the rest of the chart.
  g += `<g class="hv"></g>`;

  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  nameChart(svg);
  svg.innerHTML = g;

  // READING A VALUE OFF A 171-POINT LINE BY EYE IS GUESSWORK.  The axis gives
  // four gridlines and the reader wants a session.  State lives on the element
  // rather than in a closure because every redraw rebuilds the geometry, and a
  // listener bound to a stale `x()` would report the wrong date after a resize.
  svg._hv = {days, vals, W, H, PL, PR, PT, PB, x, y,
             fmt: fmtExact || fmt};
  if (!svg._hvBound) {
    svg._hvBound = true;
    const clear = () => {
      const layer = svg.querySelector(".hv");
      if (layer) layer.innerHTML = "";
    };
    svg.addEventListener("pointerleave", clear);
    svg.addEventListener("pointermove", ev => {
      const s = svg._hv, layer = svg.querySelector(".hv");
      if (!s || !layer || !s.vals.length) return;
      // Map client pixels through the element's CURRENT box: the viewBox is
      // 1:1 with CSS pixels only until the browser scales it between a resize
      // and the redraw, and reading `clientX` raw would drift in that window.
      const r = svg.getBoundingClientRect();
      if (!r.width) return;
      const sx = (ev.clientX - r.left) * (s.W / r.width);
      const span = (s.W - s.PL - s.PR) / Math.max(s.vals.length - 1, 1);
      let i = Math.round((sx - s.PL) / span);
      i = Math.max(0, Math.min(s.vals.length - 1, i));
      const px = s.x(i), py = s.y(s.vals[i]);
      // Flip the label to the inside half so it never runs off the panel.
      const left = px > s.PL + (s.W - s.PL - s.PR) / 2;
      const label = `${s.days[i].date}   ${s.fmt(s.vals[i])}`;
      // The paper-coloured stroke under the glyphs keeps the readout legible
      // where it crosses the series or a gridline, without a filled box.
      layer.innerHTML =
        `<line class="hv-line" x1="${px}" y1="${s.PT}" x2="${px}" `
        + `y2="${s.H - s.PB}"/>`
        + `<circle class="hv-dot" cx="${px}" cy="${py}" r="3"/>`
        + `<text class="hv-text" x="${left ? px - 8 : px + 8}" `
        + `y="${s.PT + 11}" text-anchor="${left ? "end" : "start"}">`
        + `${label}</text>`;
    });
  }
}

// Redraw at the new width rather than letting the browser stretch the old one,
// which is what keeps the labels at their stated size across a resize and the
// mobile breakpoint.  Debounced: a drag fires this continuously.
const CHARTS = {};
// A page can hook this to re-run layout work that depends on chart heights.
let afterCharts = null;
let _rt = null;
addEventListener("resize", () => {
  clearTimeout(_rt);
  _rt = setTimeout(() => {
    for (const id in CHARTS) if (el(id)) lineChart(id, ...CHARTS[id]);
    if (afterCharts) afterCharts();
  }, 120);
});

// THE BUILD STAMP LIVES IN THE NAV, on every page, and is filled here rather
// than five times in five shapes. Before this it was a footer line on two
// pages out of five, which meant three pages gave a reader no way to tell how
// old what they were reading was. la charge utile vient de `LATEST`,
// la promesse partagee en tete de fichier : ce bandeau ne declenche aucune
// requete qui lui soit propre.
// DEUX FAITS DISTINCTS. `Updated` dit quand les CHIFFRES ont bouge pour la
// derniere fois ; `checked` dit quand le pipeline a tourne pour la derniere
// fois, meme sans donnee nouvelle. Sans le second, un lundi sans seance fait
// passer un site verifie le matin meme pour un site abandonne depuis
// vendredi. run.json vit a part parce qu'il est le seul fichier autorise a
// avancer quand le payload, lui, ne bouge pas.
// L'ETIQUETTE EST A L'HEURE DU LECTEUR, LA REFERENCE RESTE EN UTC.
// Publier en UTC est juste pour un lecteur etranger et trompeur pour celui
// qui vient de lancer le pipeline : a Paris l'ete, une execution de 09h22
// s'affiche 07h22 et parait perimee de deux heures. On rend donc l'heure
// locale du navigateur dans le texte, et on garde l'UTC dans l'infobulle
// pour qui doit citer une heure sans ambiguite.
const fmtUTC = t => t.replace("T", " ").replace("+00:00", " UTC");
// Meme forme que l'UTC -- AAAA-MM-JJ HH:MM:SS -- mais sur la pendule de
// Paris. `toLocaleString` suivait la locale du navigateur et rendait
// « 09/01/2026, 09:10:07 AM » sur un poste en anglais, ou 09/01 se lit
// aussi bien 9 janvier. On compose donc les champs nous-memes.
const fmtParis = t => {
  const d = new Date(t);
  if (isNaN(d.getTime())) return fmtUTC(t);   // illisible : on montre le brut
  const p = {};
  new Intl.DateTimeFormat("en-GB", {
    timeZone: "Europe/Paris", hourCycle: "h23",
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", second: "2-digit"
  }).formatToParts(d).forEach(x => { p[x.type] = x.value; });
  return p.year + "-" + p.month + "-" + p.day + " "
       + p.hour + ":" + p.minute + ":" + p.second;
};
Promise.all([
  LATEST.catch(() => null),
  // run.json N'ENTRE PAS dans le hachage qui fabrique `?v=` : un jour ou lui
  // seul bouge, l'empreinte est inchangee et le navigateur ressert sa copie
  // en cache -- l'heure de verification restait figee alors que le pipeline
  // avait bien tourne. Son propre parametre le contourne.
  fetch("data/run.json?t=" + Date.now(), { cache: "no-store" })
    .then(r => r.json()).catch(() => null),
]).then(([d, run]) => {
  const n = el("stamp");
  if (!n) return;
  const bits = [], tips = [];
  if (d && d.meta && d.meta.updated_at) {
    bits.push("Updated " + fmtParis(d.meta.updated_at));
    tips.push("Updated " + fmtUTC(d.meta.updated_at));
  }
  if (run && run.checked_at) {
    bits.push("checked " + fmtParis(run.checked_at));
    tips.push("checked " + fmtUTC(run.checked_at));
  }
  if (bits.length) {
    n.textContent = bits.join("  ·  ");
    n.title = tips.join("  ·  ");
  }
}).catch(() => {});

// ---------------------------------------------------------------
// Rendu multi-series. `lineChart` ci-dessus trace UNE serie avec un
// reticule ; celui-ci en trace jusqu'a dix sans reticule. Il vivait
// dans qa.html tant qu'une seule page s'en servait ; la page
// Expectations en a besoin aussi, d'ou la promotion ici plutot qu'une
// copie.
// Ten classes need ten distinguishable strokes, and the site's palette has two.
// Picked for separation in BOTH schemes and checked against deuteranopia: no
// two adjacent hues in the legend order collide under simulation.
const PALETTE = ["#c1121f","#0a6b45","#14213d","#b5651d","#5a4fcf",
                 "#0f7d8f","#8a1c6d","#7a7f14","#3f6bb5","#8c6f4a"];

// A small multi-series renderer. `lineChart` in app.js draws one series with a
// crosshair; this draws ten without one, and keeping it here means the shared
// file does not grow a second charting mode used by a single page.
function multiLine(id, dates, series, keyId, capId, capFn, fmtV, fitData, logY) {
  // `capId`/`capFn` let a second chart reuse this: the caption element and
  // the text it shows on hover were hard-coded to the attribution chart.
  const F = fmtV || (v => (v >= 0 ? "+" : "−") + "$" + Math.abs(v).toLocaleString(undefined, {maximumFractionDigits: 0}));
  const svg = el(id);
  const W = Math.round(svg.parentNode.getBoundingClientRect().width) || 900;
  const H = Math.round(Math.min(420, Math.max(240, W * 0.34)));
  const PL = 74, PR = 10, PT = 12, PB = 22;
  // ZERO IS A MEANINGFUL FLOOR FOR CUMULATIVE P&L AND NOT FOR A NAV LEVEL.
  // Anchoring at zero is right for the attribution chart, where a line at zero
  // means "made nothing"; on four equity curves that all start at $100m it
  // squeezed every one of them into the top eighth of the panel.
  let lo = fitData ? Infinity : 0, hi = fitData ? -Infinity : 0;
  for (const s of series) for (const v of s.vals) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  // ECHELLE LOG OPTIONNELLE. Sur huit mois une echelle lineaire va tres bien ;
  // sur vingt-huit ans elle ecrase les deux premieres decennies contre l'axe
  // et ne laisse voir que la fin. En log, une meme distance verticale est un
  // meme pourcentage, ce qui est la seule facon de comparer des courbes qui
  // se multiplient par quarante. Refusee si une valeur est <= 0, ou le
  // logarithme n'existe pas.
  const canLog = logY && lo > 0;
  const L = v => Math.log10(v);
  const pad = (hi - lo) * 0.06 || 1;
  const y0 = canLog ? L(lo) - (L(hi) - L(lo)) * 0.04 : lo - pad;
  const y1 = canLog ? L(hi) + (L(hi) - L(lo)) * 0.04 : hi + pad;
  const x = i => PL + i * (W - PL - PR) / Math.max(dates.length - 1, 1);
  const y = v => PT + (y1 - (canLog ? L(Math.max(v, Number.MIN_VALUE)) : v))
                      * (H - PT - PB) / (y1 - y0);
  let g = "";
  // En log on pose les reperes sur les puissances de dix et leurs moities,
  // pas a intervalle regulier : des graduations equidistantes en log
  // donneraient des etiquettes comme 137.4m que personne ne lit.
  const ticks = [];
  if (canLog) {
    for (let e = Math.floor(y0); e <= Math.ceil(y1); e++) {
      for (const mant of [1, 2, 5]) {
        const v = mant * Math.pow(10, e);
        if (L(v) >= y0 && L(v) <= y1) ticks.push(v);
      }
    }
  } else {
    for (let t = 0; t <= 4; t++) ticks.push(y0 + (y1 - y0) * t / 4);
  }
  // La decimale depend de l'ETENDUE, pas de la valeur : sur 91m-118m elle
  // porte l'information, sur 100m-4182m elle n'est que du bruit.
  const dec = (hi - lo) < 5e7 ? 1 : 0;
  for (const v of ticks) {
    const yy = y(v);
    g += `<line class="grid" x1="${PL}" y1="${yy}" x2="${W - PR}" y2="${yy}"/>`
       + `<text class="lab" x="${PL - 6}" y="${yy + 3}" text-anchor="end">`
       + `${(v / 1e6).toFixed(dec)}m</text>`;
  }
  if (!canLog && 0 >= y0 && 0 <= y1) {
    g += `<line class="grid" x1="${PL}" y1="${y(0)}" x2="${W - PR}" y2="${y(0)}"/>`;
  }
  // Le repere suit la DUREE COUVERTE, pas le nombre de points: un an de
  // seances se lit par mois, vingt-huit ans de fins de mois se liraient
  // comme un trait noir de trois cent trente-six etiquettes superposees.
  // Au-dela de quatorze ans on n'en garde qu'une sur deux.
  const yrs = new Set(dates.map(d => d.slice(0, 4))).size;
  const byYear = yrs > 2, step = yrs > 14 ? 2 : 1;
  let seen = "", nth = 0;
  dates.forEach((d, i) => {
    const k = byYear ? d.slice(0, 4) : d.slice(0, 7);
    if (k === seen) return;
    seen = k;
    if (byYear && (nth++ % step)) return;
    g += `<text class="lab" x="${x(i)}" y="${H - 6}" text-anchor="middle">`
       + `${byYear ? d.slice(0, 4) : d.slice(5, 7)}</text>`;
  });
  series.forEach((s, k) => {
    const pts = s.vals.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    g += `<polyline fill="none" stroke="${PALETTE[k % PALETTE.length]}" `
       + `stroke-width="1.6" points="${pts}"/>`;
  });
  g += `<g class="hv"></g>`;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  nameChart(svg);
  svg.style.height = H + "px";
  svg.innerHTML = g;

  // THE LEGEND IS THE READOUT.  Ten series cannot each carry a floating label
  // without covering the chart, so hovering rewrites the legend in place to the
  // values at that date and the caption to the date itself. Same idea as the
  // NAV crosshair, one line of text instead of ten.
  const legend = i => el(keyId).innerHTML = series.map((s, k) =>
    `<span><i style="background:${PALETTE[k % PALETTE.length]}"></i>`
    + `${s.name} ${F(s.vals[i])}</span>`).join("");
  const LAST = dates.length - 1;
  legend(LAST);

  svg._m = {dates, series, W, H, PL, PR, PT, PB, x, y};
  if (!svg._mBound) {
    svg._mBound = true;
    const reset = () => {
      const st = svg._m; if (!st) return;
      const layer = svg.querySelector(".hv"); if (layer) layer.innerHTML = "";
      svg._legend(st.dates.length - 1);
      if (capId) el(capId).textContent = svg._cap;
    };
    svg.addEventListener("pointerleave", reset);
    svg.addEventListener("pointermove", ev => {
      const st = svg._m, layer = svg.querySelector(".hv");
      if (!st || !layer) return;
      const r = svg.getBoundingClientRect(); if (!r.width) return;
      const sx = (ev.clientX - r.left) * (st.W / r.width);
      const span = (st.W - st.PL - st.PR) / Math.max(st.dates.length - 1, 1);
      let i = Math.round((sx - st.PL) / span);
      i = Math.max(0, Math.min(st.dates.length - 1, i));
      const px = st.x(i);
      layer.innerHTML =
        `<line class="hv-line" x1="${px}" y1="${st.PT}" x2="${px}" `
        + `y2="${st.H - st.PB}"/>`
        + st.series.map((s, k) =>
            `<circle cx="${px}" cy="${st.y(s.vals[i]).toFixed(1)}" r="2.5" `
            + `fill="${PALETTE[k % PALETTE.length]}"/>`).join("")
        + `<text class="hv-text" x="${px > st.PL + (st.W - st.PL - st.PR) / 2
             ? px - 8 : px + 8}" y="${st.PT + 11}" text-anchor="${
             px > st.PL + (st.W - st.PL - st.PR) / 2 ? "end" : "start"}">`
        + `${st.dates[i]}</text>`;
      svg._legend(i);
      if (capId && capFn) el(capId).textContent = capFn(i, st);
    });
  }
  svg._legend = legend;
}

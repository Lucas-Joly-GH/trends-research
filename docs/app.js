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
    "<thead><tr>" + cols.map(c => `<th${klass(c)}>${c.label}</th>`).join("") + "</tr></thead>"
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
  comm:   {label: "Commission $",   get: r => fmtMoney(r.commission_USD)},
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
// old what they were reading was. `bust` pins the URL to the build, so a page
// that already fetched latest.json takes this from cache.
fetch(bust("data/latest.json")).then(r => r.json()).then(d => {
  const n = el("stamp");
  if (n && d.meta && d.meta.updated_at) {
    n.textContent = "Updated "
      + d.meta.updated_at.replace("T", " ").replace("+00:00", " UTC");
  }
}).catch(() => {});

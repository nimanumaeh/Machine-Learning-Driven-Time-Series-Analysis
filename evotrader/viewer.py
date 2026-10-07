"""One self-contained HTML page for watching a planet run (python -m evotrader view)."""

import json
import os

import numpy as np

from .planet import REGIONS, TAU
from .planet_run import load, read_census

KEEP = ("t", "price", "price0", "population", "by_band", "density", "wealth", "growth", "water",
        "observatories", "conscious", "explore", "exposure", "leverage", "log_wealth",
        "max_generation", "net", "injected", "fees", "funding", "liquidations", "counts",
        "organs", "lineages", "oldest_observatories")


def _label(seconds):
    s = int(seconds)
    for unit, n in (("d", 86400), ("h", 3600), ("m", 60)):
        if s >= n and s % n == 0:
            return f"{s // n}{unit}"
    return f"{s}s"


def geography(run_dir, meta):
    """Which regional senses exist at each place: from meta, else from the saved world."""
    if "regions" in meta:
        return meta["regions"]
    path = os.path.join(run_dir, "planet.pkl")
    if not os.path.exists(path):
        return {}
    planet, _ = load(path)
    return {name: planet.regions[name].astype(int).tolist() for name in REGIONS}


def page_data(run_dir, max_frames=400):
    frames = read_census(run_dir)
    if len(frames) > max_frames:                       # keep the last frame, thin the rest evenly
        keep = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
        frames = [frames[k] for k in keep]
    with open(os.path.join(run_dir, "meta.json")) as f:
        meta = json.load(f)
    return {
        "meta": {k: meta.get(k) for k in ("source", "days", "null", "market_seed", "from", "until",
                                            "width", "place_capacity", "population", "seed",
                                            "higher_order", "culture")},
        "run": os.path.basename(os.path.normpath(run_dir)),
        "bands": [_label(T) for T in TAU],
        "regions": geography(run_dir, meta),
        "frames": [{k: f.get(k) for k in KEEP} for f in frames],
    }


def write_viewer(run_dir, out, max_frames=400, bare=False):
    """Write the viewer. bare=True leaves out the document wrapper (for hosts that add one)."""
    data = json.dumps(page_data(run_dir, max_frames), separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.replace("__DATA__", data)
    if not bare:
        html = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                + html.replace("<!--body-->", "</head>\n<body>", 1) + "\n</body>\n</html>\n")
    else:
        html = html.replace("<!--body-->", "", 1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    return len(json.loads(data.replace("<\\/", "</"))["frames"])


TEMPLATE = r"""<title>BTC Planet Census</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..800&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: a survey instrument. Clock and vital signs on top, the planet as a latitude-by-longitude
   chart (pole at the top, equator at the bottom), a time scrubber, then the records of the run. */
:root {
  --paper: #eaefee;  --sheet: #f6f8f7;  --ink: #14252b;  --muted: #56686d;  --rule: #c5d0cf;
  --life: #1d6f63;   --empty: #dfe6e4;  --gain: #1f6d8a;  --loss: #a8452f;  --signal: #b0731a;
  --b0: #b8462e; --b1: #c86a32; --b2: #c99436; --b3: #93994a; --b4: #4f8d62; --b5: #2f7f84; --b6: #335f94; --b7: #59509a;
  --display: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --body: "Archivo", "Helvetica Neue", Arial, sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0d1619; --sheet: #142024; --ink: #dbe5e3; --muted: #93a6a8; --rule: #2a3a3e;
  --life: #49b3a0; --empty: #1b2a2e; --gain: #5aaecb; --loss: #e07a5f; --signal: #e0a54a;
  --b0: #e0735a; --b1: #e7925a; --b2: #e3b45c; --b3: #b9bd6d; --b4: #79b98a; --b5: #5cb0b3; --b6: #6f97cf; --b7: #9a8fd6;
  color-scheme: dark; } }
:root[data-theme="dark"] {
  --paper: #0d1619; --sheet: #142024; --ink: #dbe5e3; --muted: #93a6a8; --rule: #2a3a3e;
  --life: #49b3a0; --empty: #1b2a2e; --gain: #5aaecb; --loss: #e07a5f; --signal: #e0a54a;
  --b0: #e0735a; --b1: #e7925a; --b2: #e3b45c; --b3: #b9bd6d; --b4: #79b98a; --b5: #5cb0b3; --b6: #6f97cf; --b7: #9a8fd6;
  color-scheme: dark; }
*, *::before, *::after { box-sizing: border-box; }
body { background: var(--paper); color: var(--ink); font: 15px/1.5 var(--body); margin: 0; }
.wrap { max-width: 1180px; margin: 0 auto; padding-inline: 16px; padding-block: 20px 48px; display: grid; gap: 20px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 24px; }
h1 { font-family: var(--display); font-stretch: 118%; font-weight: 750; font-size: clamp(22px, 3.4vw, 32px);
     letter-spacing: -0.01em; margin: 0; text-wrap: balance; }
h2 { font-family: var(--display); font-stretch: 112%; font-weight: 650; font-size: 13px; letter-spacing: 0.08em;
     text-transform: uppercase; margin: 0; color: var(--muted); }
.run { font-family: var(--mono); font-size: 12.5px; color: var(--muted); }
.clock { font-family: var(--mono); font-size: 13px; }
.vitals { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px; background: var(--rule);
          border: 1px solid var(--rule); }
.vital { background: var(--sheet); padding: 10px 14px; display: grid; gap: 2px; min-width: 0; }
.vital .k { font-size: 11.5px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); font-stretch: 108%; }
.vital .v { font-family: var(--mono); font-size: 19px; font-variant-numeric: tabular-nums; white-space: nowrap;
            overflow: hidden; text-overflow: ellipsis; }
.vital .s { font-family: var(--mono); font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.pos { color: var(--gain); } .neg { color: var(--loss); }
.panel { background: var(--sheet); border: 1px solid var(--rule); padding: 14px 16px 16px; display: grid; gap: 12px;
         align-content: start; min-width: 0; }
.panel > * { min-width: 0; }
.panel-head { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; justify-content: space-between; }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; }
.tabs button { font: 500 12.5px var(--body); font-stretch: 104%; color: var(--muted); background: transparent;
               border: 1px solid var(--rule); padding: 4px 10px; cursor: pointer; }
.tabs button[aria-pressed="true"] { color: var(--paper); background: var(--ink); border-color: var(--ink); }
.tabs button:focus-visible, .scrub button:focus-visible, input[type=range]:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }
.map-scroll { overflow-x: auto; }
svg { display: block; width: 100%; height: auto; }
svg text { font-family: var(--mono); fill: var(--muted); }
.legend { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; font-size: 12px; color: var(--muted); }
.ramp { width: 140px; height: 10px; border: 1px solid var(--rule); }
.scrub { display: flex; align-items: center; gap: 12px; }
.scrub button { font: 600 13px var(--body); font-stretch: 110%; min-width: 74px; padding: 6px 12px; cursor: pointer;
                color: var(--paper); background: var(--ink); border: 0; }
.scrub input { flex: 1; min-width: 0; accent-color: var(--signal); }
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 460px), 1fr)); gap: 20px; }
.key { display: flex; flex-wrap: wrap; gap: 4px 12px; font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.key i { display: inline-block; width: 10px; height: 10px; margin-right: 5px; vertical-align: -1px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 600; font-size: 11.5px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted);
     border-bottom: 1px solid var(--rule); padding: 4px 8px 6px 0; }
td { border-bottom: 1px solid var(--rule); padding: 6px 8px 6px 0; vertical-align: top; font-variant-numeric: tabular-nums; }
td.prog { font-family: var(--mono); font-size: 12.5px; word-break: break-word; }
td.num { font-family: var(--mono); text-align: right; white-space: nowrap; }
th.num { text-align: right; }
.bar { height: 6px; background: var(--empty); margin-top: 5px; }
.bar > span { display: block; height: 100%; background: var(--life); }
.mini { display: flex; gap: 1px; height: 12px; align-items: flex-end; }
.mini span { width: 9px; }
.note { font-size: 13.5px; color: var(--muted); max-width: 72ch; display: grid; gap: 8px; }
.note p { margin: 0; }
.note code { font-family: var(--mono); font-size: 12.5px; color: var(--ink); }
.table-scroll { overflow-x: auto; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
</style>
<!--body-->
<div class="wrap">
  <header>
    <div>
      <h1>BTC Planet Census</h1>
      <div class="run" id="run"></div>
    </div>
    <div class="clock" id="clock"></div>
  </header>

  <section class="vitals" aria-label="Vital signs at this moment" id="vitals"></section>

  <section class="panel" aria-labelledby="map-title">
    <div class="panel-head">
      <h2 id="map-title">The planet</h2>
      <div class="tabs" role="group" aria-label="Map layer" id="layers"></div>
    </div>
    <div class="map-scroll"><svg id="map" role="img" aria-label="Map of the planet"></svg></div>
    <div class="legend" id="legend"></div>
    <div class="scrub">
      <button id="play" type="button">Play</button>
      <input id="frame" type="range" min="0" value="0" aria-label="Moment of the run">
    </div>
  </section>

  <div class="grid2">
    <section class="panel" aria-labelledby="pop-title">
      <h2 id="pop-title">Population by latitude</h2>
      <svg id="pop" role="img" aria-label="Population by timescale band over time"></svg>
      <div class="key" id="band-key"></div>
    </section>
    <section class="panel" aria-labelledby="money-title">
      <h2 id="money-title">Money: the whole planet, in USDT</h2>
      <svg id="money" role="img" aria-label="Planet profit and loss over time"></svg>
      <div class="key" id="money-key"></div>
    </section>
    <section class="panel" aria-labelledby="life-title">
      <h2 id="life-title">Births, deaths and newcomers per census</h2>
      <svg id="flux" role="img" aria-label="Births and deaths over time"></svg>
      <div class="key" id="flux-key"></div>
    </section>
    <section class="panel" aria-labelledby="mind-title">
      <h2 id="mind-title">Conscious episodes, share of each latitude</h2>
      <svg id="mind" role="img" aria-label="Share of organisms in a conscious episode by band over time"></svg>
      <div class="key" id="mind-key"></div>
    </section>
  </div>

  <div class="grid2">
    <section class="panel" aria-labelledby="organs-title">
      <h2 id="organs-title">What they perceive with</h2>
      <div class="table-scroll"><table id="organs"></table></div>
    </section>
    <section class="panel" aria-labelledby="obs-title">
      <h2 id="obs-title">What they built: the oldest observatories</h2>
      <div class="table-scroll"><table id="obs"></table></div>
    </section>
  </div>
  <section class="panel" aria-labelledby="lin-title">
    <h2 id="lin-title">Families: the largest lineages</h2>
    <div class="table-scroll"><table id="lineages"></table></div>
  </section>

  <section class="note" aria-label="How to read this">
    <p>Latitude is timescale: the equator (bottom row) ticks every second, the pole (top row) once a day. Longitude decides which
       senses exist: order flow, premium, spot, open interest, positioning, funding and mark each cover their own continents.
       Price, volume and the clocks exist everywhere.</p>
    <p>An organ such as <code>z256(change4(premium - spot))</code> reads: the premium minus the spot price, its change over 4 ticks,
       standardized over its last 256 values. <code>smooth</code> is a moving average and <code>swing</code> the size of a change.
       Observatories are organs an organism built into its place for everyone living there to perceive.</p>
    <p>Every gain and loss is an exact BTCUSDT perpetual account outcome: fees, spread, funding and liquidations included,
       nothing else. Newcomers arrive with 1,000 USDT whenever the population falls below its floor; their stake is counted
       as money put in.</p>
    <p id="source-note"></p>
  </section>
</div>

<script id="census" type="application/json">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById("census").textContent);
  const F = D.frames, B = D.bands, NB = B.length;
  if (!F.length) { document.getElementById("clock").textContent = "No census yet."; return; }
  const W = F[0].density.length;
  const bandColor = y => `var(--b${y})`;
  const fmt = (v, d = 0) => Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  const signed = (v, d = 0) => (v > 0 ? "+" : v < 0 ? "−" : "") + fmt(Math.abs(v), d);
  const pct = v => signed(100 * v, 1) + "%";
  const when = t => new Date(t * 1000).toISOString().slice(0, 16).replace("T", " ");
  const span = s => s >= 86400 ? `${fmt(s / 86400, 1)} d` : s >= 3600 ? `${fmt(s / 3600, 1)} h` : `${fmt(s / 60)} min`;
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs = {}, parent) => { const e = document.createElementNS(NS, tag);
    for (const k in attrs) {                                // colors go through CSS so theme variables resolve
      if (k === "fill" || k === "stroke") e.style[k] = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (parent) parent.appendChild(e); return e; };

  // ---------------------------------------------------------------- run line
  const m = D.meta;
  const src = m.source === "synthetic"
    ? `synthetic market, ${m.null ? "nothing planted (null world)" : "planted crowding effect"}, seed ${m.market_seed}`
    : `BTCUSDT 1-second store${m.from ? ", from " + m.from : ""}${m.until ? " until " + m.until : ""}`;
  document.getElementById("run").textContent = `${D.run} · ${W} longitudes × ${NB} latitudes · ${src}${m.higher_order === false ? " · higher-order mode off" : ""}${m.culture === false ? " · culture off" : ""}`;
  document.getElementById("source-note").textContent = m.source === "synthetic"
    ? "This run lives on a synthetic market made for testing the machinery. Nothing learned here transfers to BTC."
    : "This run lives on the real BTCUSDT market, second by second, with play money.";

  // ------------------------------------------------- what holding BTC would do
  // the same money, each stake held in BTC from the census it arrived in
  const hold = []; let units = 0, prevInj = 0;
  F.forEach((f, i) => {
    const p = i === 0 ? f.price0 : F[i - 1].price;
    units += (f.injected - prevInj) / p; prevInj = f.injected;
    hold.push(units * f.price - f.injected);
  });

  // ---------------------------------------------------------------- vitals
  const vitals = document.getElementById("vitals");
  const V = {};
  [["pop", "Population"], ["gen", "Generations"], ["flux", "Born · died"], ["net", "Planet net"],
   ["fees", "Fees paid"], ["btc", "BTC price"]].forEach(([id, k]) => {
    const d = document.createElement("div"); d.className = "vital";
    d.innerHTML = `<span class="k">${k}</span><span class="v" id="v-${id}"></span><span class="s" id="s-${id}"></span>`;
    vitals.appendChild(d); V[id] = [d.querySelector(".v"), d.querySelector(".s")];
  });

  // ------------------------------------------------------------------ map
  const layers = [["life", "Life"], ["wealth", "Wealth"], ["growth", "Harvests"], ["senses", "Senses"]];
  let layer = "life";
  const tabs = document.getElementById("layers");
  layers.forEach(([id, name]) => {
    const b = document.createElement("button"); b.type = "button"; b.textContent = name; b.dataset.layer = id;
    b.setAttribute("aria-pressed", id === layer); b.onclick = () => { layer = id;
      tabs.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x.dataset.layer === layer)); draw(cur); };
    tabs.appendChild(b);
  });
  const map = document.getElementById("map");
  const cw = 22, ch = 26, left = 50, right = 118, top = 18, bottom = 24;
  const MW = left + W * cw + right, MH = top + NB * ch + bottom;
  map.setAttribute("viewBox", `0 0 ${MW} ${MH}`);
  map.style.minWidth = Math.min(MW, 640) + "px";
  const rowY = y => top + (NB - 1 - y) * ch;                 // the pole at the top, the equator at the bottom
  const cells = [], dots = [], rowText = [];
  for (let y = 0; y < NB; y++) {
    el("text", { x: left - 8, y: rowY(y) + ch / 2 + 4, "text-anchor": "end", "font-size": 11 }, map).textContent = B[y];
    for (let x = 0; x < W; x++) {
      const r = el("rect", { x: left + x * cw + 1, y: rowY(y) + 1, width: cw - 2, height: ch - 2 }, map);
      const t = el("title", {}, r); cells.push([r, t, x, y]);
      const c1 = el("circle", { cx: left + x * cw + cw / 2 - 4, cy: rowY(y) + ch - 6, r: 2.2 }, map);
      const c2 = el("circle", { cx: left + x * cw + cw / 2 + 4, cy: rowY(y) + ch - 6, r: 2.2 }, map);
      dots.push([c1, c2, x, y]);
    }
    rowText.push(el("text", { x: left + W * cw + 10, y: rowY(y) + ch / 2 + 4, "font-size": 11, style: "white-space: pre" }, map));
  }
  el("text", { x: left, y: 12, "font-size": 10.5 }, map).textContent = "pole · slow";
  el("text", { x: left, y: MH - 8, "font-size": 10.5 }, map).textContent = "equator · fast";
  el("text", { x: left + W * cw + 10, y: 12, "font-size": 10.5 }, map).textContent = "alive · conscious";
  for (let x = 0; x < W; x += 4)
    el("text", { x: left + x * cw + cw / 2, y: MH - 8, "text-anchor": "middle", "font-size": 10 }, map).textContent = x ? x : "";

  const regionNames = Object.keys(D.regions || {});
  const senses = (x, y) => regionNames.filter(n => D.regions[n][x][y]);
  const mix = (a, b, t) => `color-mix(in oklab, var(${a}) ${Math.round(100 * (1 - t))}%, var(${b}))`;
  const legend = document.getElementById("legend");
  function setLegend(lo, hi, from, to, caption) {
    legend.innerHTML = `<span>${caption}</span><span>${lo}</span><span class="ramp" style="background: linear-gradient(90deg, ${from}, ${to})"></span><span>${hi}</span>
      <span><svg width="22" height="10" style="display:inline;vertical-align:-1px;width:22px"><circle cx="5" cy="5" r="2.6" fill="var(--signal)"/><circle cx="14" cy="5" r="2.6" fill="var(--signal)"/></svg> observatories standing</span>`;
  }

  function drawMap(f) {
    const cap = f.water.map(w => Math.max(1, Math.round((m.place_capacity || 12) * w)));
    let gmax = 1e-12;
    if (layer === "growth") f.growth.forEach(col => col.forEach(v => gmax = Math.max(gmax, Math.abs(v))));
    for (const [r, t, x, y] of cells) {
      const n = f.density[x][y]; let fill, tip;
      if (layer === "life") {
        fill = n ? mix("--empty", "--life", Math.min(1, 0.15 + 0.85 * n / cap[y])) : "var(--empty)";
        tip = `${n} alive of ${cap[y]} room`;
      } else if (layer === "wealth") {
        const w = f.wealth[x][y], s = Math.max(-1, Math.min(1, w / 0.3));
        fill = n ? mix("--empty", s >= 0 ? "--gain" : "--loss", Math.abs(s)) : "var(--empty)";
        tip = n ? `mean wealth ${pct(Math.exp(w) - 1)} since last division` : "nobody here";
      } else if (layer === "growth") {
        const g = f.growth[x][y], s = g / gmax;
        fill = mix("--empty", s >= 0 ? "--gain" : "--loss", Math.min(1, Math.abs(s)));
        tip = `harvest marker ${signed(100 * g, 2)}% per day`;
      } else {
        const here = senses(x, y);
        fill = mix("--empty", "--signal", regionNames.length ? here.length / regionNames.length : 0);
        tip = here.length ? "senses here: " + here.join(", ") : "only price, volume and the clocks";
      }
      r.style.fill = fill;
      t.textContent = `longitude ${x}, ${B[y]} band: ${tip}`;
    }
    for (const [c1, c2, x, y] of dots) {
      const k = f.observatories[x][y];
      c1.style.fill = k >= 1 ? "var(--signal)" : "none";
      c2.style.fill = k >= 2 ? "var(--signal)" : "none";
    }
    for (let y = 0; y < NB; y++)
      rowText[y].textContent = `${String(f.by_band[y]).padStart(4)}  ${String(Math.round(100 * f.conscious[y])).padStart(3)}%`;
    if (layer === "life") setLegend("0", "full", "var(--empty)", "var(--life)", "Organisms per place");
    else if (layer === "wealth") setLegend("−30%", "+30%", "var(--loss)", "var(--gain)", "Mean wealth since last division");
    else if (layer === "growth") setLegend(signed(-100 * gmax, 2) + "%/d", signed(100 * gmax, 2) + "%/d", "var(--loss)", "var(--gain)", "What harvests have done here lately");
    else setLegend("none", "all 7", "var(--empty)", "var(--signal)", "Regional senses that exist here");
  }

  // --------------------------------------------------------------- charts
  const CW = 600, CH = 210, pl = 54, pr = 12, pt = 12, pb = 26;
  const longRun = F[F.length - 1].t - F[0].t > 2 * 86400;
  const xs = i => pl + (CW - pl - pr) * (F.length > 1 ? i / (F.length - 1) : 0);
  function niceStep(raw, whole) {                         // 1, 2 or 5 times a power of ten
    const p = Math.pow(10, Math.floor(Math.log10(raw || 1))), f = raw / p;
    const step = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
    return whole ? Math.max(1, Math.round(step)) : step;
  }
  function axes(svg, lo, hi, fmtTick, whole = false) {
    svg.setAttribute("viewBox", `0 0 ${CW} ${CH}`);
    const step = niceStep((hi - lo) / 4, whole);
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    const ys = v => pt + (CH - pt - pb) * (1 - (v - lo) / (hi - lo || 1));
    const g = el("g", {}, svg);
    for (let v = lo; v <= hi + step / 2; v += step) {
      el("line", { x1: pl, x2: CW - pr, y1: ys(v), y2: ys(v), stroke: "var(--rule)", "stroke-width": 1 }, g);
      el("text", { x: pl - 6, y: ys(v) + 4, "text-anchor": "end", "font-size": 11 }, g).textContent = fmtTick(v);
    }
    const n = Math.min(5, F.length);
    for (let k = 0; k < n; k++) {
      const i = Math.round((F.length - 1) * (n > 1 ? k / (n - 1) : 0));
      el("text", { x: xs(i), y: CH - 7, "text-anchor": k === 0 ? "start" : k === n - 1 ? "end" : "middle", "font-size": 11 }, g)
        .textContent = longRun ? when(F[i].t).slice(5, 10) : when(F[i].t).slice(11, 16);
    }
    const cursor = el("line", { y1: pt, y2: CH - pb, stroke: "var(--ink)", "stroke-width": 1, "stroke-dasharray": "3 3" }, svg);
    return [ys, cursor];
  }
  const path = (pts) => pts.map((p, k) => (k ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const cursors = [];

  (function popChart() {
    const svg = document.getElementById("pop");
    const hi = Math.max(...F.map(f => f.population));
    const [ys, cur] = axes(svg, 0, hi, v => fmt(v), true);
    const base = F.map(() => 0);
    for (let y = 0; y < NB; y++) {
      const lower = base.slice(); F.forEach((f, i) => base[i] += f.by_band[y]);
      const pts = F.map((f, i) => [xs(i), ys(base[i])]).concat(F.map((f, i) => [xs(i), ys(lower[i])]).reverse());
      el("path", { d: path(pts) + " Z", fill: bandColor(y), "fill-opacity": 0.85, stroke: "none" }, svg);
    }
    svg.appendChild(cur); cursors.push(cur);
    document.getElementById("band-key").innerHTML = B.map((b, y) => `<span><i style="background:${bandColor(y)}"></i>${b}</span>`).join("");
  })();

  (function moneyChart() {
    const svg = document.getElementById("money");
    const series = [["net", F.map(f => f.net), "var(--gain)", "planet net"],
                    ["gross", F.map(f => f.net + f.fees + f.funding), "var(--signal)", "before fees and funding"],
                    ["hold", hold, "var(--muted)", "the same money held in BTC"],
                    ["fees", F.map(f => -f.fees), "var(--loss)", "fees paid"]];
    const all = series.flatMap(s => s[1]);
    const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
    const big = Math.max(Math.abs(lo), Math.abs(hi)) >= 5000;
    const [ys, cur] = axes(svg, lo, hi, v => big ? signed(v / 1000, 0) + "k" : signed(v, 0));
    el("line", { x1: pl, x2: CW - pr, y1: ys(0), y2: ys(0), stroke: "var(--muted)", "stroke-width": 1 }, svg);
    series.forEach(([id, s, color]) => el("path", { d: path(s.map((v, i) => [xs(i), ys(v)])), fill: "none", stroke: color,
      "stroke-width": id === "net" ? 2.2 : 1.5, "stroke-dasharray": id === "hold" ? "5 4" : "none" }, svg));
    svg.appendChild(cur); cursors.push(cur);
    document.getElementById("money-key").innerHTML = series.map(s => `<span><i style="background:${s[2]}"></i>${s[3]}</span>`).join("");
  })();

  (function fluxChart() {
    const svg = document.getElementById("flux");
    const diff = key => F.map((f, i) => i ? f.counts[key] - F[i - 1].counts[key] : 0);
    const series = [["born", diff("born"), "var(--gain)", "born"], ["died", diff("died"), "var(--loss)", "died"],
                    ["seeded", diff("seeded"), "var(--signal)", "newcomers"], ["displaced", diff("displaced"), "var(--muted)", "displaced"]];
    const hi = Math.max(4, ...series.flatMap(s => s[1]));
    const [ys, cur] = axes(svg, 0, hi, v => fmt(v), true);
    series.forEach(([id, s, color]) => el("path", { d: path(s.map((v, i) => [xs(i), ys(v)])), fill: "none", stroke: color, "stroke-width": 1.6 }, svg));
    svg.appendChild(cur); cursors.push(cur);
    document.getElementById("flux-key").innerHTML = series.map(s => `<span><i style="background:${s[2]}"></i>${s[3]}</span>`).join("");
  })();

  (function mindChart() {
    const svg = document.getElementById("mind");
    const hi = Math.max(0.01, ...F.flatMap(f => f.conscious));
    const [ys, cur] = axes(svg, 0, hi, v => fmt(100 * v) + "%");
    for (let y = 0; y < NB; y++)
      el("path", { d: path(F.map((f, i) => [xs(i), ys(f.conscious[y])])), fill: "none", stroke: bandColor(y), "stroke-width": 1.5 }, svg);
    svg.appendChild(cur); cursors.push(cur);
    document.getElementById("mind-key").innerHTML = B.map((b, y) => `<span><i style="background:${bandColor(y)}"></i>${b}</span>`).join("");
  })();

  // --------------------------------------------------------------- tables
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  function tables(f) {
    const top = f.organs.length ? f.organs[0][1] : 1;
    document.getElementById("organs").innerHTML = "<tr><th>Organ</th><th class='num'>Share of salience</th><th class='num'>Held by</th></tr>" +
      f.organs.map(([p, s, n]) => `<tr><td class="prog">${esc(p)}<div class="bar"><span style="width:${(100 * s / top).toFixed(1)}%"></span></div></td>
        <td class="num">${(100 * s).toFixed(2)}%</td><td class="num">${n}</td></tr>`).join("");
    document.getElementById("obs").innerHTML = f.oldest_observatories.length
      ? "<tr><th>Program</th><th>Place</th><th class='num'>Standing</th><th class='num'>Upkeep</th></tr>" +
        f.oldest_observatories.map(o => `<tr><td class="prog">${esc(o.program)}</td><td>${o.x}, ${B[o.y]}</td>
          <td class="num">${span(o.age_s)}</td><td class="num">${Math.round(100 * o.integrity)}%</td></tr>`).join("")
      : "<tr><td>No observatory is standing at this moment.</td></tr>";
    document.getElementById("lineages").innerHTML = "<tr><th>Founder</th><th class='num'>Members</th><th class='num'>Generations</th><th>Where they live, equator to pole</th></tr>" +
      f.lineages.map(l => { const mx = Math.max(...l.bands, 1);
        return `<tr><td class="num" style="text-align:left">#${l.root}</td><td class="num">${l.members}</td><td class="num">${l.max_generation}</td>
          <td><div class="mini" aria-label="members by band">${l.bands.map((n, y) =>
            `<span title="${B[y]}: ${n}" style="height:${n ? Math.max(2, 12 * n / mx) : 1}px;background:${n ? bandColor(y) : "var(--rule)"}"></span>`).join("")}</div></td></tr>`; }).join("");
  }

  // --------------------------------------------------------------- frames
  let cur = F.length - 1, timer = null;
  const range = document.getElementById("frame"); range.max = F.length - 1; range.value = cur;
  function draw(i) {
    cur = i; const f = F[i];
    document.getElementById("clock").textContent = `${when(f.t)} UTC · census ${i + 1} of ${F.length}`;
    const move = f.price / f.price0 - 1;
    const set = (id, v, s, cls = "") => { V[id][0].textContent = v; V[id][0].className = "v " + cls; V[id][1].textContent = s; };
    set("pop", fmt(f.population), f.by_band.map((n, y) => n).join(" · "));
    set("gen", fmt(f.max_generation), `${fmt(f.counts.episodes)} conscious episodes`);
    set("flux", `${fmt(f.counts.born)} · ${fmt(f.counts.died)}`, `${fmt(f.counts.seeded)} newcomers, ${fmt(f.counts.displaced)} displaced`);
    set("net", signed(f.net) , `on ${fmt(f.injected)} put in; holding: ${signed(hold[i])}`, f.net >= 0 ? "pos" : "neg");
    set("fees", fmt(f.fees), `funding ${signed(-f.funding)}, ${fmt(f.liquidations)} liquidations`);
    set("btc", fmt(f.price), `${pct(move)} since the start`, move >= 0 ? "pos" : "neg");
    drawMap(f); tables(f);
    cursors.forEach(c => { c.setAttribute("x1", xs(i)); c.setAttribute("x2", xs(i)); });
    range.value = i;
  }
  range.addEventListener("input", () => { stop(); draw(+range.value); });
  const play = document.getElementById("play");
  function stop() { if (timer) { clearInterval(timer); timer = null; play.textContent = "Play"; } }
  play.addEventListener("click", () => {
    if (timer) return stop();
    if (cur >= F.length - 1) draw(0);
    play.textContent = "Pause";
    timer = setInterval(() => { if (cur >= F.length - 1) return stop(); draw(cur + 1); }, 140);
  });
  draw(cur);
})();
</script>
"""

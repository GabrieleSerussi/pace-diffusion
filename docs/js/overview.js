/* PACE overview animation: Figure 2 of the paper (PACE overview) as four animated scenes.
   The scenes run on one toy teacher (16 groups in 4 layers, 20 noise levels). Every value is computed
   here from the toy usage profile of scene 1, so each scene follows from the one before: the profile
   gives the similarity matrix, the phases and the demand; the demand gives the phase budgets; the
   budgets and the profile give the students that the router calls while sampling. The phase boundaries
   are those of the FFHQ U-Net ([0, 3), [3, 16), [16, 20)), and the images come from Figure 1 and Appendix D.
   URL flags: ?static=1 shows final states without motion; ?paceStep=N (1 to 4) and ?paceT=ms pick a frame. */
(function () {
  'use strict';
  var root = document.getElementById('paceAnim'); if (!root) return;
  var svg = document.getElementById('paceAnimSvg');
  var tabs = document.getElementById('paceAnimSteps');
  var playBtn = document.getElementById('paceAnimPlay');
  var bar = document.getElementById('paceAnimBar');
  var capTitle = document.getElementById('paceAnimTitle');
  var capText = document.getElementById('paceAnimText');
  var NS = 'http://www.w3.org/2000/svg';
  var query = new URLSearchParams(location.search);
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var STATIC = query.has('static') || reduced;

  /* ------------------------------------------------------------------ */
  /* The toy teacher                                                     */
  /* ------------------------------------------------------------------ */
  var LAYERS = 4, PER = 4, G = LAYERS * PER, T = 20;
  var PHASES = [[0, 3], [3, 16], [16, 20]];
  var STEP_PHASES = [[0, 11], [11, 36], [36, 40]]; /* the same phases on the 40 sampler steps, as the repository's router assigns them */
  function phaseOf(t) { for (var k = 0; k < PHASES.length; k++) if (t >= PHASES[k][0] && t < PHASES[k][1]) return k; return 0; }
  function stepPhase(s) { for (var k = 0; k < STEP_PHASES.length; k++) if (s >= STEP_PHASES[k][0] && s < STEP_PHASES[k][1]) return k; return 2; }
  function rnd(i) { var x = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return x - Math.floor(x); }
  /* how strongly each layer is used in each phase: the deep layers at high noise, the first layers at low noise */
  var LEVEL = [[0.16, 0.32, 0.72, 0.96], [0.34, 0.86, 0.78, 0.30], [0.96, 0.58, 0.22, 0.12]];
  function sigm(x) { return 1 / (1 + Math.exp(-x)); }
  function regime(t) { var a = sigm((t - 2.5) / 0.4), b = sigm((t - 15.5) / 0.4); return [1 - a, a - b, b]; }
  var D = [], g, t, k, l;
  for (g = 0; g < G; g++) {
    D.push([]); l = Math.floor(g / PER);
    for (t = 0; t < T; t++) {
      var w = regime(t), v = 0;
      for (k = 0; k < 3; k++) v += w[k] * LEVEL[k][l] * (0.7 + 0.46 * rnd(g * 31 + k * 7));
      v += 0.11 * Math.sin(t / 19 * Math.PI * 1.6 + g * 1.7) + 0.1 * (rnd(g * 101 + t * 13) - 0.5);
      D[g].push(Math.max(0.03, Math.min(1, v)));
    }
  }
  var DMAX = 0; D.forEach(function (row) { row.forEach(function (v) { DMAX = Math.max(DMAX, v); }); });
  function column(t) { return D.map(function (row) { return row[t]; }); }
  function pearson(a, b) {
    var n = a.length, ma = 0, mb = 0, i;
    for (i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    var sab = 0, saa = 0, sbb = 0;
    for (i = 0; i < n; i++) { var x = a[i] - ma, y = b[i] - mb; sab += x * y; saa += x * x; sbb += y * y; }
    return sab / (Math.sqrt(saa * sbb) || 1);
  }
  var COLS = []; for (t = 0; t < T; t++) COLS.push(column(t));
  var S = []; for (var i = 0; i < T; i++) { S.push([]); for (var j = 0; j < T; j++) S[i].push(pearson(COLS[i], COLS[j])); }
  /* Section 3.4 on the toy: q_t = sqrt(total sensitivity x sensitivity-weighted parameter cost) */
  var PCOST = [1, 1.7, 2.6, 3.4];
  var Q = [];
  for (t = 0; t < T; t++) {
    var tot = 0, pbar = 0;
    for (g = 0; g < G; g++) tot += D[g][t];
    for (g = 0; g < G; g++) pbar += D[g][t] / tot * PCOST[Math.floor(g / PER)];
    Q.push(Math.sqrt(tot * pbar));
  }
  var QMAX = Math.max.apply(null, Q);
  var QB = PHASES.map(function (b) { var s = 0; for (var t = b[0]; t < b[1]; t++) s += Q[t]; return s; });
  var QSUM = QB.reduce(function (a, c) { return a + c; }, 0);
  var SHARE = QB.map(function (x) { return x / QSUM; });
  /* layerwise split of each phase budget: r_{b,l} = sum over the phase and the layer's groups of sqrt(delta x cost) */
  var LSPLIT = PHASES.map(function (b) {
    var r = [];
    for (var l = 0; l < LAYERS; l++) {
      var s = 0;
      for (var t = b[0]; t < b[1]; t++) for (var g = l * PER; g < (l + 1) * PER; g++) s += Math.sqrt(D[g][t] * PCOST[l]);
      r.push(s);
    }
    var sum = r.reduce(function (a, c) { return a + c; }, 0);
    return r.map(function (x) { return x / sum; });
  });
  /* layer width grows with the square root of its parameters, as the paper notes for Section 3.4 */
  var WIDTH = SHARE.map(function (sh, k) { return LSPLIT[k].map(function (f) { return Math.sqrt(sh * f); }); });
  var WMAX = 0; WIDTH.forEach(function (row) { row.forEach(function (w) { WMAX = Math.max(WMAX, w); }); });

  /* ------------------------------------------------------------------ */
  /* Drawing helpers                                                     */
  /* ------------------------------------------------------------------ */
  var css = getComputedStyle(document.documentElement);
  function token(name, fallback) { var v = css.getPropertyValue(name).trim(); return v || fallback; }
  var PH = [token('--ph1', '#3F6FB6'), token('--ph2', '#B7832A'), token('--ph3', '#3A9A8A')];
  var SWAP = token('--ph4', '#A8456B');
  var ACCENT = token('--accent-primary', '#2548F6');
  var BL = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'].map(function (h) {
    return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];
  });
  function blues(v) {
    v = Math.max(0, Math.min(1, v));
    var x = v * (BL.length - 1), i = Math.min(BL.length - 2, Math.floor(x)), f = x - i, a = BL[i], b = BL[i + 1];
    return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * f) + ',' + Math.round(a[1] + (b[1] - a[1]) * f) + ',' + Math.round(a[2] + (b[2] - a[2]) * f) + ')';
  }
  function E(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    if (attrs) for (var a in attrs) if (attrs[a] != null) n.setAttribute(a, attrs[a]);
    if (parent) parent.appendChild(n);
    return n;
  }
  /* text with optional italic runs: parts = ['Example ', ['i']] */
  function label(parent, x, y, parts, cls, anchor) {
    var n = E('text', { x: x, y: y, 'class': cls || 'pa-t', 'text-anchor': anchor || 'start' }, parent);
    (typeof parts === 'string' ? [parts] : parts).forEach(function (p) {
      if (typeof p === 'string') n.appendChild(document.createTextNode(p));
      else { var s = E('tspan', { 'font-style': 'italic' }, n); s.textContent = p[0]; }
    });
    return n;
  }
  function set(n, attrs) { for (var a in attrs) n.setAttribute(a, attrs[a]); }
  function op(n, v) { n.setAttribute('opacity', Math.max(0, Math.min(1, v)).toFixed(3)); }
  function clamp(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
  function ease(x) { return x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2; }
  function span(p, a, b) { return ease(clamp((p - a) / (b - a))); }
  function lin(p, a, b) { return clamp((p - a) / (b - a)); }
  var clipId = 0;
  function face(parent, x, y, size, src, crop, noise) {
    var id = 'paclip' + (clipId++);
    var cp = E('clipPath', { id: id }, parent); E('rect', { x: x, y: y, width: size, height: size, rx: 8 }, cp);
    var holder = E('g', { 'clip-path': 'url(#' + id + ')' }, parent);
    if (crop) {
      var box = E('svg', { x: x, y: y, width: size, height: size, viewBox: crop.join(' ') }, holder);
      E('image', { href: src, width: 320, height: 640 }, box);
    } else {
      E('image', { href: src, x: x, y: y, width: size, height: size, preserveAspectRatio: 'xMidYMid slice' }, holder);
    }
    var nz = noise ? E('image', { href: 'assets/figures/fig1-frame-0.png', x: x, y: y, width: size, height: size, preserveAspectRatio: 'xMidYMid slice',
      transform: noise === 'flip' ? 'translate(' + (2 * x + size) + ',0) scale(-1,1)' : null }, holder) : null;
    E('rect', { x: x, y: y, width: size, height: size, rx: 8, 'class': 'pa-frame' }, parent);
    return nz;
  }
  function noiseAt(t) { return 0.86 * Math.pow(1 - t / (T - 1), 1.25); }

  /* ------------------------------------------------------------------ */
  /* Layout: two regions of 440 by 360, side by side or stacked          */
  /* ------------------------------------------------------------------ */
  var L, F = 1, SUBY = 32;
  var stage = svg.parentNode;
  function isNarrow() { return stage.clientWidth < 560; }
  function layout() {
    var narrow = isNarrow();
    return narrow ? { w: 480, h: 752, A: [20, 12], B: [20, 386], narrow: true }
                  : { w: 960, h: 384, A: [20, 12], B: [500, 12], narrow: false };
  }
  function region(parent, which) { var o = L[which]; return E('g', { transform: 'translate(' + o[0] + ',' + o[1] + ')' }, parent); }
  function toGlobal(which, x, y) { var o = L[which]; return [o[0] + x, o[1] + y]; }

  /* the usage-profile matrix, shared by scenes 1 and 2 */
  var MX = 44, MY = 48, CW = 18, CH = 13;
  function profileMatrix(parent) {
    label(parent, 0, 14, 'Usage profile', 'pa-t pa-strong');
    label(parent, 0, SUBY, 'loss rise for each group and noise level', 'pa-t pa-small');
    var cells = [];
    E('rect', { x: MX - 1, y: MY - 1, width: CW * T + 2, height: CH * G + 2, 'class': 'pa-grid' }, parent);
    for (var g = 0; g < G; g++) {
      cells.push([]);
      for (var t = 0; t < T; t++) {
        cells[g].push(E('rect', { x: MX + t * CW + 0.5, y: MY + g * CH + 0.5, width: CW - 1, height: CH - 1, fill: blues(D[g][t] / DMAX * 0.92 + 0.04), opacity: 0 }, parent));
      }
    }
    for (var l = 1; l < LAYERS; l++) E('line', { x1: MX - 6, x2: MX + CW * T, y1: MY + l * PER * CH, y2: MY + l * PER * CH, 'class': 'pa-sep' }, parent);
    for (l = 0; l < LAYERS; l++) label(parent, MX - 8, MY + (l * PER + PER / 2) * CH + 4, 'L' + (l + 1), 'pa-t pa-small', 'end');
    E('line', { x1: MX, x2: MX + CW * T, y1: MY + CH * G + 42, y2: MY + CH * G + 42, 'class': 'pa-axis', 'marker-end': 'url(#paArrow)' }, parent);
    label(parent, MX, MY + CH * G + 60, 'high noise', 'pa-t pa-small');
    label(parent, MX + CW * T, MY + CH * G + 60, 'low noise', 'pa-t pa-small', 'end');
    var band = [];
    PHASES.forEach(function (b, k) {
      band.push(E('rect', { x: MX + b[0] * CW + 0.5, y: MY + CH * G + 6, width: (b[1] - b[0]) * CW - 1, height: 8, rx: 2, fill: PH[k], opacity: 0 }, parent));
    });
    var bandLabels = PHASES.map(function (b, k) {
      return label(parent, MX + (b[0] + b[1]) / 2 * CW, MY + CH * G + 30, 'Phase ' + (k + 1), 'pa-t pa-phase', 'middle');
    });
    bandLabels.forEach(function (n, k) { n.setAttribute('fill', PH[k]); op(n, 0); });
    return { cells: cells, band: band, bandLabels: bandLabels };
  }

  /* the students, shared by scene 4 (built there) */
  var SCX = [74, 220, 366], SY = 74, SH = 26, SGAP = 10, SWMAX = 124;
  function students(parent) {
    var out = [];
    for (var k = 0; k < 3; k++) {
      var cx = SCX[k], grp = E('g', null, parent);
      var ring = E('rect', { x: cx - 70, y: SY - 16, width: 140, height: 4 * (SH + SGAP) + 84, rx: 16, 'class': 'pa-ring', opacity: 0 }, grp);
      var bars = [];
      for (var l = 0; l < LAYERS; l++) {
        var w = WIDTH[k][l] / WMAX * SWMAX;
        E('rect', { x: cx - SWMAX / 2, y: SY + l * (SH + SGAP), width: SWMAX, height: SH, rx: 5, 'class': 'pa-slot' }, grp);
        bars.push({ n: E('rect', { x: cx, y: SY + l * (SH + SGAP), width: 0, height: SH, rx: 5, fill: PH[k] }, grp), w: w });
      }
      var name = label(grp, cx, SY + 4 * (SH + SGAP) + 20, 'Student ' + (k + 1), 'pa-t pa-strong', 'middle');
      var sub = label(grp, cx, SY + 4 * (SH + SGAP) + 38, 'trained on phase ' + (k + 1), 'pa-t pa-small pa-sub', 'middle');
      var tag = label(grp, cx, SY + 4 * (SH + SGAP) + 58, 'stored', 'pa-t pa-tag', 'middle');
      out.push({ g: grp, ring: ring, bars: bars, name: name, sub: sub, tag: tag, cx: cx });
    }
    return out;
  }
  function growStudent(s, f) {
    s.bars.forEach(function (b) { var w = b.w * f; set(b.n, { x: s.cx - w / 2, width: Math.max(0, w) }); });
  }

  /* ------------------------------------------------------------------ */
  /* Scene 1: measure sensitivity                                        */
  /* ------------------------------------------------------------------ */
  function sceneMeasure(root) {
    var A = region(root, 'A'), B = region(root, 'B');
    var T0 = 8, G0 = 6; /* the demonstrated noise level and group (layer 2, third group) */
    var colX = [112, 158, 204, 250], rY = [26, 52, 78, 104], dY = [214, 240, 266, 292], NODE = 20;
    label(A, 0, 14, ['Example ', ['i']], 'pa-t pa-strong');
    var nzA = face(A, 0, 26, 84, 'assets/figures/fig1-frame-40.png', null, true);
    var donor = E('g', { opacity: 0 }, A);
    label(donor, 0, 202, ['Another example ', ['π(i)']], 'pa-t pa-strong');
    var nzB = face(donor, 0, 214, 64, 'assets/samples/convnet-ffhq.jpg', [0, 128, 64, 64], 'flip');
    function net(parent, ys, x0y) {
      var nodes = [];
      for (var c = 0; c < 4; c++) for (var r = 0; r < 4; r++) {
        var y = ys[r] + NODE / 2;
        if (c === 0) E('line', { x1: x0y[0], y1: x0y[1], x2: colX[0], y2: y, 'class': 'pa-wire' }, parent);
        else for (var r2 = 0; r2 < 4; r2++) E('line', { x1: colX[c - 1] + NODE, y1: ys[r2] + NODE / 2, x2: colX[c], y2: y, 'class': 'pa-wire' }, parent);
      }
      for (c = 0; c < 4; c++) for (r = 0; r < 4; r++) {
        var base = E('rect', { x: colX[c], y: ys[r], width: NODE, height: NODE, rx: 5, 'class': 'pa-node' }, parent);
        var hi = E('rect', { x: colX[c], y: ys[r], width: NODE, height: NODE, rx: 5, fill: ACCENT, opacity: 0 }, parent);
        nodes.push({ base: base, hi: hi, c: c, r: r });
      }
      return nodes;
    }
    var netA = net(A, rY, [84, 68]);
    var netB = net(donor, dY, [64, 246]);
    var gc = Math.floor(G0 / PER), gr = G0 % PER;
    function nodeAt(nodes, g) { var c = Math.floor(g / PER), r = g % PER; return nodes[c * 4 + r]; }
    var swapped = E('rect', { x: colX[gc], y: rY[gr], width: NODE, height: NODE, rx: 5, fill: SWAP, opacity: 0 }, A);
    var ringA = E('rect', { x: colX[gc] - 4, y: rY[gr] - 4, width: NODE + 8, height: NODE + 8, rx: 8, 'class': 'pa-hl', opacity: 0 }, A);
    var ringB = E('rect', { x: colX[gc] - 4, y: dY[gr] - 4, width: NODE + 8, height: NODE + 8, rx: 8, 'class': 'pa-hl', opacity: 0 }, donor);
    var arrow = E('line', { x1: colX[gc] + NODE / 2, y1: dY[gr] - 8, x2: colX[gc] + NODE / 2, y2: rY[gr] + NODE + 8, 'class': 'pa-swap', 'marker-end': 'url(#paArrowSwap)', opacity: 0 }, A);
    var swapTxt = E('g', { opacity: 0 }, A);
    label(swapTxt, colX[gc] + 34, 158, ['swap group ', ['g']], 'pa-t pa-strong');
    label(swapTxt, colX[gc] + 34, 176, 'same layer, same noise level', 'pa-t pa-small');
    var tokenN = E('rect', { x: colX[gc], y: dY[gr], width: NODE, height: NODE, rx: 5, fill: SWAP, opacity: 0 }, A);
    /* loss meter */
    var MXL = 312, MYT = 26, MH = 98, MW = 24;
    E('line', { x1: colX[3] + NODE, y1: 75, x2: MXL - 4, y2: 75, 'class': 'pa-axis', 'marker-end': 'url(#paArrow)' }, A);
    E('rect', { x: MXL, y: MYT, width: MW, height: MH, rx: 6, 'class': 'pa-slot' }, A);
    var LB = 0.40, LP = 0.74;
    var fillBase = E('rect', { x: MXL, y: MYT + MH, width: MW, height: 0, rx: 6, 'class': 'pa-lossbase' }, A);
    var fillUp = E('rect', { x: MXL, y: MYT + MH * (1 - LB), width: MW, height: 0, fill: SWAP }, A);
    label(A, MXL + MW / 2, MYT + MH + 18, 'loss', 'pa-t pa-small', 'middle');
    var ell = label(A, MXL + MW + 10, MYT + MH * (1 - LB) + 4, [['ℓ']], 'pa-t', 'start'); op(ell, 0);
    var delta = E('g', { opacity: 0 }, A);
    E('line', { x1: MXL + MW + 6, x2: MXL + MW + 6, y1: MYT + MH * (1 - LP), y2: MYT + MH * (1 - LB), 'class': 'pa-bracket' }, delta);
    label(delta, MXL + MW + 12, MYT + MH * (1 - (LB + LP) / 2) + 5, 'Δ rise', 'pa-t pa-strong');
    var fly = E('rect', { width: 12, height: 9, rx: 2, fill: SWAP, opacity: 0 }, root);
    /* region B: the profile */
    var M = profileMatrix(B);
    var marker = E('path', { d: 'M0,0 l-5,-8 h10 z', fill: ACCENT, opacity: 0 }, B);
    var order = []; for (t = 0; t < T; t++) if (t !== T0) order.push(t);
    var scanOrder = []; for (var gg = 0; gg < G; gg++) if (gg !== G0) scanOrder.push(gg);
    var from = toGlobal('A', MXL + MW + 8, MYT + MH * (1 - (LB + LP) / 2));
    var to = toGlobal('B', MX + T0 * CW + 3, MY + G0 * CH + 2);
    return {
      title: 'Measure sensitivity.',
      text: "Give one group of the teacher the activation of another example at the same noise level. The rise in denoising loss shows how much the teacher relies on that group there.",
      dur: 8200, still: 4000,
      update: function (p, full) {
        var inA = span(p, 0, 500);
        op(A, inA); op(B, span(p, 200, 700));
        /* first pass, then the donor, the swap and the second pass */
        var pass1 = lin(p, 600, 1400), pass2 = lin(p, 2900, 3600);
        netA.forEach(function (n) {
          var a = Math.max(0, 1 - Math.abs(pass1 * 4.4 - n.c - 0.6) * 1.3);
          var b2 = n.c >= gc ? Math.max(0, 1 - Math.abs(pass2 * 3.2 + gc - n.c - 0.4) * 1.3) : 0;
          op(n.hi, 0.55 * Math.max(a, b2));
        });
        netB.forEach(function (n) { op(n.hi, 0.35 * Math.max(0, 1 - Math.abs(lin(p, 1500, 2100) * 4.4 - n.c - 0.6) * 1.3)); });
        var lb = span(p, 1000, 1500) * LB;
        set(fillBase, { y: MYT + MH * (1 - lb), height: MH * lb });
        op(ell, span(p, 1300, 1600) * (1 - span(p, 3300, 3600)));
        op(donor, span(p, 1500, 2000));
        var ring = span(p, 1700, 2100) * (1 - span(p, 7200, 7600));
        op(ringB, ring * (1 - span(p, 4400, 4600)));
        op(arrow, span(p, 2000, 2300) * (1 - span(p, 3700, 4000)));
        op(swapTxt, span(p, 2000, 2300) * (1 - span(p, 4200, 4500)));
        var mv = span(p, 2200, 2900);
        set(tokenN, { y: dY[gr] + (rY[gr] - dY[gr]) * mv });
        op(tokenN, p >= 2200 && p < 2950 ? 1 : 0);
        op(swapped, span(p, 2850, 2950) * (1 - span(p, 4400, 4700)));
        var up = span(p, 3150, 3700) * (LP - LB);
        set(fillUp, { y: MYT + MH * (1 - LB - up), height: MH * up });
        op(fillUp, 1 - span(p, 4400, 4700));
        op(delta, span(p, 3600, 3900) * (1 - span(p, 4400, 4700)));
        /* the loss rise flies into the profile */
        var f = span(p, 3900, 4500);
        set(fly, { x: from[0] + (to[0] - from[0]) * f, y: from[1] + (to[1] - from[1]) * f });
        op(fly, p >= 3900 && p < 4520 ? 1 : 0);
        /* scan the other groups at this noise level, then sweep every other noise level */
        var scan = lin(p, 4550, 5600), sweep = lin(p, 5700, 7300);
        M.cells[G0][T0].setAttribute('opacity', p >= 4500 ? 1 : 0);
        var nScan = Math.floor(scan * scanOrder.length + (scan >= 1 ? 1 : 0));
        scanOrder.forEach(function (gg, i) { M.cells[gg][T0].setAttribute('opacity', i < nScan ? 1 : 0); });
        var cur = scanOrder[Math.min(scanOrder.length - 1, nScan)];
        var inScan = p >= 4550 && p < 5650;
        var cn = nodeAt(netA, inScan ? cur : G0);
        set(ringA, { x: +cn.base.getAttribute('x') - 4, y: +cn.base.getAttribute('y') - 4 });
        op(ringA, ring);
        var nSweep = Math.floor(sweep * order.length + (sweep >= 1 ? 1 : 0));
        order.forEach(function (tt, i) { for (var gg = 0; gg < G; gg++) if (tt !== T0) M.cells[gg][tt].setAttribute('opacity', i < nSweep ? 1 : 0); });
        var curT = p < 5700 ? T0 : order[Math.min(order.length - 1, nSweep)];
        set(marker, { transform: 'translate(' + (MX + curT * CW + CW / 2) + ',' + (MY - 4) + ')' });
        op(marker, span(p, 4400, 4600) * (1 - span(p, 7300, 7600)));
        if (full) { M.cells.forEach(function (row) { row.forEach(function (c) { c.setAttribute('opacity', 1); }); }); op(marker, 0); op(fly, 0); }
        var nz = noiseAt(p < 5700 || p >= 7400 ? T0 : curT);
        if (nzA) op(nzA, nz); if (nzB) op(nzB, nz);
      }
    };
  }

  /* ------------------------------------------------------------------ */
  /* Scene 2: discover phases                                            */
  /* ------------------------------------------------------------------ */
  function scenePhases(root) {
    var A = E('g', null, root), B = region(root, 'B');
    var M = profileMatrix(A);
    M.cells.forEach(function (row) { row.forEach(function (c) { c.setAttribute('opacity', 1); }); });
    var CX = 70, CY = Math.max(44, SUBY + 8), CS = 14;
    label(B, 0, 14, 'Similarity between noise levels', 'pa-t pa-strong');
    label(B, 0, SUBY, 'Pearson correlation of two profiles', 'pa-t pa-small');
    E('rect', { x: CX - 1, y: CY - 1, width: CS * T + 2, height: CS * T + 2, 'class': 'pa-grid' }, B);
    var cells = [];
    for (var i = 0; i < T; i++) {
      cells.push([]);
      for (var j = 0; j < T; j++) cells[i].push(E('rect', { x: CX + j * CS + 0.5, y: CY + i * CS + 0.5, width: CS - 1, height: CS - 1, fill: blues(Math.max(0, S[i][j])), opacity: 0 }, B));
    }
    label(B, CX, CY + CS * T + 30, 'high noise', 'pa-t pa-small');
    label(B, CX + CS * T, CY + CS * T + 30, 'low noise', 'pa-t pa-small', 'end');
    var bandB = PHASES.map(function (b, k) { return E('rect', { x: CX + b[0] * CS + 0.5, y: CY + CS * T + 6, width: (b[1] - b[0]) * CS - 1, height: 8, rx: 2, fill: PH[k], opacity: 0 }, B); });
    var blocks = PHASES.map(function (b, k) {
      return E('rect', { x: CX + b[0] * CS, y: CY + b[0] * CS, width: (b[1] - b[0]) * CS, height: (b[1] - b[0]) * CS, 'class': 'pa-block', stroke: PH[k], opacity: 0 }, B);
    });
    var cuts = [3, 16].map(function (c) {
      var gg = E('g', { opacity: 0 }, B);
      E('line', { x1: CX + c * CS, x2: CX + c * CS, y1: CY - 4, y2: CY + CS * T + 4, 'class': 'pa-cut' }, gg);
      E('line', { y1: CY + c * CS, y2: CY + c * CS, x1: CX - 4, x2: CX + CS * T + 4, 'class': 'pa-cut' }, gg);
      return gg;
    });
    /* two example comparisons: across a boundary (low) and within a phase (high) */
    var PAIRS = [[1, 10], [6, 10]];
    var hl = PAIRS.map(function (pr) {
      var gg = E('g', { opacity: 0 }, root);
      pr.forEach(function (t) {
        var o = toGlobal('A', MX + t * CW, MY);
        E('rect', { x: o[0] - 1.5, y: o[1] - 1.5, width: CW + 3, height: CH * G + 3, 'class': 'pa-colhl' }, gg);
      });
      var a = toGlobal('A', MX + pr[0] * CW + CW / 2, MY + CH * G), b = toGlobal('A', MX + pr[1] * CW + CW / 2, MY + CH * G);
      var c = toGlobal('B', CX + pr[1] * CS + CS / 2, CY + pr[0] * CS + CS / 2);
      var mid = [(a[0] + b[0]) / 2, Math.max(a[1], b[1]) + 18];
      E('path', { d: 'M' + a[0] + ',' + a[1] + ' Q' + a[0] + ',' + mid[1] + ' ' + mid[0] + ',' + mid[1] + ' Q' + b[0] + ',' + mid[1] + ' ' + b[0] + ',' + b[1], 'class': 'pa-link' }, gg);
      E('path', { d: 'M' + mid[0] + ',' + mid[1] + ' C' + mid[0] + ',' + (mid[1] + 40) + ' ' + (c[0] - (L.narrow ? 0 : 60)) + ',' + (c[1] + (L.narrow ? -60 : 0)) + ' ' + c[0] + ',' + c[1], 'class': 'pa-link', 'marker-end': 'url(#paArrow)' }, gg);
      E('rect', { x: c[0] - CS / 2 - 2, y: c[1] - CS / 2 - 2, width: CS + 4, height: CS + 4, rx: 3, 'class': 'pa-hl' }, gg);
      var v = S[pr[0]][pr[1]];
      label(gg, c[0] + CS, c[1] - CS, (v < 0.5 ? 'low' : 'high') + ' similarity', 'pa-t pa-strong pa-halo');
      return gg;
    });
    var dist = []; for (i = 0; i < T; i++) for (j = 0; j < T; j++) dist.push([Math.abs(i - j), i, j]);
    dist.sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; });
    var from = L.B, to = L.A;
    return {
      title: 'Discover phases.',
      text: 'Compare the usage profiles of every pair of noise levels. Dynamic programming then cuts the noise range into contiguous phases whose profiles are alike.',
      dur: 7600,
      update: function (p) {
        var mv = span(p, 0, 800);
        set(A, { transform: 'translate(' + (from[0] + (to[0] - from[0]) * mv) + ',' + (from[1] + (to[1] - from[1]) * mv) + ')' });
        op(B, span(p, 500, 1000));
        op(hl[0], span(p, 1000, 1300) * (1 - span(p, 2100, 2300)));
        op(hl[1], span(p, 2300, 2600) * (1 - span(p, 3400, 3600)));
        var fill = lin(p, 3500, 5200), nFill = Math.floor(fill * dist.length + (fill >= 1 ? 1 : 0));
        dist.forEach(function (d, n) {
          var shown = n < nFill;
          PAIRS.forEach(function (pr, m) {
            var hit = (pr[0] === d[1] && pr[1] === d[2]) || (pr[1] === d[1] && pr[0] === d[2]);
            if (hit && p >= (m === 0 ? 1300 : 2600)) shown = true;
          });
          cells[d[1]][d[2]].setAttribute('opacity', shown ? 1 : 0);
        });
        cuts.forEach(function (c, n) { op(c, span(p, 5300 + n * 250, 5700 + n * 250)); });
        blocks.forEach(function (b, n) { op(b, span(p, 5900 + n * 150, 6300 + n * 150)); });
        var bandOn = span(p, 6000, 6500);
        bandB.forEach(function (b) { op(b, bandOn); });
        M.band.forEach(function (b) { op(b, bandOn); });
        M.bandLabels.forEach(function (b) { op(b, bandOn); });
      }
    };
  }

  /* ------------------------------------------------------------------ */
  /* Scene 3: allocate capacity                                          */
  /* ------------------------------------------------------------------ */
  function sceneAllocate(root) {
    var A = region(root, 'A'), B = region(root, 'B');
    var BX = 44, PITCH = 18, BW = 13, BASE = 250, BH = 180;
    label(A, 0, 14, 'Demand per noise level', 'pa-t pa-strong');
    label(A, 0, SUBY, 'sensitivity weighted by parameter cost', 'pa-t pa-small');
    E('line', { x1: BX - 4, x2: BX + PITCH * T, y1: BASE + 0.5, y2: BASE + 0.5, 'class': 'pa-sep' }, A);
    var bars = Q.map(function (q, t) { return E('rect', { x: BX + t * PITCH, y: BASE, width: BW, height: 0, rx: 3, fill: PH[phaseOf(t)] }, A); });
    var bandA = PHASES.map(function (b, k) { return E('rect', { x: BX + b[0] * PITCH, y: BASE + 8, width: (b[1] - b[0]) * PITCH - 5, height: 8, rx: 2, fill: PH[k], opacity: 0 }, A); });
    var bandL = PHASES.map(function (b, k) { var n = label(A, BX + ((b[0] + b[1]) / 2) * PITCH - 2, BASE + 34, 'Phase ' + (k + 1), 'pa-t pa-phase', 'middle'); n.setAttribute('fill', PH[k]); op(n, 0); return n; });
    E('line', { x1: BX, x2: BX + PITCH * T - 5, y1: BASE + 50, y2: BASE + 50, 'class': 'pa-axis', 'marker-end': 'url(#paArrow)' }, A);
    label(A, BX, BASE + 68, 'high noise', 'pa-t pa-small');
    label(A, BX + PITCH * T - 5, BASE + 68, 'low noise', 'pa-t pa-small', 'end');
    /* region B: the shared budget split into phase budgets */
    label(B, 0, 14, 'Phase budgets', 'pa-t pa-strong');
    label(B, 0, 32, "shares of the teacher's parameter count", 'pa-t pa-small');
    var Y0 = 104, H = 44, W = 440;
    E('rect', { x: 0, y: Y0, width: W, height: H, rx: 10, 'class': 'pa-slot' }, B);
    var x = 0, segs = [];
    SHARE.forEach(function (s, k) {
      var w = s * W, gg = E('g', null, B);
      var clip = 'paseg' + (clipId++);
      var cp = E('clipPath', { id: clip }, gg); E('rect', { x: 0, y: Y0, width: W, height: H, rx: 10 }, cp);
      var r = E('rect', { x: x, y: Y0, width: 0, height: H, fill: PH[k], 'clip-path': 'url(#' + clip + ')' }, gg);
      var pct = label(gg, x + w / 2, Y0 + H / 2 + 5, Math.round(s * 100) + '%', 'pa-t pa-pct', 'middle');
      var name = label(gg, x + w / 2, Y0 - 12, 'Phase ' + (k + 1), 'pa-t pa-phase', 'middle'); name.setAttribute('fill', PH[k]);
      op(pct, 0); op(name, 0);
      if (k > 0) E('line', { x1: x, x2: x, y1: Y0, y2: Y0 + H, 'class': 'pa-segline' }, gg);
      segs.push({ r: r, x: x, w: w, pct: pct, name: name });
      x += w;
    });
    var rule = label(B, W / 2, Y0 + H + 30, 'share of a phase = its demand / the total demand', 'pa-t pa-small', 'middle'); op(rule, 0);
    var EQY = Y0 + H + 96, eq = E('g', { opacity: 0 }, B);
    label(eq, 0, EQY - 12, 'For comparison: equal budgets', 'pa-t pa-strong');
    for (var e = 0; e < 3; e++) {
      E('rect', { x: e * W / 3 + (e ? 1 : 0), y: EQY, width: W / 3 - (e ? 2 : 1), height: 28, rx: e === 0 ? 8 : e === 2 ? 8 : 0, 'class': 'pa-equal' }, eq);
      label(eq, e * W / 3 + W / 6, EQY + 19, '33%', 'pa-t pa-eqpct', 'middle');
    }
    return {
      title: 'Allocate capacity.',
      text: "Each phase receives a share of the teacher's parameter count in proportion to its measured demand, the sum of its bars.",
      dur: 7000,
      update: function (p) {
        op(A, span(p, 0, 400)); op(B, span(p, 300, 700));
        bars.forEach(function (b, t) {
          var f = span(p, 200 + t * 50, 700 + t * 50), h = Q[t] / QMAX * BH * f;
          set(b, { y: BASE - h, height: h });
        });
        var bandOn = span(p, 1400, 1800);
        bandA.forEach(function (b) { op(b, bandOn); }); bandL.forEach(function (b) { op(b, bandOn); });
        segs.forEach(function (s, k) {
          var a = 2000 + k * 700, f = span(p, a, a + 600);
          set(s.r, { width: s.w * f });
          op(s.pct, span(p, a + 400, a + 700)); op(s.name, span(p, a + 300, a + 600));
          /* the bars of the phase being summed stay bright */
          var active = p >= a - 100 && p < a + 700;
          bars.forEach(function (b, t) { if (phaseOf(t) === k && p >= 2000 && p < 4200) op(b, active ? 1 : 0.35); });
        });
        if (p >= 4200 || p < 2000) bars.forEach(function (b) { op(b, 1); });
        op(rule, span(p, 4200, 4600));
        op(eq, span(p, 4700, 5100));
      }
    };
  }

  /* ------------------------------------------------------------------ */
  /* Scene 4: build and route                                            */
  /* ------------------------------------------------------------------ */
  function sceneRoute(root) {
    var A = region(root, 'A'), B = region(root, 'B');
    label(B, 0, 14, 'One student per phase', 'pa-t pa-strong');
    label(B, 0, SUBY, 'bars are layers, sized from the phase budget', 'pa-t pa-small');
    var ST = students(B);
    label(A, 0, 14, 'Sampling', 'pa-t pa-strong');
    label(A, 0, SUBY, 'one call per step, from high to low noise', 'pa-t pa-small');
    var IMG = [['assets/figures/fig1-frame-0.png', 0], ['assets/figures/fig1-frame-10.png', 10], ['assets/figures/fig1-frame-35.png', 35], ['assets/figures/fig1-frame-40.png', 40]];
    var clip = 'paimg' + (clipId++), cp = E('clipPath', { id: clip }, A); E('rect', { x: 0, y: 50, width: 150, height: 150, rx: 10 }, cp);
    var imgs = IMG.map(function (f) { return E('image', { href: f[0], x: 0, y: 50, width: 150, height: 150, 'clip-path': 'url(#' + clip + ')', opacity: 0 }, A); });
    E('rect', { x: 0, y: 50, width: 150, height: 150, rx: 10, 'class': 'pa-frame' }, A);
    label(A, 0, 218, 'denoised estimate', 'pa-t pa-small');
    var TX = 174, TP = 6.6, TW = 5, TY = 112, TH = 24;
    var ticks = [];
    for (var s = 0; s < 40; s++) ticks.push(E('rect', { x: TX + s * TP, y: TY, width: TW, height: TH, rx: 1.5, fill: PH[stepPhase(s)], opacity: 0.25 }, A));
    STEP_PHASES.forEach(function (b, k) {
      var n = label(A, k === 0 ? TX : k === 2 ? TX + 40 * TP - (TP - TW) : TX + ((b[0] + b[1]) / 2) * TP, TY + TH + 18, 'P' + (k + 1), 'pa-t pa-phase', k === 0 ? 'start' : k === 2 ? 'end' : 'middle');
      n.setAttribute('fill', PH[k]);
    });
    E('line', { x1: TX, x2: TX + 40 * TP - (TP - TW), y1: TY + TH + 32, y2: TY + TH + 32, 'class': 'pa-axis', 'marker-end': 'url(#paArrow)' }, A);
    label(A, TX, TY + TH + 50, 'high noise', 'pa-t pa-small');
    label(A, TX + 40 * TP - (TP - TW), TY + TH + 50, 'low noise', 'pa-t pa-small', 'end');
    var stepTxt = label(A, TX, TY - 16, 'step 0 / 40', 'pa-t pa-mono');
    var ptr = E('path', { d: 'M0,0 l-5,-8 h10 z', fill: ACCENT }, A);
    var callTxt = label(A, TX, TY + TH + 84, 'call runs student 1', 'pa-t pa-strong');
    var link = E('path', { 'class': 'pa-route', 'marker-end': 'url(#paArrowAccent)', opacity: 0 }, root);
    return {
      title: 'Build and route.',
      text: "Each phase budget becomes one student, split across layers and distilled from the teacher on its own phase. While sampling, every call runs only the student of the current phase.",
      dur: 9400, still: 5000,
      update: function (p) {
        op(B, span(p, 0, 400));
        ST.forEach(function (st, k) { growStudent(st, span(p, 300 + k * 350, 1100 + k * 350)); op(st.sub, span(p, 900 + k * 350, 1300 + k * 350)); });
        op(A, span(p, 1500, 2000));
        var sp = lin(p, 2200, 8400), step = Math.min(39, Math.floor(sp * 40)), phase = stepPhase(step);
        var run = p >= 2200;
        ticks.forEach(function (tk, s) { op(tk, s < step ? 0.9 : s === step && run ? 1 : 0.25); });
        var px = TX + step * TP + TW / 2;
        set(ptr, { transform: 'translate(' + px + ',' + (TY - 3) + ')' });
        stepTxt.textContent = 'step ' + (run ? Math.min(40, Math.round(sp * 40)) : 0) + ' / 40';
        callTxt.textContent = 'call runs student ' + (phase + 1);
        /* the image: crossfade between the Figure 1 frames */
        var at = sp * 40, w = [0, 0, 0, 0];
        for (var n = 0; n < IMG.length - 1; n++) {
          var a = IMG[n][1], b = IMG[n + 1][1];
          if (at >= a && at <= b) { var f = (at - a) / (b - a); w[n] = 1 - f; w[n + 1] = f; break; }
        }
        if (at >= 40) w = [0, 0, 0, 1];
        imgs.forEach(function (im, n) { op(im, w[n]); });
        ST.forEach(function (st, k) {
          var on = run && k === phase;
          op(st.ring, on ? 1 : 0);
          op(st.g, !run || on ? 1 : 0.42);
          st.tag.textContent = on ? 'running' : 'stored';
          st.tag.setAttribute('class', 'pa-t pa-tag' + (on ? ' is-on' : ''));
        });
        var a0 = toGlobal('A', px, TY + TH + 2), b0 = toGlobal('B', ST[phase].cx, SY - 18);
        var d = L.narrow
          ? 'M' + a0[0] + ',' + a0[1] + ' C' + a0[0] + ',' + (a0[1] + 120) + ' ' + b0[0] + ',' + (b0[1] - 120) + ' ' + b0[0] + ',' + b0[1]
          : 'M' + a0[0] + ',' + a0[1] + ' C' + (a0[0] + 70) + ',' + (a0[1] + 30) + ' ' + b0[0] + ',' + (b0[1] - 50) + ' ' + b0[0] + ',' + b0[1];
        link.setAttribute('d', d);
        op(link, run ? span(p, 2200, 2500) : 0);
      }
    };
  }

  /* ------------------------------------------------------------------ */
  /* Player                                                              */
  /* ------------------------------------------------------------------ */
  var SCENES = [], state = { step: 0, t: 0, playing: false, done: false, userPaused: false }, raf = 0, last = 0;
  function defs() {
    var d = E('defs', null, svg);
    [['paArrow', 'pa-arrowhead'], ['paArrowSwap', 'pa-arrowhead-swap'], ['paArrowAccent', 'pa-arrowhead-accent']].forEach(function (m) {
      var mk = E('marker', { id: m[0], viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, d);
      E('path', { d: 'M0,1 L9,5 L0,9 z', 'class': m[1] }, mk);
    });
  }
  function build() {
    L = layout();
    svg.setAttribute('viewBox', '0 0 ' + L.w + ' ' + L.h);
    root.classList.toggle('is-narrow', L.narrow);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    clipId = 0; defs();
    textScale();
    SCENES = [sceneMeasure, scenePhases, sceneAllocate, sceneRoute].map(function (make) {
      var g = E('g', { 'class': 'pa-scene' }, svg), s = make(g); s.g = g; return s;
    });
    show(state.step);
  }
  /* keep the SVG text near 12.5 px on screen in the stacked layout, whatever the rendered width */
  function textScale() {
    var wpx = svg.getBoundingClientRect().width || stage.clientWidth, sc = wpx / L.w;
    F = Math.max(1, Math.min(1.54, 12.5 / 13 / (sc || 1)));
    SUBY = F > 1.2 ? 40 : 32;
    svg.style.setProperty('--pa-f', F.toFixed(3));
    return F;
  }
  function show(n) {
    SCENES.forEach(function (s, i) { s.g.style.display = i === n ? '' : 'none'; });
    var btns = tabs.querySelectorAll('button[data-step]');
    Array.prototype.forEach.call(btns, function (b) { b.setAttribute('aria-selected', String(+b.getAttribute('data-step') === n)); });
    capTitle.textContent = SCENES[n].title;
    capText.textContent = SCENES[n].text;
    svg.setAttribute('aria-label', SCENES[n].title + ' ' + SCENES[n].text);
  }
  function render() {
    var s = SCENES[state.step];
    s.update(STATIC ? (s.still != null ? s.still : s.dur) : Math.min(state.t, s.dur), STATIC);
    bar.style.width = (STATIC ? 100 : Math.min(100, state.t / s.dur * 100)).toFixed(2) + '%';
  }
  function go(n) { state.step = n; state.t = 0; show(n); render(); }
  function frame(now) {
    raf = 0; if (!state.playing) return;
    var dt = last ? Math.min(64, now - last) : 16; last = now;
    state.t += dt;
    if (state.t >= SCENES[state.step].dur) {
      if (state.step < SCENES.length - 1) go(state.step + 1);
      else { state.t = SCENES[state.step].dur; state.playing = false; state.done = true; render(); button(); return; }
    }
    render(); raf = requestAnimationFrame(frame);
  }
  function play() {
    if (STATIC) return;
    if (state.done) { state.done = false; go(0); }
    state.playing = true; last = 0;
    if (!raf) raf = requestAnimationFrame(frame);
    button();
  }
  function pause() { state.playing = false; button(); }
  function button() {
    if (!playBtn) return;
    var mode = state.playing ? 'pause' : state.done ? 'replay' : 'play';
    playBtn.setAttribute('data-mode', mode);
    playBtn.setAttribute('aria-label', mode === 'pause' ? 'Pause the animation' : mode === 'replay' ? 'Replay the animation' : 'Play the animation');
  }
  tabs.addEventListener('click', function (e) {
    var b = e.target.closest('button[data-step]'); if (!b) return;
    state.done = false; go(+b.getAttribute('data-step'));
    if (!STATIC && !state.userPaused) play();
  });
  if (playBtn) playBtn.addEventListener('click', function () {
    if (state.playing) { state.userPaused = true; pause(); } else { state.userPaused = false; play(); }
  });

  build();
  var qs = parseInt(query.get('paceStep'), 10), qt = parseInt(query.get('paceT'), 10);
  if (qs >= 1 && qs <= SCENES.length) go(qs - 1);
  if (STATIC) { root.classList.add('is-static'); render(); button(); }
  else if (!isNaN(qt)) { state.t = qt; render(); button(); }
  else {
    render(); button();
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting && en.intersectionRatio >= 0.35) { if (!state.userPaused && !state.done) play(); }
          else if (state.playing) pause();
        });
      }, { threshold: [0, 0.35, 0.6] }).observe(root);
    } else play();
  }
  var lastNarrow = L.narrow, rt = 0;
  window.addEventListener('resize', function () {
    clearTimeout(rt);
    rt = setTimeout(function () {
      var f0 = F, s0 = SUBY;
      if (isNarrow() !== lastNarrow) { lastNarrow = !lastNarrow; build(); render(); }
      else if (Math.abs(textScale() - f0) > 0.04 || SUBY !== s0) { build(); render(); }
    }, 150);
  });
})();

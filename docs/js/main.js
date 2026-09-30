/* Different Noise Levels, Different Needs (PACE): project page interactions.
   Vanilla JS, no dependencies. Sections:
   1. Mobile nav drawer (ported from Decart Research nav.js)
   2. Cover: a schematic of sampling with phase students (Figure 1 phases and frames)
   3. Charts built from the paper's tables (Table 1 scatter, Table 2 allocation rows)
   4. Phase explorer (Figures 4, 5 and 6 redrawn from the decoded figure data, Table 3)
   5. Matched-seed samples (Appendix D, Figures 7 to 14)
   6. Small screens: jump bar and the short version of each act
   7. Quick-links rail, share, BibTeX copy, reveal on scroll
   8. Colab links */
(function () {
  'use strict';

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };
  var reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var SVGNS = 'http://www.w3.org/2000/svg';
  /* ?static=1 freezes every transition so headless captures show the final state */
  if (/[?&]static\b/.test(location.search)) document.documentElement.classList.add('is-static');

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function mk(tag, attrs, text) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function segSelect(seg, attr, value) {
    $$('button', seg).forEach(function (b) { b.setAttribute('aria-selected', b.getAttribute(attr) === value ? 'true' : 'false'); });
  }

  /* ------------------------------------------------------------------ */
  /* 1. Mobile navigation drawer                                          */
  /* ------------------------------------------------------------------ */
  (function nav() {
    var toggle = $('.nav-toggle'), links = $('#navLinks');
    if (!toggle || !links) return;
    function setOpen(open) {
      document.body.classList.toggle('nav-open', open);
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
      toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
    }
    toggle.addEventListener('click', function () { setOpen(!document.body.classList.contains('nav-open')); });
    links.addEventListener('click', function (e) { if (e.target.closest('a')) setOpen(false); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') setOpen(false); });
    window.addEventListener('resize', function () { if (window.innerWidth > 900) setOpen(false); });
  })();

  /* ------------------------------------------------------------------ */
  /* 2. Cover: schematic of phase-routed sampling                         */
  /* ------------------------------------------------------------------ */
  /* What is taken from the paper: the three FFHQ phases and the steps at
     which Figure 1 shows the estimates near the two boundaries (steps 10
     and 35 of 40), the four frames of Figure 1, and the EDM noise schedule
     of Appendix A.2.3 (sigma_max 80, sigma_min 0.002, rho 7, 40 steps).
     What is illustrative: the student sizes and their layer shapes. */
  (function cover() {
    var track = $('#simTrack'), studentsEl = $('#simStudents');
    if (!track || !studentsEl) return;
    var stepEl = $('#simStep'), sigmaEl = $('#simSigma'), routeEl = $('#simRouteText'), labelsEl = $('#simPhaseLabels'),
        frame = $('#simFrame'), frameLabel = $('#simFrameLabel'), meterVal = $('#simMeterVal'), meterFill = $('#simMeterFill'),
        play = $('#simPlay'), slider = $('#simSlider'), fill = $('#simFill'), caption = $('#simCaption'), modes = $('#simModes');

    var N = 40;
    var PHASES = [ /* first and last sampling step of each phase, as the repository's router assigns the 40 calls
                      (BlockwiseEDMStudent, FFHQ-64 phases [0, 3), [3, 16), [16, 20)); Figure 1 shows steps 10 and 35 */
      { name: 'Phase 1', short: 'P1', from: 0, to: 10, cls: 'p1' },
      { name: 'Phase 2', short: 'P2', from: 11, to: 35, cls: 'p2' },
      { name: 'Phase 3', short: 'P3', from: 36, to: 39, cls: 'p3' }
    ];
    /* Illustrative only: share of the stored budget per student and relative layer widths. */
    var SHARE = [0.22, 0.53, 0.25];
    var SHAPES = [[0.55, 0.8, 1, 0.8, 0.55], [0.75, 0.95, 1, 0.95, 0.75], [1, 0.7, 0.45, 0.7, 1]];
    var FRAMES = [
      { step: 0, src: 'assets/figures/fig1-frame-0.png', label: 'Initial noise · step 0' },
      { step: 10, src: 'assets/figures/fig1-frame-10.png', label: 'Denoised estimate · step 10' },
      { step: 35, src: 'assets/figures/fig1-frame-35.png', label: 'Denoised estimate · step 35' },
      { step: 40, src: 'assets/figures/fig1-frame-40.png', label: 'Final sample · step 40' }
    ];
    FRAMES.forEach(function (f) { var im = new Image(); im.src = f.src; });

    /* EDM time steps (Karras et al. 2022), as used by the paper's U-Net sampler */
    var SMAX = 80, SMIN = 0.002, RHO = 7;
    function sigmaAt(i) {
      if (i >= N) return 0;
      var a = Math.pow(SMAX, 1 / RHO), b = Math.pow(SMIN, 1 / RHO);
      return Math.pow(a + i / (N - 1) * (b - a), RHO);
    }
    function fmtSigma(s) { return s === 0 ? '0' : (s >= 10 ? s.toFixed(1) : (s >= 1 ? s.toFixed(2) : (s >= 0.01 ? s.toFixed(3) : s.toFixed(4)))); }
    function phaseOf(step) { for (var i = 0; i < PHASES.length; i++) if (step >= PHASES[i].from && step <= PHASES[i].to) return i; return -1; }

    /* build the 40-cell trajectory and the phase labels once */
    var cells = [];
    for (var s = 0; s < N; s++) { var c = el('span', 'pace-cell ' + PHASES[phaseOf(s)].cls); track.appendChild(c); cells.push(c); }
    PHASES.forEach(function (p) {
      var l = el('span', 'pace-phase-label ' + p.cls, p.short); l.title = p.name + ', steps ' + p.from + ' to ' + p.to;
      l.style.left = (p.from / N * 100) + '%'; l.style.width = ((p.to - p.from + 1) / N * 100) + '%';
      labelsEl.appendChild(l);
    });

    /* student cards */
    var cards = [];
    function buildStudents(mode) {
      studentsEl.innerHTML = ''; cards = [];
      var list = mode === 'pace'
        ? PHASES.map(function (p, i) { return { title: 'Student ' + (i + 1), sub: 'Steps ' + p.from + ' to ' + p.to, cls: p.cls, shape: SHAPES[i], scale: 0.45 + 0.55 * SHARE[i] / Math.max.apply(null, SHARE) }; })
        : [{ title: 'Global student', sub: 'Every step', cls: 'pg', shape: [1, 1, 1, 1, 1], scale: 1 }];
      studentsEl.classList.toggle('is-global', mode !== 'pace');
      list.forEach(function (d) {
        var card = el('div', 'pace-student ' + d.cls);
        var head = el('div', 'pace-student-head');
        head.appendChild(el('span', 'pace-student-title', d.title));
        head.appendChild(el('span', 'pace-student-tag', 'Stored'));
        card.appendChild(head);
        var net = el('div', 'pace-net');
        d.shape.forEach(function (w) { var bar = el('i'); bar.style.width = Math.round(w * d.scale * 100) + '%'; net.appendChild(bar); });
        card.appendChild(net);
        card.appendChild(el('span', 'pace-student-sub', d.sub));
        studentsEl.appendChild(card); cards.push(card);
      });
    }

    var CAPTIONS = {
      pace: '<strong>Phase students.</strong> The sampler moves from high to low noise, and the router sends each denoising call to the student of the current phase while the others stay stored. Phases follow Figure 1 of the paper (FFHQ U-Net), and the images are its initial noise, its estimates at steps 10 and 35 and its final sample. The card shows one call per step (the Heun sampler makes 79 calls in 40 steps), and student sizes and layer shapes are illustrative.',
      global: '<strong>One global student.</strong> A single network holding the whole budget runs at every call, whatever the noise level. PACE stores the same total budget as phase students and runs only one of them per call. Sizes and layer shapes are illustrative.'
    };

    var mode = 'pace', step = 0, timer = null, playing = false;
    function render() {
      var done = step >= N, ph = done ? -1 : phaseOf(step);
      cells.forEach(function (c, i) { c.classList.toggle('is-past', i < step); c.classList.toggle('is-now', i === step); });
      var f = FRAMES[0]; FRAMES.forEach(function (x) { if (x.step <= step) f = x; });
      if (frame.getAttribute('src') !== f.src) frame.setAttribute('src', f.src);
      frameLabel.textContent = f.label;
      frame.setAttribute('alt', f.label + ', from the FFHQ trajectory in Figure 1 of the paper');
      sigmaEl.textContent = 'σ = ' + fmtSigma(sigmaAt(step));
      var active = mode === 'pace' ? ph : (done ? -1 : 0);
      cards.forEach(function (c, i) {
        var on = i === active;
        c.classList.toggle('is-active', on);
        c.querySelector('.pace-student-tag').textContent = on ? 'Running' : 'Stored';
      });
      if (done) routeEl.textContent = 'Sampling finished · no further denoising call';
      else routeEl.innerHTML = 'Call at σ = ' + fmtSigma(sigmaAt(step)) + ' <b>→</b> ' + (mode === 'pace' ? PHASES[ph].name.toLowerCase() + ' <b>→</b> student ' + (ph + 1) : 'global student');
      var share = done ? 0 : (mode === 'pace' ? SHARE[ph] : 1);
      meterVal.textContent = done ? 'none' : Math.round(share * 100) + '%';
      meterFill.style.width = (share * 100) + '%';
      meterFill.className = 'pace-meter-fill ' + (done ? '' : (mode === 'pace' ? PHASES[ph].cls : 'pg'));
      stepEl.textContent = step;
      slider.value = step;
      fill.style.width = (step / N * 100) + '%';
    }
    function setStep(s) { step = Math.max(0, Math.min(N, s)); render(); }
    function setMode(m) {
      mode = m; segSelect(modes, 'data-mode', m);
      caption.innerHTML = CAPTIONS[m];
      buildStudents(m); setStep(0);
    }
    var hold = 0;
    function tick() {
      if (step >= N) { hold++; if (hold > 6) { hold = 0; setStep(0); } return; }
      setStep(step + 1);
    }
    function setPlaying(p) {
      playing = p;
      play.classList.toggle('is-playing', p);
      play.setAttribute('aria-label', p ? 'Pause animation' : 'Play animation');
      if (timer) { clearInterval(timer); timer = null; }
      if (p) timer = setInterval(tick, 260);
    }
    play.addEventListener('click', function () { setPlaying(!playing); });
    slider.addEventListener('input', function () { setPlaying(false); setStep(parseInt(slider.value, 10)); });
    modes.addEventListener('click', function (e) {
      var b = e.target.closest('button[data-mode]'); if (!b) return;
      var was = playing; setMode(b.getAttribute('data-mode')); if (was || !reduced) setPlaying(true);
    });
    setMode(/[?&]mode=global\b/.test(location.search) ? 'global' : 'pace');
    if (/[?&]static\b/.test(location.search)) { setStep(18); return; }
    if (!reduced) {
      setPlaying(true);
      if ('IntersectionObserver' in window) {
        var wasAuto = true;
        new IntersectionObserver(function (entries) {
          entries.forEach(function (en) {
            if (!en.isIntersecting && playing) { setPlaying(false); wasAuto = true; }
            else if (en.isIntersecting && !playing && wasAuto) setPlaying(true);
          });
        }, { threshold: 0.15 }).observe($('#heroDemo'));
        play.addEventListener('click', function () { wasAuto = false; });
      }
    }
  })();

  /* ------------------------------------------------------------------ */
  /* 3. Charts                                                            */
  /* ------------------------------------------------------------------ */
  var tip = $('#chartTip');
  function bindTips(root) {
    $$('[data-tip]', root).forEach(function (n) {
      n.addEventListener('pointerenter', function (e) { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); moveTip(e); });
      n.addEventListener('pointermove', moveTip);
      n.addEventListener('pointerleave', function () { tip.classList.remove('is-on'); });
      n.addEventListener('focus', function () { tip.textContent = n.getAttribute('data-tip'); tip.classList.add('is-on'); var r = n.getBoundingClientRect(); tip.style.left = (r.left + r.width / 2) + 'px'; tip.style.top = r.top + 'px'; });
      n.addEventListener('blur', function () { tip.classList.remove('is-on'); });
    });
  }
  function moveTip(e) { tip.style.left = e.clientX + 'px'; tip.style.top = e.clientY + 'px'; }

  /* Horizontal bar renderer (from the reference page).
     groups: [{label, sub, isBreak, bars:[{value, cls, text, tip}]}], max: axis max */
  function renderHBars(container, groups, max, ticks) {
    container.innerHTML = '';
    var frag = document.createDocumentFragment();
    groups.forEach(function (g) {
      var row = el('div', 'hbar-group' + (g.isBreak ? ' is-family-break' : '') + (g.cls ? ' ' + g.cls : ''));
      var lab = el('div', 'hbar-label'); lab.appendChild(document.createTextNode(g.label));
      if (g.sub) lab.appendChild(el('small', null, g.sub));
      row.appendChild(lab);
      var set = el('div', 'hbar-track-set');
      g.bars.forEach(function (b) {
        var r = el('div', 'hbar-row'); r.setAttribute('tabindex', '0');
        if (b.tip) r.setAttribute('data-tip', b.tip);
        r.appendChild(el('span', 'hbar-baseline'));
        var bar = el('div', 'bar' + (b.cls ? ' ' + b.cls : '')); bar.style.width = '0%'; bar.setAttribute('data-w', Math.max(0, Math.min(100, b.value / max * 100)));
        r.appendChild(bar);
        var v = el('span', 'val'); v.innerHTML = b.text; r.appendChild(v);
        set.appendChild(r);
      });
      row.appendChild(set);
      frag.appendChild(row);
    });
    if (ticks) {
      var gr = el('div', 'hbar-group');
      gr.appendChild(el('div'));
      var grid = el('div', 'hbar-grid');
      ticks.forEach(function (t, i) { var s = el('span', i === 0 ? 'is-first' : (i === ticks.length - 1 ? 'is-last' : null), t.label); s.style.left = (t.value / max * 100) + '%'; grid.appendChild(s); });
      gr.appendChild(grid); frag.appendChild(gr);
    }
    container.appendChild(frag);
    bindTips(container);
    requestAnimationFrame(function () { $$('.bar', container).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
  }

  /* The results charts cover ImageNet, FFHQ and LSUN Bedrooms; the phase explorer and the samples viewer keep CIFAR-10. */
  var DATASETS = ['imagenet', 'ffhq', 'lsun'];
  var DATASET_NAMES = { imagenet: 'ImageNet', cifar10: 'CIFAR-10', ffhq: 'FFHQ', lsun: 'LSUN Bedrooms' };
  var FAMILY_NAMES = { dit: 'DiT', convnet: 'U-Net' };
  function pct(a, b) { return (parseFloat(a) / parseFloat(b) - 1) * 100; }
  function fmtPct(v) { var r = Math.round(v * 10) / 10; return (r > 0 ? '+' : (r < 0 ? '−' : '')) + Math.abs(r).toFixed(1) + '%'; }

  /* Table 2 FIDs as printed, per model family and dataset:
     [Uniform blockwise, Phase-aware blockwise, Phase-aware layerwise]. The chart shows the change relative to
     Uniform blockwise (equal budgets per phase), computed here from these values; every shown change is a reduction. */
  var ALLOC_FID = {
    dit: { imagenet: ['5.906', '5.823', '5.770'], ffhq: ['7.512', '7.362', '7.423'], lsun: ['7.172', '7.058', '7.073'] },
    convnet: { imagenet: ['4.239', '4.034', '3.839'], ffhq: ['4.149', '2.546', '2.400'], lsun: ['3.708', '3.411', '2.913'] }
  };

  /* --- Table 2: FID reduction of the phase-aware variants against equal budgets --- */
  (function gains() {
    var host = $('#chartGain'); if (!host) return;
    var seg = $('#gainVariant'), note = $('#gainNote');
    var MAX = 50; /* fixed axis in percent so both variants share one scale */
    var VARIANTS = { blockwise: { i: 1, name: 'Phase-aware blockwise' }, layerwise: { i: 2, name: 'Phase-aware layerwise (PACE)' } };
    function xOf(v) { return Math.max(0, Math.min(v, MAX)) / MAX * 100; }
    function one(v) { return (Math.round(v * 10) / 10).toFixed(1); }
    function draw(key) {
      var V = VARIANTS[key], rows = [];
      /* v is the FID reduction in percent: positive when the variant has lower FID than equal budgets */
      ['dit', 'convnet'].forEach(function (f) { DATASETS.forEach(function (d) { rows.push({ f: f, d: d, v: -pct(ALLOC_FID[f][d][V.i], ALLOC_FID[f][d][0]) }); }); });
      host.innerHTML = '';
      rows.forEach(function (r, idx) {
        var row = el('div', 'gain-row' + (idx === DATASETS.length ? ' is-family-break' : ''));
        var lab = el('div', 'gain-label'); lab.appendChild(document.createTextNode(DATASET_NAMES[r.d])); lab.appendChild(el('small', null, FAMILY_NAMES[r.f]));
        row.appendChild(lab);
        var track = el('div', 'gain-track'); track.setAttribute('tabindex', '0');
        track.setAttribute('data-tip', V.name + ' against equal budgets · ' + FAMILY_NAMES[r.f] + ', ' + DATASET_NAMES[r.d] + ' · FID ' + one(Math.abs(r.v)) + '% ' + (r.v >= 0 ? 'lower' : 'higher'));
        track.appendChild(el('span', 'gain-zero'));
        var w = xOf(r.v), inside = w > 72;
        var bar = el('div', 'gain-bar ' + (r.v >= 0 ? 'is-gain' : 'is-loss'));
        bar.style.width = '0%'; bar.setAttribute('data-w', Math.max(0.6, w));
        track.appendChild(bar);
        var val = el('span', 'gain-val ' + (r.v >= 0 ? 'is-gain' : 'is-loss') + (inside ? ' is-inside' : ''), (r.v >= 0 ? '' : '+') + one(Math.abs(r.v)) + '%');
        if (inside) { val.style.right = (100 - w) + '%'; } else { val.style.left = w + '%'; }
        track.appendChild(val);
        row.appendChild(track);
        host.appendChild(row);
      });
      var axis = el('div', 'gain-row gain-axis'); axis.appendChild(el('div'));
      var ticks = el('div', 'gain-ticks');
      [0, 10, 20, 30, 40].forEach(function (t) { var s = el('span', t === 0 ? 'is-first' : null, t === 0 ? '0' : t + '%'); s.style.left = xOf(t) + '%'; ticks.appendChild(s); });
      axis.appendChild(ticks); host.appendChild(axis);
      bindTips(host);
      requestAnimationFrame(function () { $$('.gain-bar', host).forEach(function (b) { b.style.width = b.getAttribute('data-w') + '%'; }); });
      /* note generated from the same numbers */
      function range(list) {
        var m = list.map(function (r) { return Math.round(Math.abs(r.v) * 10) / 10; }), lo = Math.min.apply(null, m), hi = Math.max.apply(null, m);
        return lo === hi ? lo.toFixed(1) + '%' : lo.toFixed(1) + ' to ' + hi.toFixed(1) + '%';
      }
      var lower = rows.filter(function (r) { return r.v > 0; }), higher = rows.filter(function (r) { return r.v < 0; });
      var parts = ['dit', 'convnet'].map(function (f) {
        var l = lower.filter(function (r) { return r.f === f; });
        return l.length ? 'by ' + range(l) + ' for the ' + FAMILY_NAMES[f] + 's' : null;
      }).filter(Boolean);
      var txt = '<strong>' + V.name + '.</strong> FID falls ' + parts.join(' and ') + ', relative to equal budgets.';
      if (higher.length) txt += ' It rises by ' + higher.map(function (r) { return one(-r.v) + '% for the ' + DATASET_NAMES[r.d] + ' ' + FAMILY_NAMES[r.f]; }).join(' and ') + '.';
      note.innerHTML = txt + ' All variants target the teacher’s parameter count.';
    }
    seg.addEventListener('click', function (e) { var b = e.target.closest('button[data-variant]'); if (!b) return; segSelect(seg, 'data-variant', b.getAttribute('data-variant')); draw(b.getAttribute('data-variant')); });
    draw('layerwise');
  })();

  /* Table 1, DiT rows as printed: [PACE FID, PACE samples/s, TinyFusion FID, TinyFusion samples/s] */
  var VS_TF = { imagenet: ['5.770', '13.39', '5.973', '9.86'], ffhq: ['7.423', '113.73', '7.380', '96.06'], lsun: ['7.073', '32.13', '7.185', '24.31'] };

  /* --- Table 1: PACE against TinyFusion, throughput ratio and FID change per dataset --- */
  (function versusTinyFusion() {
    var host = $('#chartTF'); if (!host) return;
    DATASETS.forEach(function (d) {
      var r = VS_TF[d], ratio = parseFloat(r[1]) / parseFloat(r[3]), dF = pct(r[0], r[2]);
      var cell = el('div', 'rel-cell'); cell.setAttribute('tabindex', '0');
      cell.setAttribute('data-tip', DATASET_NAMES[d] + ' · PACE ' + r[1] + ' against TinyFusion ' + r[3] + ' samples/s · FID ' + r[0] + ' against ' + r[2] + ' (Table 1)');
      cell.appendChild(el('span', 'rel-ds', DATASET_NAMES[d]));
      var big = el('div', 'rel-big'); big.appendChild(document.createTextNode(ratio.toFixed(2) + '×')); cell.appendChild(big);
      cell.appendChild(el('span', 'rel-sub', 'throughput'));
      var r2 = Math.round(dF * 10) / 10;
      cell.appendChild(el('span', 'rel-fid ' + (r2 <= 0 ? 'is-gain' : 'is-loss'), 'FID ' + Math.abs(r2).toFixed(1) + '% ' + (r2 <= 0 ? 'lower' : 'higher')));
      host.appendChild(cell);
    });
    bindTips(host);
  })();

  /* ------------------------------------------------------------------ */
  /* 4. Phase explorer                                                    */
  /* ------------------------------------------------------------------ */
  /* Matrices are decoded from the paper's vector figures (js/data.js, built
     from assets/data/phases.json and validated against Table 3); the statistics
     in the right pane and the notes quote Table 3 and Appendices B and C. */
  (function phases() {
    var tabs = $('#phaseTabs'), heat = $('#phaseHeat'); if (!tabs || !heat) return;
    var DATA = window.PACE_PHASES || null;
    var statsEl = $('#phaseStats'), listEl = $('#phaseList'), note = $('#phaseNote'), fam = $('#phaseFam'), statsTitle = $('#phaseStatsTitle'), negKey = $('#phaseNegKey');
    var TEACHERS = [
      { key: 'cifar10', tab: 'DDPM++ · CIFAR-10', fam: 'U-Net', rule: 'auto K = 2',
        phases: [[0, 8], [8, 20]], within: ['0.912', '0.923'],
        note: 'The two phases cover [0, 8) and [8, 20). Within-phase correlations are 0.912 and 0.923, compared with 0.363 between phases.' },
      { key: 'imagenet', tab: 'ADM · ImageNet', fam: 'U-Net', rule: 'auto K = 2',
        phases: [[0, 16], [16, 20]], within: ['0.904', '0.951'],
        note: 'The split occurs later than on CIFAR-10, at bin 16. The phases [0, 16) and [16, 20) have within-phase correlations of 0.904 and 0.951; their cross-phase mean is 0.429. The final four bins form a compact, distinct sensitivity.' },
      { key: 'ffhq', tab: 'DDPM++ · FFHQ', fam: 'U-Net', rule: 'fixed K = 3',
        phases: [[0, 3], [3, 16], [16, 20]], within: ['0.917', '0.984', '0.982'],
        note: 'Here K = 3 is chosen manually by inspecting the matrix, and the boundaries are optimized: [0, 3), [3, 16) and [16, 20) contain 3, 13 and 4 bins. The middle to late correlation is 0.884, compared with 0.554 for early to late, which suggests some continuity across the phase boundaries.' },
      { key: 'lsun', tab: 'ADM-derived · Bedroom', fam: 'U-Net', rule: 'auto K = 2',
        phases: [[0, 16], [16, 20]], within: ['0.950', '0.940'],
        note: 'The weighted profile selects [0, 16) and [16, 20), with within-phase means of 0.950 and 0.940 and a cross-phase mean of 0.547. It samples 4,387 filters representing a population of 119,555.' },
      { key: 'ditmicro', tab: 'DiT-Micro · CIFAR-10', fam: 'DiT', rule: 'auto K = 4',
        phases: [[0, 4], [4, 8], [8, 16], [16, 20]], within: ['0.956', '0.832', '0.914', '0.928'],
        note: 'An archived profile of an eight-block DiT-Micro over its 24 attention heads selects four phases. DiT-Micro is a separate model, smaller than the DiT-S/2-style teacher of Tables 1 and 2. The early phase correlates negatively with the third and fourth phases (−0.146 and −0.189). The paper notes that this profile establishes neither phase stability across independent runs nor the partition of a different DiT teacher.' },
      { key: 'diffwave', tab: 'DiffWave · audio', fam: 'Audio', rule: 'fixed K = 3',
        phases: [[0, 13], [13, 17], [17, 20]], within: ['0.992', '0.943', '0.912'], demand: ['23.4', '30.4', '46.2'],
        note: 'This profiling-only transfer to audio uses SC09 and trains no audio students. With K = 3 fixed, the phases are t = 199 to 70, 69 to 30 and 29 to 0, with 13, 4 and 3 bins. The early and late phases correlate at 0.268, much lower than within each phase. The late phase receives the largest capacity demand even though it contains only three bins.' }
    ];
    var PH_CLS = ['p1', 'p2', 'p3', 'p4'];
    /* matplotlib "Blues" (ColorBrewer 9-class), sampled linearly */
    var BLUES = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'];
    function hex(h) { return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var BL = BLUES.map(hex);
    function blues(v) {
      v = Math.max(0, Math.min(1, v)) * (BL.length - 1);
      var i = Math.min(BL.length - 2, Math.floor(v)), f = v - i, a = BL[i], b = BL[i + 1];
      return 'rgb(' + Math.round(a[0] + (b[0] - a[0]) * f) + ',' + Math.round(a[1] + (b[1] - a[1]) * f) + ',' + Math.round(a[2] + (b[2] - a[2]) * f) + ')';
    }
    function binLabel(t, i) {
      if (t.key !== 'diffwave') return String(i);
      var hi = 199 - 10 * i; return hi + ' to ' + (hi - 9);
    }
    TEACHERS.forEach(function (t, i) {
      var b = el('button'); b.type = 'button'; b.setAttribute('role', 'tab'); b.setAttribute('data-key', t.key); b.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
      if (t.fam !== 'U-Net') b.className = 'is-dlm';
      b.appendChild(el('i')); b.appendChild(document.createTextNode(t.tab)); tabs.appendChild(b);
    });
    function drawHeat(t) {
      heat.innerHTML = '';
      var panel = DATA && DATA[t.key];
      var M = panel && panel.matrix, n = 20;
      var S = 400, PAD = 34, TOP = 30, cell = (S - PAD) / n;
      var svg = mk('svg', { viewBox: '0 0 ' + (S + 8) + ' ' + (S + TOP + 30), role: 'img', 'aria-label': 'Correlation matrix for ' + t.tab + ' with ' + t.phases.length + ' phases' });
      var ox = PAD, oy = TOP;
      /* phase strips above the matrix */
      t.phases.forEach(function (p, k) {
        svg.appendChild(mk('rect', { x: ox + p[0] * cell, y: 4, width: (p[1] - p[0]) * cell - 1.5, height: 18, rx: 3, 'class': 'strip ' + PH_CLS[k] }));
        svg.appendChild(mk('text', { x: ox + (p[0] + p[1]) / 2 * cell, y: 17, 'class': 'strip-lbl', 'text-anchor': 'middle' }, 'P' + (k + 1)));
      });
      if (!M) {
        svg.appendChild(mk('text', { x: ox + (S - PAD) / 2, y: oy + (S - PAD) / 2, 'class': 'tick', 'text-anchor': 'middle' }, 'Matrix data unavailable'));
      } else {
        for (var r = 0; r < n; r++) {
          for (var c = 0; c < n; c++) {
            var v = M[r][c];
            var rect = mk('rect', { x: ox + c * cell, y: oy + r * cell, width: cell + 0.3, height: cell + 0.3, fill: v == null ? '#d6d6d6' : blues(v), 'class': 'cell' });
            rect.setAttribute('data-tip', 'Bins ' + binLabel(t, r) + ' and ' + binLabel(t, c) + ' · ' + (v == null ? 'correlation below 0' : 'r = ' + v.toFixed(2)));
            svg.appendChild(rect);
          }
        }
      }
      /* phase boundaries on both axes */
      t.phases.slice(1).forEach(function (p) {
        var x = ox + p[0] * cell, y = oy + p[0] * cell;
        svg.appendChild(mk('line', { x1: x, x2: x, y1: oy, y2: oy + n * cell, 'class': 'bound' }));
        svg.appendChild(mk('line', { x1: ox, x2: ox + n * cell, y1: y, y2: y, 'class': 'bound' }));
      });
      /* axis ticks */
      var tickIdx = t.key === 'diffwave' ? [0, 4, 8, 12, 16, 19] : [0, 5, 10, 15, 19];
      tickIdx.forEach(function (i) {
        var lab = t.key === 'diffwave' ? String(199 - 10 * i) : String(i);
        svg.appendChild(mk('text', { x: ox - 6, y: oy + (i + 0.5) * cell + 3.5, 'class': 'tick', 'text-anchor': 'end' }, lab));
        svg.appendChild(mk('text', { x: ox + (i + 0.5) * cell, y: oy + n * cell + 13, 'class': 'tick', 'text-anchor': 'middle' }, lab));
      });
      svg.appendChild(mk('text', { x: ox + n * cell / 2, y: oy + n * cell + 27, 'class': 'axis-title', 'text-anchor': 'middle' }, t.key === 'diffwave' ? 'Timestep bin, high to low noise' : 'Noise bin, high to low noise'));
      heat.appendChild(svg);
      heat.setAttribute('aria-label', 'Correlation matrix for ' + t.tab);
      bindTips(heat);
    }
    function show(key) {
      var t = TEACHERS.filter(function (x) { return x.key === key; })[0];
      segSelect(tabs, 'data-key', key);
      drawHeat(t);
      fam.textContent = t.fam;
      negKey.hidden = t.key !== 'ditmicro';
      /* right pane: within-phase correlation (and capacity demand for audio) */
      var groups = t.phases.map(function (p, k) {
        return { label: 'Phase ' + (k + 1), sub: (p[1] - p[0]) + ' bin' + (p[1] - p[0] > 1 ? 's' : ''), cls: 'ph-' + PH_CLS[k],
          bars: [{ value: parseFloat(t.within[k]), cls: 'is-ph ' + PH_CLS[k], text: t.within[k], tip: t.tab + ' · phase ' + (k + 1) + ' · mean within-phase correlation ' + t.within[k] }] };
      });
      statsTitle.textContent = 'Mean within-phase correlation';
      renderHBars(statsEl, groups, 1, [{ value: 0, label: '0' }, { value: 0.5, label: '0.5' }, { value: 1, label: '1' }]);
      if (t.demand) {
        var extra = el('div', 'pane-title is-second', 'Relative capacity demand, Figure 4(b)');
        statsEl.appendChild(extra);
        var box = el('div', 'hbars'); statsEl.appendChild(box);
        renderHBars(box, t.phases.map(function (p, k) {
          return { label: ['Early', 'Middle', 'Late'][k], sub: 't = ' + (199 - 10 * p[0]) + ' to ' + (199 - 10 * (p[1] - 1) - 9), bars: [{ value: parseFloat(t.demand[k]), cls: 'is-ph ' + PH_CLS[k], text: t.demand[k] + '%', tip: 'DiffWave · ' + ['early', 'middle', 'late'][k] + ' specialist · ' + t.demand[k] + '% of the capacity demand' }] };
        }), 50, [{ value: 0, label: '0' }, { value: 25, label: '25%' }, { value: 50, label: '50%' }]);
      }
      listEl.innerHTML = '';
      var li = el('li'); li.innerHTML = '<span>Rule</span>' + t.rule; listEl.appendChild(li);
      var li2 = el('li'); li2.innerHTML = '<span>Phases</span>' + (t.key === 'diffwave' ? 't = 199 to 70, 69 to 30, 29 to 0' : t.phases.map(function (p) { return '[' + p[0] + ', ' + p[1] + ')'; }).join(', ')); listEl.appendChild(li2);
      note.innerHTML = '<strong>' + t.tab + '.</strong> ' + t.note;
    }
    tabs.addEventListener('click', function (e) { var b = e.target.closest('button[data-key]'); if (b) show(b.getAttribute('data-key')); });
    show('cifar10');
  })();

  /* ------------------------------------------------------------------ */
  /* 5. Matched-seed samples (Appendix D)                                 */
  /* ------------------------------------------------------------------ */
  (function samples() {
    var grid = $('#samplesGrid'), head = $('#samplesHead'); if (!grid) return;
    var famSeg = $('#smFamily'), dsSel = $('#smDataset'), note = $('#samplesNote');
    var MAN = window.PACE_SAMPLES || null;
    var COLS = ['Teacher', 'Global', 'Uniform blockwise', 'Phase-aware blockwise', 'Phase-aware layerwise'];
    COLS.forEach(function (c, i) { head.appendChild(el('span', i === 4 ? 'is-ours' : null, c)); });
    var state = { family: 'dit', dataset: 'imagenet', open: false };
    var more = el('button', 'btn-ghost samples-more'); more.type = 'button';
    grid.parentNode.insertBefore(more, note);
    function entry() {
      if (!MAN) return null;
      return MAN.filter(function (m) { return m.family === state.family && m.dataset === state.dataset; })[0] || null;
    }
    function draw() {
      var m = entry();
      grid.innerHTML = '';
      if (!m) { grid.appendChild(el('p', 'chart-note', 'Samples unavailable.')); more.hidden = true; note.textContent = ''; return; }
      var rows = m.rows, shown = state.open ? rows : Math.min(4, rows);
      var frame = el('div', 'samples-frame' + (m.cell_src <= 32 ? ' is-pixel' : ''));
      frame.style.aspectRatio = m.cols + ' / ' + shown;
      var img = el('img'); img.src = m.file; img.alt = m.dataset_label + ' samples from Figure ' + m.figure + ': ' + rows + ' rows of matched seeds, columns ' + COLS.join(', ') + '.'; img.loading = 'lazy';
      img.width = m.cols * m.cell_px; img.height = rows * m.cell_px;
      frame.appendChild(img);
      frame.appendChild(el('span', 'samples-ours-col'));
      grid.appendChild(frame);
      more.hidden = rows <= 4;
      more.textContent = state.open ? 'Show four rows' : 'Show all ' + (rows === 10 ? 'ten' : rows) + ' rows';
      note.innerHTML = '<strong>Figure ' + m.figure + ' · ' + (state.family === 'dit' ? 'DiT' : 'U-Net') + ' · ' + m.dataset_label + '.</strong> ' + m.note;
    }
    more.addEventListener('click', function () { state.open = !state.open; draw(); });
    famSeg.addEventListener('click', function (e) { var b = e.target.closest('button[data-family]'); if (!b) return; state.family = b.getAttribute('data-family'); segSelect(famSeg, 'data-family', state.family); draw(); });
    dsSel.addEventListener('change', function () { state.dataset = dsSel.value; draw(); });
    draw();
  })();

  /* ------------------------------------------------------------------ */
  /* 6. Small screens: jump bar and the short version of each act         */
  /* ------------------------------------------------------------------ */
  (function smallScreens() {
    var main = $('.article-main'), toc = $$('.toc a');
    if (main && toc.length) {
      var bar = el('nav', 'jumpbar'); bar.setAttribute('aria-label', 'Jump to a section');
      toc.forEach(function (a) {
        var link = el('a', 'jump'); link.setAttribute('href', a.getAttribute('href'));
        var num = a.querySelector('small');
        if (num) link.appendChild(el('small', null, num.textContent));
        link.appendChild(document.createTextNode(a.textContent.replace(num ? num.textContent : '', '').trim()));
        bar.appendChild(link);
      });
      main.insertBefore(bar, main.firstChild);
    }
    var mq = window.matchMedia ? window.matchMedia('(max-width: 700px)') : null;
    if (!mq) return;
    $$('.text-block[id]').forEach(function (act) {
      var deep = $$('.deep', act); if (!deep.length) return;
      var btn = el('button', 'more-btn'); btn.type = 'button';
      var lbl = el('span'), chev = el('i', 'chev'); btn.appendChild(lbl); btn.appendChild(chev);
      act.appendChild(btn);
      function apply() {
        var narrow = mq.matches, open = act.classList.contains('is-open');
        deep.forEach(function (d) { if (narrow && !open) d.setAttribute('hidden', ''); else d.removeAttribute('hidden'); });
        btn.hidden = !narrow;
        lbl.textContent = open ? 'Show the short version' : 'Read the full section';
        btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      }
      btn.addEventListener('click', function () {
        act.classList.toggle('is-open'); apply();
        if (!act.classList.contains('is-open')) act.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
      });
      apply();
      if (mq.addEventListener) mq.addEventListener('change', apply); else if (mq.addListener) mq.addListener(apply);
    });
  })();

  /* ------------------------------------------------------------------ */
  /* 7. Quick-links rail, share, BibTeX copy, reveal                       */
  /* ------------------------------------------------------------------ */
  (function rail() {
    var links = $$('.toc a, .jumpbar a'); if (!links.length) return;
    var map = {}; links.forEach(function (a) { var k = a.getAttribute('href').slice(1); (map[k] = map[k] || []).push(a); });
    if ('IntersectionObserver' in window) {
      var current = null;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) { if (en.isIntersecting) current = en.target.id; });
        links.forEach(function (a) { a.classList.toggle('is-active', !!(map[current] && map[current].indexOf(a) !== -1)); if (a.classList.contains('is-active') && a.closest('.jumpbar') && a.scrollIntoView) { try { a.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' }); } catch (e) {} } });
      }, { rootMargin: '-25% 0px -60% 0px', threshold: 0 });
      Object.keys(map).forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
    }
    var share = $('#shareBtn'), toast = $('#toast');
    function showToast(msg) { toast.textContent = msg; toast.classList.add('is-on'); clearTimeout(showToast.t); showToast.t = setTimeout(function () { toast.classList.remove('is-on'); }, 2200); }
    if (share) share.addEventListener('click', function () {
      var url = location.href.split('#')[0], title = document.title;
      if (navigator.share) { navigator.share({ title: title, url: url }).catch(function () {}); return; }
      if (navigator.clipboard) navigator.clipboard.writeText(url).then(function () { showToast('Link copied'); }, function () { showToast(url); });
      else showToast(url);
    });
    window.__toast = showToast;
  })();

  (function bib() {
    var btn = $('#bibtexCopy'), pre = $('#bibtexText'); if (!btn || !pre) return;
    btn.addEventListener('click', function () {
      var txt = pre.textContent;
      var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy BibTeX'; }, 1800); if (window.__toast) window.__toast('BibTeX copied'); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, function () { selectText(pre); });
      else selectText(pre);
    });
    function selectText(node) { var r = document.createRange(); r.selectNodeContents(node); var s = window.getSelection(); s.removeAllRanges(); s.addRange(r); }
  })();

  (function reveal() {
    var items = $$('.reveal'); if (!items.length) return;
    if (reduced || !('IntersectionObserver' in window) || /[?&]static\b/.test(location.search)) { items.forEach(function (i) { i.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); } });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
    items.forEach(function (i) { io.observe(i); });
    function revealAll() { items.forEach(function (i) { i.classList.add('in'); }); }
    setTimeout(revealAll, 3500);
    window.addEventListener('beforeprint', revealAll);
  })();

  /* ------------------------------------------------------------------ */
  /* 8. Colab links                                                       */
  /* ------------------------------------------------------------------ */
  /* On GitHub Pages the notebook lives in the same repository, so a placeholder
     Colab URL can be derived from the host (USER.github.io) and the first path
     segment (REPO). Anywhere else the link falls back to the notebook file. */
  (function colab() {
    var links = $$('a.js-colab').filter(function (a) { return /USER\/REPO/.test(a.getAttribute('href') || ''); });
    if (!links.length) return; /* configure.py has filled in the real link */
    var host = location.hostname, m = /^([a-z0-9-]+)\.github\.io$/i.exec(host);
    var href = null, noteText = null;
    if (m) {
      var user = m[1], seg = location.pathname.split('/').filter(Boolean)[0];
      var repo = seg && !/\.html?$/i.test(seg) ? seg : user + '.github.io';
      href = 'https://colab.research.google.com/github/' + user + '/' + repo + '/blob/main/colab/pace_playground.ipynb';
    } else {
      href = 'colab/pace_playground.ipynb';
      noteText = 'This preview is not served from GitHub Pages, so the link downloads the notebook; open it in Colab with File, Upload notebook.';
    }
    links.forEach(function (a) {
      if (href) a.setAttribute('href', href);
      if (noteText) { a.setAttribute('title', noteText); a.removeAttribute('target'); }
    });
  })();
})();

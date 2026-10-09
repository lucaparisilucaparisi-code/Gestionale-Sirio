// Visualizzatore della scansione: zoom (rotellina, pulsanti, adatta), trascinamento,
// evidenziazione di righe e celle tramite la griglia rilevata (TableGrid).

import { clamp } from './util.js';

const NS = 'http://www.w3.org/2000/svg';
export const GRID_COLUMNS = ['giorno', 'prog_entrata', 'prog_uscita', 'eff_entrata', 'eff_uscita',
  'ore_dichiarate', 'assenza_alunno', 'assenza_operatore', 'firma', 'note'];

export class ScanViewer {
  /**
   * @param {HTMLElement} canvas area di visualizzazione (overflow nascosto)
   * @param {{onZoom?: Function, onPick?: Function}} opts onPick(giorno, colonna) al clic sulla tabella
   */
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = opts;
    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.natW = 0;
    this.natH = 0;
    this.grid = null;
    this.ready = false;
    this.follow = true;
    this.showMarks = true;
    this.mode = 'page';

    this.stage = document.createElement('div');
    this.stage.className = 'viewer-stage';
    this.img = document.createElement('img');
    this.img.alt = 'Scansione del foglio firma';
    // decodifica sincrona: evita che Chrome mostri la pagina vuota durante lo zoom (checker-imaging)
    this.img.decoding = 'sync';
    this.svg = document.createElementNS(NS, 'svg');
    this.svg.setAttribute('class', 'viewer-overlay');
    this.svg.setAttribute('preserveAspectRatio', 'none');
    this.svg.setAttribute('aria-hidden', 'true');
    this.gMarks = this.mk('g', {});
    this.rRow = this.mk('rect', { class: 'hl-row', visibility: 'hidden' });
    this.rHover = this.mk('rect', { class: 'hl-hover', visibility: 'hidden' });
    this.rCell = this.mk('rect', { class: 'hl-cell', visibility: 'hidden' });
    this.svg.append(this.gMarks, this.rRow, this.rHover, this.rCell);
    this.stage.append(this.img, this.svg);
    canvas.appendChild(this.stage);
    this.stage.style.visibility = 'hidden';

    this.bind();
    this.ro = new ResizeObserver(() => { if (this.ready) this.refit(); });
    this.ro.observe(canvas);
  }

  mk(tag, attrs) {
    const el = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  }

  destroy() {
    this.ro.disconnect();
    window.removeEventListener('pointermove', this._move);
    window.removeEventListener('pointerup', this._up);
  }

  /* ------------------------------------------------------------ caricamento */
  load(url, grid) {
    this.grid = grid || null;
    this.ready = false;
    this.stage.style.visibility = 'hidden';
    return new Promise((resolve, reject) => {
      this.img.onload = async () => {
        try { await this.img.decode(); } catch { /* già decodificata o non necessario */ }
        this.natW = this.img.naturalWidth;
        this.natH = this.img.naturalHeight;
        this.stage.style.width = `${this.natW}px`;
        this.stage.style.height = `${this.natH}px`;
        const gw = this.grid?.width || this.natW;
        const gh = this.grid?.height || this.natH;
        this.svg.setAttribute('viewBox', `0 0 ${gw} ${gh}`);
        this.ready = true;
        this.fit('page', false);
        this.stage.style.visibility = 'visible';
        resolve();
      };
      this.img.onerror = () => reject(new Error('Immagine della scansione non disponibile.'));
      this.img.src = url;
    });
  }

  setGrid(grid) {
    this.grid = grid || null;
    if (this.ready) {
      this.svg.setAttribute('viewBox', `0 0 ${grid?.width || this.natW} ${grid?.height || this.natH}`);
    }
  }

  /* ------------------------------------------------------------ trasformazioni */
  get ratio() { return this.grid?.width ? this.natW / this.grid.width : 1; }

  apply(animate = false) {
    this.stage.classList.toggle('is-animating', !!animate);
    this.stage.style.transform = `translate(${this.tx}px, ${this.ty}px) scale(${this.scale})`;
    this.svg.style.setProperty('--px', String(1 / (this.scale * this.ratio)));
    if (animate) {
      clearTimeout(this._animT);
      this._animT = setTimeout(() => this.stage.classList.remove('is-animating'), 300);
    }
    this.opts.onZoom?.(this.scale);
  }

  bounds() {
    const cw = this.canvas.clientWidth;
    const ch = this.canvas.clientHeight;
    return { cw, ch };
  }

  clampPan() {
    const { cw, ch } = this.bounds();
    const sw = this.natW * this.scale;
    const sh = this.natH * this.scale;
    const m = 24;
    this.tx = sw + 2 * m <= cw ? (cw - sw) / 2 : clamp(this.tx, cw - sw - m, m);
    this.ty = sh + 2 * m <= ch ? (ch - sh) / 2 : clamp(this.ty, ch - sh - m, m);
  }

  fitScale(mode) {
    const { cw, ch } = this.bounds();
    const m = 24;
    if (!this.natW || !cw) return 1;
    if (mode === 'width') {
      // larghezza della tabella (se rilevata) o della pagina
      if (this.grid?.col_x?.length) {
        const r = this.ratio;
        const tw = (this.grid.col_x[this.grid.col_x.length - 1] - this.grid.col_x[0]) * r;
        return (cw - 2 * m) / (tw * 1.04);
      }
      return (cw - 2 * m) / this.natW;
    }
    return Math.min((cw - 2 * m) / this.natW, (ch - 2 * m) / this.natH);
  }

  fit(mode = 'page', animate = true) {
    if (!this.ready) return;
    this.mode = mode;
    this.scale = this.fitScale(mode);
    if (mode === 'width' && this.grid?.col_x?.length) {
      const { cw } = this.bounds();
      const r = this.ratio;
      const x0 = this.grid.col_x[0] * r;
      const tw = (this.grid.col_x[this.grid.col_x.length - 1] - this.grid.col_x[0]) * r;
      this.tx = (cw - tw * this.scale) / 2 - x0 * this.scale;
      this.ty = 24 - (this.grid.header_top || 0) * r * this.scale * 0.92;
    }
    this.clampPan();
    this.apply(animate);
  }

  refit() {
    if (this.mode === 'page' || this.mode === 'width') this.fit(this.mode, false);
    else { this.clampPan(); this.apply(false); }
  }

  zoomAt(factor, cx, cy, animate = false) {
    if (!this.ready) return;
    const min = this.fitScale('page') * 0.6;
    const max = Math.max(3, this.fitScale('page') * 8);
    const ns = clamp(this.scale * factor, min, max);
    const f = ns / this.scale;
    this.tx = cx - (cx - this.tx) * f;
    this.ty = cy - (cy - this.ty) * f;
    this.scale = ns;
    this.mode = 'free';
    this.clampPan();
    this.apply(animate);
  }

  zoomBy(factor) {
    const { cw, ch } = this.bounds();
    this.zoomAt(factor, cw / 2, ch / 2, true);
  }

  /* ------------------------------------------------------------ interazione */
  bind() {
    const c = this.canvas;
    c.addEventListener('wheel', (e) => {
      if (!this.ready) return;
      e.preventDefault();
      const r = c.getBoundingClientRect();
      const k = e.ctrlKey ? 0.012 : 0.0016;
      const delta = e.deltaMode === 1 ? e.deltaY * 32 : e.deltaY;
      this.zoomAt(Math.exp(-delta * k), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });

    let start = null;
    this._move = (e) => {
      if (!start) return;
      const dx = e.clientX - start.x;
      const dy = e.clientY - start.y;
      if (!start.moved && Math.hypot(dx, dy) > 4) { start.moved = true; c.classList.add('is-panning'); }
      if (start.moved) {
        this.tx = start.tx + dx;
        this.ty = start.ty + dy;
        this.mode = 'free';
        this.clampPan();
        this.apply(false);
      }
    };
    this._up = (e) => {
      if (!start) return;
      const s = start;
      start = null;
      c.classList.remove('is-panning');
      window.removeEventListener('pointermove', this._move);
      window.removeEventListener('pointerup', this._up);
      if (!s.moved) this.pick(e);
    };
    c.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || !this.ready || e.target.closest('.viewer-tools')) return;
      start = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty, moved: false };
      window.addEventListener('pointermove', this._move);
      window.addEventListener('pointerup', this._up);
    });
    c.addEventListener('dblclick', (e) => {
      if (!this.ready || e.target.closest('.viewer-tools')) return;
      const r = c.getBoundingClientRect();
      this.zoomAt(2, e.clientX - r.left, e.clientY - r.top, true);
    });
    c.addEventListener('keydown', (e) => {
      if (!this.ready) return;
      const step = 60;
      if (e.key === '+' || e.key === '=') { e.preventDefault(); this.zoomBy(1.25); }
      else if (e.key === '-') { e.preventDefault(); this.zoomBy(0.8); }
      else if (e.key === '0') { e.preventDefault(); this.fit('page'); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); this.tx += step; this.clampPan(); this.apply(true); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); this.tx -= step; this.clampPan(); this.apply(true); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); this.ty += step; this.clampPan(); this.apply(true); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); this.ty -= step; this.clampPan(); this.apply(true); }
    });
  }

  /** Clic sulla scansione: individua giorno e colonna nella tabella. */
  pick(e) {
    if (!this.grid || !this.opts.onPick) return;
    const r = this.canvas.getBoundingClientRect();
    const x = ((e.clientX - r.left - this.tx) / this.scale) / this.ratio;
    const y = ((e.clientY - r.top - this.ty) / this.scale) / this.ratio;
    const { col_x: cx, row_y: ry } = this.grid;
    if (!cx || !ry || y < ry[0] || y > ry[ry.length - 1] || x < cx[0] || x > cx[cx.length - 1]) {
      this.opts.onPick(null, null, { x, y });
      return;
    }
    let g = 1;
    while (g < 31 && y > ry[g]) g++;
    let ci = 0;
    while (ci < cx.length - 2 && x > cx[ci + 1]) ci++;
    this.opts.onPick(g, GRID_COLUMNS[ci], { x, y });
  }

  /* ------------------------------------------------------------ geometria */
  cellBox(giorno, colonna) {
    const g = this.grid;
    if (!g?.row_y || !g?.col_x || giorno < 1 || giorno > 31) return null;
    const ci = GRID_COLUMNS.indexOf(colonna);
    if (ci < 0) return null;
    return [g.col_x[ci], g.row_y[giorno - 1], g.col_x[ci + 1], g.row_y[giorno]];
  }

  spanBox(giorno, c0, c1) {
    const a = this.cellBox(giorno, c0); const b = this.cellBox(giorno, c1);
    return a && b ? [a[0], a[1], b[2], b[3]] : null;
  }

  rowBox(giorno) {
    const g = this.grid;
    if (!g?.row_y || giorno < 1 || giorno > 31) return null;
    return [g.col_x[0], g.row_y[giorno - 1], g.col_x[g.col_x.length - 1], g.row_y[giorno]];
  }

  headerBox() {
    const g = this.grid;
    if (!g) return null;
    const x0 = g.col_x?.[0] ?? 0;
    const x1 = g.col_x?.[g.col_x.length - 1] ?? g.width;
    return [Math.max(0, x0 - 20), Math.round((g.header_top || 0) * 0.12), Math.min(g.width, x1 + 20), g.header_top || Math.round(g.height * 0.2)];
  }

  totalBox() {
    const g = this.grid;
    if (!g?.total_row) return null;
    const ci = GRID_COLUMNS.indexOf('ore_dichiarate');
    return [g.col_x[ci], g.total_row[0], g.col_x[ci + 1], g.total_row[1]];
  }

  coordinatorBox() {
    const g = this.grid;
    if (!g?.total_row) return null;
    return [g.col_x[9], g.total_row[0], g.col_x[10], g.total_row[1]];
  }

  footerBox() {
    const g = this.grid;
    if (!g?.row_y) return null;
    const y0 = g.total_row ? g.total_row[1] : g.row_y[31];
    return [g.col_x[0], y0, g.col_x[g.col_x.length - 1], Math.min(g.height, y0 + (g.row_y[31] - g.row_y[0]) / 31 * 4)];
  }

  setRect(rect, box) {
    if (!box) { rect.setAttribute('visibility', 'hidden'); return; }
    const pad = 2;
    rect.setAttribute('x', box[0] - pad);
    rect.setAttribute('y', box[1] - pad);
    rect.setAttribute('width', Math.max(1, box[2] - box[0] + 2 * pad));
    rect.setAttribute('height', Math.max(1, box[3] - box[1] + 2 * pad));
    rect.setAttribute('visibility', 'visible');
  }

  /** Evidenzia la riga e la cella selezionate; con «segui» porta la cella in vista. */
  highlight({ rowBox = null, cellBox = null, regionClass = null } = {}) {
    this.setRect(this.rRow, rowBox);
    this.rCell.setAttribute('class', regionClass || 'hl-cell');
    this.setRect(this.rCell, cellBox);
    const focus = cellBox || rowBox;
    if (focus && this.follow && this.ready) this.reveal(focus);
  }

  hover(box) { this.setRect(this.rHover, box); }

  setMarks(marks) {
    this.gMarks.innerHTML = '';
    if (!this.showMarks) return;
    for (const m of marks) {
      if (!m.box) continue;
      const r = this.mk('rect', { class: m.kind === 'illeggibile' ? 'mk-illeg' : 'mk-uncert' });
      this.setRect(r, m.box);
      this.gMarks.appendChild(r);
    }
  }

  /** Porta un riquadro (coordinate della griglia) in vista, ingrandendo se troppo piccolo. */
  reveal(box) {
    const { cw, ch } = this.bounds();
    const r = this.ratio;
    const rowH = this.grid?.row_y ? (this.grid.row_y[31] - this.grid.row_y[0]) / 31 : 80;
    let changed = false;
    // ingrandimento di lettura: righe alte almeno ~22 px sullo schermo
    if (rowH * r * this.scale < 22) {
      this.scale = Math.min(this.fitScale('width'), 24 / (rowH * r)) || this.scale;
      this.scale = Math.max(this.scale, 22 / (rowH * r));
      this.mode = 'free';
      changed = true;
    }
    const s = this.scale * r;
    const x0 = box[0] * s + this.tx; const x1 = box[2] * s + this.tx;
    const y0 = box[1] * s + this.ty; const y1 = box[3] * s + this.ty;
    const mx = 40; const my = 70;
    if (changed || x0 < mx || x1 > cw - mx) {
      const tableW = this.grid?.col_x ? (this.grid.col_x[this.grid.col_x.length - 1] - this.grid.col_x[0]) * s : 0;
      if (tableW && tableW < cw - 2 * mx) this.tx = (cw - tableW) / 2 - this.grid.col_x[0] * s;
      else this.tx = cw / 2 - ((box[0] + box[2]) / 2) * s;
      changed = true;
    }
    if (changed || y0 < my || y1 > ch - my - 40) {
      this.ty = ch * 0.42 - ((box[1] + box[3]) / 2) * s;
      changed = true;
    }
    if (changed) {
      this.clampPan();
      this.apply(true);
    }
  }
}

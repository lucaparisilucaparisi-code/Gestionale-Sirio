// Griglia stile foglio di calcolo: selezione, modifica in cella, tastiera, copia/incolla,
// annulla/ripeti e rendering virtualizzato per elenchi lunghi.

import { CHECK_SVG, CROSS_SVG, icon } from './icons.js';
import { toast } from './ui.js';
import { colLetter, esc, fmtOre, parseHours, parseTime } from './util.js';

const EDITABLE_TYPES = new Set(['time', 'hours', 'text', 'bool']);
const NON_SELECTABLE = new Set(['rownum', 'rowhead']);
const VIRTUAL_THRESHOLD = 160;
const OVERSCAN = 12;

let gridSeq = 0;

export class SheetGrid {
  /**
   * @param {HTMLElement} host contenitore (diventa l'area scorrevole della griglia)
   * @param {object} opts vedi le proprietà lette nel costruttore
   */
  constructor(host, opts) {
    this.host = host;
    this.id = `sg${++gridSeq}`;
    this.opts = opts;
    this.rowKey = opts.rowKey || ((row, i) => String(i));
    this.cols = opts.excel ? [{ key: '__n', type: 'rownum', width: 46 }, ...opts.columns] : [...opts.columns];
    this.rows = opts.rows || [];
    this.rowH = opts.rowHeight || 30;
    this.active = null;            // {r, c}
    this.activeKey = null;         // {row, col} chiavi per mantenere la selezione
    this.editing = null;
    this.undoStack = [];
    this.redoStack = [];
    this.range = [0, 0];
    this.hoverKey = null;
    this.readOnly = !!opts.readOnly;
    this.errEl = null;

    host.classList.add('sheet');
    if (opts.zebra) host.classList.add('is-zebra');
    host.tabIndex = 0;
    host.setAttribute('role', 'grid');
    if (opts.label) host.setAttribute('aria-label', opts.label);

    this._onKey = (e) => this.onKeyDown(e);
    this._onDown = (e) => this.onMouseDown(e);
    this._onClick = (e) => this.onClick(e);
    this._onDbl = (e) => this.onDblClick(e);
    this._onScroll = () => this.onScroll();
    this._onFocus = () => host.classList.add('is-focused');
    this._onBlur = (e) => { if (!host.contains(e.relatedTarget)) host.classList.remove('is-focused'); };
    this._onCopy = (e) => this.onCopy(e, false);
    this._onCut = (e) => this.onCopy(e, true);
    this._onPaste = (e) => this.onPaste(e);
    this._onOver = (e) => this.onMouseOver(e);
    this._onLeave = () => this.opts.onHoverRow?.(null);

    host.addEventListener('keydown', this._onKey);
    host.addEventListener('mousedown', this._onDown);
    host.addEventListener('click', this._onClick);
    host.addEventListener('dblclick', this._onDbl);
    host.addEventListener('scroll', this._onScroll, { passive: true });
    host.addEventListener('focusin', this._onFocus);
    host.addEventListener('focusout', this._onBlur);
    host.addEventListener('mouseover', this._onOver);
    host.addEventListener('mouseleave', this._onLeave);
    document.addEventListener('copy', this._onCopy);
    document.addEventListener('cut', this._onCut);
    document.addEventListener('paste', this._onPaste);

    this.render();
  }

  destroy() {
    this.cancelEdit();
    this.hideError();
    const h = this.host;
    h.removeEventListener('keydown', this._onKey);
    h.removeEventListener('mousedown', this._onDown);
    h.removeEventListener('click', this._onClick);
    h.removeEventListener('dblclick', this._onDbl);
    h.removeEventListener('scroll', this._onScroll);
    h.removeEventListener('focusin', this._onFocus);
    h.removeEventListener('focusout', this._onBlur);
    h.removeEventListener('mouseover', this._onOver);
    h.removeEventListener('mouseleave', this._onLeave);
    document.removeEventListener('copy', this._onCopy);
    document.removeEventListener('cut', this._onCut);
    document.removeEventListener('paste', this._onPaste);
  }

  /* ============================================================ struttura */
  isEditable(c, row = null) {
    const col = this.cols[c];
    if (!col || this.readOnly) return false;
    const editable = col.editable ?? EDITABLE_TYPES.has(col.type);
    if (!editable) return false;
    if (row && this.opts.canEdit && !this.opts.canEdit(row, col)) return false;
    return true;
  }

  isSelectable(c) {
    const col = this.cols[c];
    return !!col && !NON_SELECTABLE.has(col.type);
  }

  firstSelectable() {
    for (let c = 0; c < this.cols.length; c++) if (this.isSelectable(c)) return c;
    return 0;
  }

  lastSelectable() {
    for (let c = this.cols.length - 1; c >= 0; c--) if (this.isSelectable(c)) return c;
    return this.cols.length - 1;
  }

  render() {
    const cols = this.cols;
    const hasGroups = cols.some((c) => c.group);
    const colgroup = cols.map((c) => `<col style="width:${c.width ? c.width + 'px' : 'auto'}">`).join('');
    const rows = [];
    let top = 0;
    if (this.opts.excel) {
      const letters = cols.map((c, i) => (i === 0
        ? `<th class="corner" style="top:${top}px" aria-hidden="true"></th>`
        : `<th data-c="${i}" style="top:${top}px" aria-hidden="true">${colLetter(i - 1)}</th>`)).join('');
      rows.push(`<tr class="h-letters">${letters}</tr>`);
      top += 24;
    }
    if (hasGroups) {
      const g = [];
      for (let i = 0; i < cols.length; i++) {
        const c = cols[i];
        if (!c.group) {
          g.push(this.headCell(c, i, top, 2));
          continue;
        }
        let span = 1;
        while (i + span < cols.length && cols[i + span].group === c.group) span++;
        g.push(`<th colspan="${span}" style="top:${top}px">${esc(c.group)}</th>`);
        i += span - 1;
      }
      rows.push(`<tr class="h-group">${g.join('')}</tr>`);
      top += 24;
      const l = cols.map((c, i) => (c.group ? this.headCell(c, i, top, 1) : '')).join('');
      rows.push(`<tr class="h-labels">${l}</tr>`);
      top += 28;
    } else {
      rows.push(`<tr class="h-labels">${cols.map((c, i) => this.headCell(c, i, top, 1)).join('')}</tr>`);
      top += 28;
    }
    this.headH = top;
    const minW = cols.reduce((s, c) => s + (c.width || c.minWidth || 120), 0);
    this.host.innerHTML = `<table style="min-width:${minW}px"><colgroup>${colgroup}</colgroup><thead>${rows.join('')}</thead><tbody></tbody></table>`;
    this.table = this.host.querySelector('table');
    this.tbody = this.table.tBodies[0];
    this.colHeads = [];
    this.letterHeads = [];
    this.host.querySelectorAll('thead th[data-c]').forEach((th) => {
      const c = Number(th.dataset.c);
      if (th.closest('.h-letters')) this.letterHeads[c] = th;
      else this.colHeads[c] = th;
    });
    this.renderBody(true);
  }

  headCell(c, i, top, rowspan) {
    const cls = c.type === 'rownum' ? 'corner' : (c.type === 'rowhead' ? 'corner' : '');
    const label = c.type === 'rownum' ? '1' : esc(c.label || '');
    const tip = c.tip ? ` data-tip="${esc(c.tip)}"` : '';
    return `<th${cls ? ` class="${cls}"` : ''} data-c="${i}"${rowspan > 1 ? ` rowspan="${rowspan}"` : ''} style="top:${top}px"${tip} scope="col">${label}</th>`;
  }

  /* ============================================================ corpo */
  computeRange() {
    const n = this.rows.length;
    if (n <= VIRTUAL_THRESHOLD) return [0, n];
    const viewTop = Math.max(0, this.host.scrollTop - this.headH);
    const visible = Math.ceil(this.host.clientHeight / this.rowH) + 1;
    const start = Math.max(0, Math.floor(viewTop / this.rowH) - OVERSCAN);
    const end = Math.min(n, start + visible + OVERSCAN * 2);
    return [start, end];
  }

  renderBody(force = false) {
    const [start, end] = this.computeRange();
    if (!force && start === this.range[0] && end === this.range[1]) return;
    this.range = [start, end];
    const parts = [];
    const span = this.cols.length;
    if (!this.rows.length) {
      parts.push(`<tr class="empty-row"><td colspan="${span}" style="height:auto;padding:0;cursor:default">${this.opts.emptyHtml || ''}</td></tr>`);
    } else {
      if (start > 0) parts.push(`<tr class="spacer" aria-hidden="true"><td colspan="${span}" style="height:${start * this.rowH}px;padding:0;border:0"></td></tr>`);
      for (let r = start; r < end; r++) parts.push(this.rowHTML(r));
      if (end < this.rows.length) parts.push(`<tr class="spacer" aria-hidden="true"><td colspan="${span}" style="height:${(this.rows.length - end) * this.rowH}px;padding:0;border:0"></td></tr>`);
    }
    this.tbody.innerHTML = parts.join('');
    this.applyActive(false);
    if (this.hoverKey !== null) this.setHoverRow(this.hoverKey);
  }

  rowHTML(r) {
    const row = this.rows[r];
    const info = this.opts.rowInfo ? this.opts.rowInfo(row, r) : null;
    let cls = 'r';
    if (info?.cls) cls += ' ' + info.cls;
    if (r % 2 === 1) cls += ' is-even';
    if (this.active && this.active.r === r) cls += ' is-active-row';
    const cells = [];
    for (let c = 0; c < this.cols.length; c++) cells.push(this.cellHTML(row, r, c, info));
    return `<tr class="${cls}" data-r="${r}" role="row">${cells.join('')}</tr>`;
  }

  cellHTML(row, r, c, info) {
    const col = this.cols[c];
    if (col.type === 'rownum') {
      return `<td class="rowhead is-excel" data-c="${c}">${r + 2}</td>`;
    }
    const st = this.opts.cellState ? this.opts.cellState(row, col) : null;
    let cls = col.align ? `ta-${col.align}` : (col.type === 'bool' || col.type === 'time' ? 'ta-c' : (col.type === 'hours' || col.type === 'num' ? 'ta-r' : ''));
    if (col.type === 'rowhead') cls += ' rowhead';
    if (col.cls) cls += ' ' + col.cls;
    if (!this.isEditable(c, row) && col.type !== 'rowhead') cls += col.type === 'calc' ? ' is-calc' : ' is-ro';
    let content = '';
    const value = this.valueOf(row, col);
    if (col.render) {
      content = col.render(row, { value, state: st?.state, info });
    } else if (col.type === 'bool') {
      const variant = col.variant === 'cross' ? ' is-cross' : (col.variant === 'sign' ? ' is-sign' : '');
      const ro = this.isEditable(c, row) ? '' : ' is-ro';
      content = `<span class="cb${value ? ' is-on' : ''}${variant}${ro}" role="checkbox" aria-checked="${value ? 'true' : 'false'}">${col.variant === 'cross' ? CROSS_SVG : CHECK_SVG}</span>`;
    } else if (col.type === 'time') {
      if (value === '-') { content = '–'; cls += ' is-muted-dash'; }
      else content = esc(value ?? '');
    } else if (col.type === 'hours' || col.type === 'calc') {
      content = esc(fmtOre(value));
    } else {
      content = esc(value ?? '');
    }
    let tip = '';
    if (st && st.state) {
      cls += ` st-${st.state}`;
      if (st.state === 'illeggibile' && (value === null || value === undefined || value === '' || value === false)) {
        content = `<span class="illeg-tag">${icon('alert-triangle')}Illeggibile</span>`;
      }
    }
    if (st && st.tip) {
      tip = ` data-tip="${esc(st.tip)}"${st.tipTitle ? ` data-tip-title="${esc(st.tipTitle)}"` : ''}${st.tone ? ` data-tip-tone="${st.tone}"` : ''}`;
    } else if (col.cellTip) {
      const t = col.cellTip(row, info);
      if (t) tip = ` data-tip="${esc(t)}"`;
    }
    if (this.active && this.active.r === r && this.active.c === c) cls += ' is-active';
    return `<td class="${cls.trim()}" data-c="${c}" role="gridcell"${tip}>${content}</td>`;
  }

  valueOf(row, col) {
    if (this.opts.getValue) return this.opts.getValue(row, col);
    return row[col.key];
  }

  trAt(r) {
    if (r < this.range[0] || r >= this.range[1]) return null;
    return this.tbody.querySelector(`tr[data-r="${r}"]`);
  }

  tdAt(r, c) {
    const tr = this.trAt(r);
    return tr ? tr.querySelector(`td[data-c="${c}"]`) : null;
  }

  /** Aggiorna i dati (mantiene la cella attiva per chiave). */
  setRows(rows) {
    this.rows = rows || [];
    if (this.editing) this.cancelEdit();
    if (this.activeKey) {
      const r = this.rows.findIndex((row, i) => this.rowKey(row, i) === this.activeKey.row);
      const c = this.cols.findIndex((col) => col.key === this.activeKey.col);
      this.active = r >= 0 && c >= 0 ? { r, c } : null;
    }
    this.renderBody(true);
  }

  refresh() { this.renderBody(true); }

  refreshRow(r) {
    const tr = this.trAt(r);
    if (!tr) return;
    const editingHere = this.editing && this.editing.r === r;
    if (editingHere) return;
    const fresh = document.createElement('tbody');
    fresh.innerHTML = this.rowHTML(r);
    tr.replaceWith(fresh.firstElementChild);
  }

  refreshRowByKey(key) {
    const r = this.rows.findIndex((row, i) => this.rowKey(row, i) === key);
    if (r >= 0) this.refreshRow(r);
  }

  /* ============================================================ selezione */
  applyActive(scroll = true) {
    this.host.querySelectorAll('td.is-active').forEach((td) => td.classList.remove('is-active'));
    this.host.querySelectorAll('tr.is-active-row').forEach((tr) => tr.classList.remove('is-active-row'));
    this.host.querySelectorAll('th.is-active-col').forEach((th) => th.classList.remove('is-active-col'));
    if (!this.active) return;
    const { r, c } = this.active;
    const tr = this.trAt(r);
    if (tr) {
      tr.classList.add('is-active-row');
      tr.querySelector(`td[data-c="${c}"]`)?.classList.add('is-active');
    }
    this.colHeads[c]?.classList.add('is-active-col');
    this.letterHeads[c]?.classList.add('is-active-col');
    if (scroll) this.scrollToCell(r, c);
  }

  scrollToCell(r, c) {
    const h = this.host;
    const rowTop = this.headH + r * this.rowH;
    const viewTop = h.scrollTop;
    const viewH = h.clientHeight;
    if (rowTop - this.headH < viewTop) h.scrollTop = rowTop - this.headH;
    else if (rowTop + this.rowH > viewTop + viewH) h.scrollTop = rowTop + this.rowH - viewH + 1;
    const th = this.colHeads[c];
    if (th) {
      const stickyW = this.stickyWidth();
      const left = th.offsetLeft;
      const w = th.offsetWidth;
      if (left - stickyW < h.scrollLeft) h.scrollLeft = Math.max(0, left - stickyW);
      else if (left + w > h.scrollLeft + h.clientWidth) h.scrollLeft = left + w - h.clientWidth;
    }
    if (this.rows.length > VIRTUAL_THRESHOLD) this.renderBody();
  }

  stickyWidth() {
    let w = 0;
    for (let c = 0; c < this.cols.length; c++) {
      if (!NON_SELECTABLE.has(this.cols[c].type)) break;
      w += this.colHeads[c]?.offsetWidth || this.cols[c].width || 0;
    }
    return w;
  }

  select(r, c, { scroll = true, focus = false, silent = false } = {}) {
    if (!this.rows.length) return;
    r = Math.max(0, Math.min(this.rows.length - 1, r));
    c = Math.max(this.firstSelectable(), Math.min(this.lastSelectable(), c));
    const changed = !this.active || this.active.r !== r || this.active.c !== c;
    this.active = { r, c };
    this.activeKey = { row: this.rowKey(this.rows[r], r), col: this.cols[c].key };
    if (scroll && this.rows.length > VIRTUAL_THRESHOLD) this.scrollToCell(r, c);
    this.applyActive(scroll);
    if (focus) this.host.focus({ preventScroll: true });
    if (changed && !silent) this.opts.onActiveChange?.(this.rows[r], this.cols[c], r, c);
  }

  selectByKey(rowKey, colKey, opts) {
    const r = this.rows.findIndex((row, i) => this.rowKey(row, i) === rowKey);
    let c = this.cols.findIndex((col) => col.key === colKey);
    if (r < 0) return false;
    if (c < 0 || !this.isSelectable(c)) c = this.active ? this.active.c : this.firstSelectable();
    this.select(r, c, opts);
    return true;
  }

  get activeRow() { return this.active ? this.rows[this.active.r] : null; }
  get activeCol() { return this.active ? this.cols[this.active.c] : null; }

  move(dr, dc, { wrap = false } = {}) {
    if (!this.rows.length) return;
    if (!this.active) { this.select(0, this.firstSelectable()); return; }
    let { r, c } = this.active;
    if (dc) {
      let nc = c + dc;
      while (nc >= 0 && nc < this.cols.length && !this.isSelectable(nc)) nc += dc;
      if (nc > this.lastSelectable()) {
        if (wrap && r < this.rows.length - 1) { r += 1; nc = this.firstSelectable(); } else nc = this.lastSelectable();
      } else if (nc < this.firstSelectable()) {
        if (wrap && r > 0) { r -= 1; nc = this.lastSelectable(); } else nc = this.firstSelectable();
      }
      c = nc;
    }
    r += dr;
    this.select(r, c);
  }

  setHoverRow(key) {
    this.hoverKey = key;
    this.host.querySelectorAll('tr.is-hover-row').forEach((tr) => tr.classList.remove('is-hover-row'));
    if (key === null || key === undefined) return;
    const r = this.rows.findIndex((row, i) => this.rowKey(row, i) === key);
    if (r >= 0) this.trAt(r)?.classList.add('is-hover-row');
  }

  flash(r, c) {
    const td = this.tdAt(r, c);
    if (!td) return;
    td.classList.remove('is-flash');
    void td.offsetWidth; // riavvia l'animazione
    td.classList.add('is-flash');
    setTimeout(() => td.classList.remove('is-flash'), 900);
  }

  focus() { this.host.focus({ preventScroll: true }); }

  /* ============================================================ eventi mouse */
  cellFromEvent(e) {
    const td = e.target.closest?.('td[data-c]');
    if (!td || !this.host.contains(td)) return null;
    const tr = td.parentElement;
    if (!tr?.dataset.r) return null;
    return { r: Number(tr.dataset.r), c: Number(td.dataset.c), td };
  }

  onMouseDown(e) {
    if (e.button !== 0) return;
    if (e.target.classList?.contains('cell-editor')) return;
    const cell = this.cellFromEvent(e);
    if (!cell) return;
    if (this.editing && !this.commitEdit()) { e.preventDefault(); return; }
    if (!this.isSelectable(cell.c)) {
      e.preventDefault();
      this.select(cell.r, this.active ? this.active.c : this.firstSelectable(), { focus: true });
      return;
    }
    e.preventDefault();
    this.select(cell.r, cell.c, { focus: true });
  }

  onClick(e) {
    const cb = e.target.closest?.('.cb');
    if (!cb) return;
    const cell = this.cellFromEvent(e);
    if (!cell) return;
    this.toggle(cell.r, cell.c);
  }

  onDblClick(e) {
    const cell = this.cellFromEvent(e);
    if (!cell || !this.isSelectable(cell.c)) return;
    if (this.cols[cell.c].type === 'bool') return;
    if (!this.isEditable(cell.c, this.rows[cell.r])) {
      this.opts.onReadOnlyActivate?.(this.rows[cell.r], this.cols[cell.c]);
      return;
    }
    this.startEdit('edit');
  }

  onMouseOver(e) {
    if (!this.opts.onHoverRow) return;
    const tr = e.target.closest?.('tr[data-r]');
    const key = tr ? this.rowKey(this.rows[Number(tr.dataset.r)], Number(tr.dataset.r)) : null;
    if (key !== this._lastHover) {
      this._lastHover = key;
      this.opts.onHoverRow(key === null ? null : this.rows[Number(tr.dataset.r)]);
    }
  }

  onScroll() {
    if (this.rows.length <= VIRTUAL_THRESHOLD || this.editing) return;
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = null; this.renderBody(); });
  }

  /* ============================================================ tastiera */
  onKeyDown(e) {
    if (this.editing) return; // gestito dall'editor
    const ctrl = e.ctrlKey || e.metaKey;
    const k = e.key;
    if (!this.active && ['ArrowDown', 'ArrowUp', 'ArrowLeft', 'ArrowRight', 'Enter', 'Tab'].includes(k)) {
      if (!this.rows.length) return;
      e.preventDefault();
      this.select(0, this.firstSelectable());
      return;
    }
    if (ctrl && !e.altKey) {
      const lk = k.toLowerCase();
      if (lk === 'z' && !e.shiftKey) { e.preventDefault(); this.undo(); return; }
      if ((lk === 'y') || (lk === 'z' && e.shiftKey)) { e.preventDefault(); this.redo(); return; }
      if (k === 'Home') { e.preventDefault(); this.select(0, this.firstSelectable()); return; }
      if (k === 'End') { e.preventDefault(); this.select(this.rows.length - 1, this.lastSelectable()); return; }
      if (k === 'ArrowUp') { e.preventDefault(); this.select(0, this.active.c); return; }
      if (k === 'ArrowDown') { e.preventDefault(); this.select(this.rows.length - 1, this.active.c); return; }
      if (k === 'ArrowLeft') { e.preventDefault(); this.select(this.active.r, this.firstSelectable()); return; }
      if (k === 'ArrowRight') { e.preventDefault(); this.select(this.active.r, this.lastSelectable()); return; }
      return; // copia/incolla e altre scorciatoie
    }
    if (e.altKey) return;
    const col = this.activeCol;
    const row = this.activeRow;
    const c = this.active?.c;
    switch (k) {
      case 'ArrowUp': e.preventDefault(); this.move(-1, 0); return;
      case 'ArrowDown': e.preventDefault(); this.move(1, 0); return;
      case 'ArrowLeft': e.preventDefault(); this.move(0, -1); return;
      case 'ArrowRight': e.preventDefault(); this.move(0, 1); return;
      case 'Tab': e.preventDefault(); this.move(0, e.shiftKey ? -1 : 1, { wrap: true }); return;
      case 'Home': e.preventDefault(); this.select(this.active.r, this.firstSelectable()); return;
      case 'End': e.preventDefault(); this.select(this.active.r, this.lastSelectable()); return;
      case 'PageDown': e.preventDefault(); this.move(Math.max(1, Math.floor(this.host.clientHeight / this.rowH) - 2), 0); return;
      case 'PageUp': e.preventDefault(); this.move(-Math.max(1, Math.floor(this.host.clientHeight / this.rowH) - 2), 0); return;
      case 'Enter': {
        e.preventDefault();
        if (e.shiftKey) { this.move(-1, 0); return; }
        if (col?.type === 'bool') {
          if (this.isEditable(c, row)) this.confirmIfFlagged(row, col);
          this.move(1, 0);
          return;
        }
        if (this.isEditable(c, row)) this.startEdit('edit');
        else { this.opts.onReadOnlyActivate?.(row, col); this.move(1, 0); }
        return;
      }
      case 'F2':
        e.preventDefault();
        if (this.isEditable(c, row) && col.type !== 'bool') this.startEdit('edit');
        return;
      case 'Delete':
      case 'Backspace':
        e.preventDefault();
        if (this.isEditable(c, row)) this.clearCell(this.active.r, c);
        return;
      case ' ':
        if (col?.type === 'bool') {
          e.preventDefault();
          if (this.isEditable(c, row)) this.toggle(this.active.r, c);
          return;
        }
        break;
      case 'Escape':
        this.hideError();
        return;
      default:
        break;
    }
    if (k.length === 1 && this.active && this.isEditable(c, row)) {
      e.preventDefault();
      if (col.type === 'bool') {
        const lk = k.toLowerCase();
        if ('xsv1'.includes(lk)) this.setValue(this.active.r, c, true, 'edit');
        else if ('n0-'.includes(lk)) this.setValue(this.active.r, c, false, 'edit');
        return;
      }
      this.startEdit('enter', k);
    }
  }

  /* ============================================================ modifica */
  editText(row, col) {
    const v = this.valueOf(row, col);
    if (v === null || v === undefined) return '';
    if (col.type === 'hours') return fmtOre(v);
    return String(v);
  }

  startEdit(mode, initial) {
    if (!this.active || this.editing) return;
    const { r, c } = this.active;
    const row = this.rows[r];
    const col = this.cols[c];
    if (!this.isEditable(c, row) || col.type === 'bool') return;
    this.scrollToCell(r, c);
    const td = this.tdAt(r, c);
    if (!td) return;
    const input = document.createElement('input');
    input.className = 'cell-editor';
    input.type = 'text';
    input.spellcheck = false;
    input.autocomplete = 'off';
    input.setAttribute('aria-label', `${col.label || col.key}`);
    if (col.type === 'time' || col.type === 'hours') input.inputMode = 'decimal';
    if (col.type === 'text') input.maxLength = col.maxLength || 500;
    const original = this.editText(row, col);
    input.value = initial !== undefined ? initial : original;
    td.appendChild(input);
    this.editing = { r, c, input, mode, original };
    input.focus({ preventScroll: true });
    const end = input.value.length;
    input.setSelectionRange(end, end);
    input.addEventListener('keydown', (e) => this.onEditorKey(e));
    input.addEventListener('input', () => { input.classList.remove('is-invalid'); this.hideError(); });
    input.addEventListener('blur', () => {
      // un clic fuori conferma la modifica (se valida), come nei fogli di calcolo
      setTimeout(() => { if (this.editing && this.editing.input === input) this.commitEdit({ keepOnError: false }); }, 0);
    });
  }

  onEditorKey(e) {
    const ed = this.editing;
    if (!ed) return;
    const k = e.key;
    const move = (dr, dc, wrap = false) => {
      e.preventDefault();
      if (this.commitEdit()) { this.move(dr, dc, { wrap }); this.focus(); }
    };
    if (k === 'Enter') { move(e.shiftKey ? -1 : 1, 0); return; }
    if (k === 'Tab') { move(0, e.shiftKey ? -1 : 1, true); return; }
    if (k === 'Escape') { e.preventDefault(); this.cancelEdit(); this.focus(); return; }
    if (k === 'ArrowUp') { move(-1, 0); return; }
    if (k === 'ArrowDown') { move(1, 0); return; }
    if (ed.mode === 'enter' && (k === 'ArrowLeft' || k === 'ArrowRight')) { move(0, k === 'ArrowLeft' ? -1 : 1); return; }
    if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 's') {
      e.preventDefault();
      if (this.commitEdit()) this.focus();
      this.opts.onSaveShortcut?.();
    }
    e.stopPropagation();
  }

  parseFor(col, text) {
    if (col.type === 'time') {
      const p = parseTime(text);
      if (!p.ok) return { ok: false, error: p.error };
      return { ok: true, value: p.dash ? '-' : p.value };
    }
    if (col.type === 'hours') {
      const p = parseHours(text, col.max || 24);
      return p.ok ? { ok: true, value: p.value } : { ok: false, error: p.error };
    }
    if (col.type === 'bool') {
      const t = String(text ?? '').trim().toLowerCase();
      if (!t || ['0', 'n', 'no', 'false', 'falso', '-'].includes(t)) return { ok: true, value: false };
      if (['x', '1', 's', 'si', 'sì', 'v', 'vero', 'true', '✓', '✔'].includes(t)) return { ok: true, value: true };
      return { ok: false, error: 'Valore non valido per una casella (usare X oppure lasciare vuoto).' };
    }
    const t = String(text ?? '').replace(/\s+/g, ' ').trim();
    if (t.length > (col.maxLength || 500)) return { ok: false, error: `Testo troppo lungo (massimo ${col.maxLength || 500} caratteri).` };
    return { ok: true, value: t || null };
  }

  /** Conferma la modifica in corso; false se il valore non è valido (l'editor resta aperto). */
  commitEdit({ keepOnError = true } = {}) {
    const ed = this.editing;
    if (!ed) return true;
    const row = this.rows[ed.r];
    const col = this.cols[ed.c];
    const parsed = this.parseFor(col, ed.input.value);
    if (!parsed.ok) {
      if (keepOnError) {
        ed.input.classList.add('is-invalid');
        this.showError(ed.input, parsed.error);
        ed.input.focus();
        return false;
      }
      this.cancelEdit();
      toast({ type: 'warning', title: 'Valore non salvato', message: parsed.error });
      return false;
    }
    this.editing = null;
    ed.input.remove();
    this.hideError();
    this.setValue(ed.r, ed.c, parsed.value, 'edit');
    this.opts.onEditEnd?.();
    return true;
  }

  cancelEdit() {
    const ed = this.editing;
    if (!ed) return;
    this.editing = null;
    ed.input.remove();
    this.hideError();
    this.opts.onEditEnd?.();
  }

  showError(anchor, message) {
    this.hideError();
    const el = document.createElement('div');
    el.className = 'cell-error';
    el.setAttribute('role', 'alert');
    el.textContent = message;
    document.body.appendChild(el);
    const r = anchor.getBoundingClientRect();
    let top = r.bottom + 6;
    if (top + el.offsetHeight > window.innerHeight - 8) top = r.top - el.offsetHeight - 6;
    el.style.left = `${Math.max(8, Math.min(window.innerWidth - el.offsetWidth - 8, r.left))}px`;
    el.style.top = `${top}px`;
    this.errEl = el;
    const td = anchor.closest('td');
    if (td) { td.classList.remove('is-shake'); void td.offsetWidth; td.classList.add('is-shake'); }
  }

  hideError() {
    if (this.errEl) { this.errEl.remove(); this.errEl = null; }
  }

  sameValue(a, b) {
    if ((a === null || a === undefined || a === '') && (b === null || b === undefined || b === '')) return true;
    if (typeof a === 'number' && typeof b === 'number') return Math.abs(a - b) < 1e-9;
    return a === b;
  }

  /** Imposta un valore (con annulla/ripeti) e lo passa al proprietario della griglia. */
  setValue(r, c, value, source) {
    const row = this.rows[r];
    const col = this.cols[c];
    const old = this.valueOf(row, col);
    if (this.sameValue(old, value)) {
      this.confirmIfFlagged(row, col);
      return;
    }
    this.undoStack.push({ items: [{ key: this.rowKey(row, r), col: col.key, old, value }] });
    if (this.undoStack.length > 200) this.undoStack.shift();
    this.redoStack = [];
    this.opts.commit([{ row, col, value, old }], { source });
    this.refreshRowByKey(this.rowKey(row, r));
  }

  /** Invio su una cella incerta/illeggibile senza modificarla: conferma la lettura. */
  confirmIfFlagged(row, col) {
    const st = this.opts.cellState ? this.opts.cellState(row, col) : null;
    if (!st || (st.state !== 'incerto' && st.state !== 'illeggibile')) return false;
    this.opts.commit([{ row, col, value: this.valueOf(row, col), confirm: true }], { source: 'confirm' });
    const r = this.rows.indexOf(row);
    if (r >= 0) { this.refreshRow(r); this.flash(r, this.cols.indexOf(col)); }
    return true;
  }

  toggle(r, c) {
    const row = this.rows[r];
    const col = this.cols[c];
    if (col.type !== 'bool' || !this.isEditable(c, row)) return;
    this.select(r, c, { focus: true });
    this.setValue(r, c, !this.valueOf(row, col), 'toggle');
    const cb = this.tdAt(r, c)?.querySelector('.cb');
    if (cb) cb.classList.add('is-pop');
  }

  clearCell(r, c) {
    const col = this.cols[c];
    const row = this.rows[r];
    const value = col.type === 'bool' ? false : null;
    const st = this.opts.cellState ? this.opts.cellState(row, col) : null;
    if (this.sameValue(this.valueOf(row, col), value) && st?.state === 'illeggibile') {
      // Canc su una cella illeggibile vuota: l'utente conferma che il campo è vuoto
      this.confirmIfFlagged(row, col);
      return;
    }
    this.setValue(r, c, value, 'clear');
  }

  /* ============================================================ annulla/ripeti */
  applyHistory(entry, useOld, source) {
    const changes = [];
    let last = null;
    for (const it of entry.items) {
      const r = this.rows.findIndex((row, i) => this.rowKey(row, i) === it.key);
      const c = this.cols.findIndex((col) => col.key === it.col);
      if (r < 0 || c < 0) continue;
      changes.push({ row: this.rows[r], col: this.cols[c], value: useOld ? it.old : it.value, old: useOld ? it.value : it.old });
      last = { r, c };
    }
    if (!changes.length) return false;
    this.opts.commit(changes, { source });
    for (const ch of changes) this.refreshRowByKey(this.rowKey(ch.row, this.rows.indexOf(ch.row)));
    if (last) { this.select(last.r, last.c); this.flash(last.r, last.c); }
    return true;
  }

  undo() {
    const entry = this.undoStack.pop();
    if (!entry) { toast({ type: 'info', title: 'Niente da annullare', message: 'Non ci sono modifiche recenti in questa griglia.', duration: 2500 }); return; }
    if (this.applyHistory(entry, true, 'undo')) this.redoStack.push(entry);
  }

  redo() {
    const entry = this.redoStack.pop();
    if (!entry) return;
    if (this.applyHistory(entry, false, 'redo')) this.undoStack.push(entry);
  }

  /* ============================================================ appunti */
  ownsFocus() {
    const a = document.activeElement;
    return !!a && (a === this.host) && !this.editing;
  }

  copyText(row, col) {
    const v = this.valueOf(row, col);
    if (col.copy) return col.copy(row, v);
    if (v === null || v === undefined) return '';
    if (col.type === 'bool') return v ? 'X' : '';
    if (col.type === 'hours' || col.type === 'calc') return fmtOre(v);
    return String(v);
  }

  onCopy(e, cut) {
    if (!this.ownsFocus() || !this.active) return;
    const row = this.activeRow; const col = this.activeCol;
    e.clipboardData.setData('text/plain', this.copyText(row, col));
    e.preventDefault();
    if (cut && this.isEditable(this.active.c, row)) this.clearCell(this.active.r, this.active.c);
  }

  onPaste(e) {
    if (!this.ownsFocus() || !this.active || this.readOnly) return;
    const text = e.clipboardData?.getData('text/plain');
    if (text === undefined || text === null) return;
    e.preventDefault();
    const lines = text.replace(/\r\n?/g, '\n').split('\n');
    if (lines.length > 1 && lines[lines.length - 1] === '') lines.pop();
    const grid = lines.map((l) => l.split('\t'));
    const changes = [];
    const history = [];
    let skipped = 0;
    let firstError = '';
    let maxR = this.active.r; let maxC = this.active.c;
    for (let i = 0; i < grid.length; i++) {
      const r = this.active.r + i;
      if (r >= this.rows.length) break;
      let c = this.active.c;
      for (let j = 0; j < grid[i].length; j++, c++) {
        while (c < this.cols.length && !this.isSelectable(c)) c++;
        if (c >= this.cols.length) break;
        const row = this.rows[r];
        const col = this.cols[c];
        if (!this.isEditable(c, row)) { skipped++; continue; }
        const p = this.parseFor(col, grid[i][j]);
        if (!p.ok) { skipped++; firstError = firstError || p.error; continue; }
        const old = this.valueOf(row, col);
        if (this.sameValue(old, p.value)) continue;
        changes.push({ row, col, value: p.value, old });
        history.push({ key: this.rowKey(row, r), col: col.key, old, value: p.value });
        maxR = Math.max(maxR, r); maxC = Math.max(maxC, c);
      }
    }
    if (changes.length) {
      this.undoStack.push({ items: history });
      this.redoStack = [];
      this.opts.commit(changes, { source: 'paste' });
      for (const ch of changes) this.refreshRowByKey(this.rowKey(ch.row, this.rows.indexOf(ch.row)));
    }
    if (skipped) {
      toast({
        type: 'warning',
        title: `${skipped === 1 ? 'Un valore non incollato' : `${skipped} valori non incollati`}`,
        message: firstError || 'Alcune celle di destinazione non sono modificabili.',
      });
    } else if (changes.length > 1) {
      toast({ type: 'success', title: `Incollati ${changes.length} valori`, duration: 2500 });
    }
  }
}

// Revisione documento: scansione affiancata alla griglia stile Excel dei 31 giorni,
// intestazione modificabile, anomalie, totali, salvataggio automatico.

import { authUrl, del, get, post, put } from '../api.js';
import { SheetGrid } from '../grid.js';
import { ART, icon } from '../icons.js';
import { docById, loadRegistry, pokePolling, refreshState, store } from '../state.js';
import { confirmDialog, dialog, emptyState, showMenu, statusBadge, toast, toastError } from '../ui.js';
import {
  $, MESI, capitalize, clamp, debounce, esc, fmtOre, fmtPeriodo, fmtSigned, fmtTime, hoursBetween,
  html, isTypingTarget, parseHours, plural, raw, setHTML, storageGet, storageSet,
} from '../util.js';
import { ScanViewer } from '../viewer.js';
import { reviewOrder } from './documents.js';

const DAY_COLS = ['prog_entrata', 'prog_uscita', 'eff_entrata', 'eff_uscita', 'ore_dichiarate',
  'assenza_alunno', 'assenza_operatore', 'firma', 'note'];
const LABELS = {
  prog_entrata: 'Entrata programmata', prog_uscita: 'Uscita programmata', eff_entrata: 'Entrata effettiva',
  eff_uscita: 'Uscita effettiva', ore_dichiarate: 'Ore dichiarate', assenza_alunno: 'Assenza alunno',
  assenza_operatore: 'Assenza operatore', firma: 'Firma operatore', note: 'Note', trattino_effettivo: 'Trattino negli orari effettivi',
  anno_scolastico: 'Anno scolastico', lotto: 'Lotto', municipalita: 'Municipalità', ente: 'Ente',
  istituto: 'Istituto scolastico', operatore: 'Operatore', alunno: 'Alunno', mese: 'Mese', anno: 'Anno',
  ore_pei: 'Ore PEI', sostituzione: 'Sostituzione', data_compilazione: 'Data di compilazione',
  firma_coordinatore: 'Firma coordinatore', timbro_referente: 'Timbro referente', totale_mensile_dichiarato: 'Totale mensile',
};
// campi dell'intestazione nell'ordine del modulo (per F8 e per il modulo)
const HEADER_ORDER = ['operatore', 'alunno', 'istituto', 'ente', 'mese', 'anno', 'ore_pei', 'sostituzione',
  'totale_mensile_dichiarato', 'firma_coordinatore', 'timbro_referente', 'lotto', 'municipalita', 'anno_scolastico', 'data_compilazione'];
const HEADER_MORE = new Set(['lotto', 'municipalita', 'anno_scolastico', 'data_compilazione']);
const SEV = [
  { id: 'errore', label: 'Errori', one: 'errore', icon: 'x-circle', chip: 'is-err' },
  { id: 'attenzione', label: 'Attenzioni', one: 'attenzione', icon: 'alert-triangle', chip: 'is-warn' },
  { id: 'info', label: 'Informazioni', one: 'info', icon: 'info', chip: 'is-info' },
];

function asOriginal(v) {
  if (v === null || v === undefined) return null;
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  return String(v);
}
const emptyVal = (v) => v === null || v === undefined || v === '';
const fmtOrig = (v) => {
  if (v === null || v === undefined) return 'vuoto';
  if (v === 'true') return 'sì (presente)';
  if (v === 'false') return 'no (assente)';
  if (/^\d+(\.\d+)?$/.test(v)) return fmtOre(Number(v));
  return `«${v}»`;
};

export function mount(root, ctx) {
  const view = new Review(root, ctx);
  view.open(ctx.params[0], ctx.query);
  return {
    unmount: () => view.destroy(),
    onState: (kind) => view.onState(kind),
    beforeLeave: () => view.flush(),
    update: (params, query) => {
      if (view.broken) return false;
      if (params[0] === view.id) { view.applyQuery(query); return true; }
      view.open(params[0], query);
      return true;
    },
  };
}

class Review {
  constructor(root, ctx) {
    this.root = root;
    this.ctx = ctx;
    this.id = null;
    this.doc = null;
    this.giorni = new Map();
    this.pending = { rows: new Map(), header: {}, extra: {} };
    this.saving = null;
    this.saveState = 'idle';
    this.lastSaved = null;
    this.needsGridRefresh = false;
    this.sevFilter = new Set(storageGet('sirio.rv.sev', ['errore', 'attenzione', 'info']));
    this.hdrCollapsed = storageGet('sirio.rv.hdrCollapsed', false);
    this.hdrMore = false;
    this.anomsCollapsed = storageGet('sirio.rv.anomsCollapsed', false);
    this.focusArea = 'grid';
    this.scheduleSave = debounce(() => this.flush(), 450);
    this.build();
    loadRegistry().then(() => { if (this.doc) this.renderHeader(); });
    this._key = (e) => this.onKey(e);
    document.addEventListener('keydown', this._key);
  }

  destroy() {
    document.removeEventListener('keydown', this._key);
    this.scheduleSave.cancel();
    this.grid?.destroy();
    this.viewer?.destroy();
  }

  /* ============================================================ struttura */
  build() {
    const split = storageGet('sirio.rv.split', 37);
    this.root.innerHTML = `
      <div class="review">
        <div class="review-bar">
          <a class="btn btn-sm btn-ghost btn-icon" href="#/documenti" aria-label="Torna ai documenti" data-tip="Torna ai documenti">${icon('arrow-left')}</a>
          <div class="review-title"><h2 id="rv-title"><span class="skeleton" style="display:inline-block;width:220px;height:16px"></span></h2><div class="sub" id="rv-sub"></div></div>
          <span id="rv-badge"></span>
          <span class="spacer"></span>
          <span class="save-ind" id="rv-save" aria-live="polite"></span>
          <button type="button" class="btn btn-sm" id="rv-next-flag" data-tip="Vai al prossimo campo illeggibile o incerto (F8)">${icon('flag')}<span class="btn-label-opt">Da verificare</span><span class="count num" id="rv-flag-count">0</span></button>
          <div class="review-nav" role="group" aria-label="Navigazione tra i documenti">
            <button type="button" class="btn btn-sm btn-icon" id="rv-prev" aria-label="Documento precedente" data-tip="Documento precedente (Alt+←)">${icon('chevron-left')}</button>
            <span class="review-nav-pos" id="rv-pos"></span>
            <button type="button" class="btn btn-sm btn-icon" id="rv-next" aria-label="Documento successivo" data-tip="Documento successivo (Alt+→)">${icon('chevron-right')}</button>
          </div>
          <button type="button" class="btn btn-sm btn-icon" id="rv-more" aria-label="Altre azioni" data-tip="Altre azioni">${icon('more')}</button>
          <button type="button" class="btn btn-sm" id="rv-reprocess">${icon('refresh')}<span class="btn-label-opt">Rielabora</span></button>
          <button type="button" class="btn btn-sm btn-success" id="rv-confirm">${icon('check')}Conferma documento</button>
        </div>
        <div class="review-body" id="rv-body" style="--split:${clamp(split, 24, 62)}%">
          <section class="pane" aria-label="Scansione e anomalie">
            <div class="card viewer">
              <div class="viewer-canvas" id="rv-canvas" tabindex="0" aria-label="Scansione del foglio firma: rotellina per ingrandire, trascina per spostare"></div>
              <div class="viewer-tag" id="rv-tag"></div>
              <div class="viewer-loading" id="rv-loading"><span class="spinner" style="color:var(--accent);width:22px;height:22px"></span></div>
              <div class="viewer-tools" role="toolbar" aria-label="Strumenti di visualizzazione">
                <button type="button" class="btn btn-sm btn-icon" data-vt="out" aria-label="Riduci" data-tip="Riduci (−)">${icon('zoom-out')}</button>
                <button type="button" class="viewer-zoom" data-vt="page" id="rv-zoom" data-tip="Adatta alla finestra (0)">100%</button>
                <button type="button" class="btn btn-sm btn-icon" data-vt="in" aria-label="Ingrandisci" data-tip="Ingrandisci (+)">${icon('zoom-in')}</button>
                <span class="sep"></span>
                <button type="button" class="btn btn-sm btn-icon" data-vt="page" aria-label="Pagina intera" data-tip="Pagina intera">${icon('fit')}</button>
                <button type="button" class="btn btn-sm btn-icon" data-vt="width" aria-label="Larghezza tabella" data-tip="Adatta alla larghezza della tabella">${icon('fit-width')}</button>
                <span class="sep"></span>
                <button type="button" class="btn btn-sm btn-icon" data-vt="follow" aria-pressed="true" aria-label="Segui la selezione" data-tip="Segui la cella selezionata">${icon('crosshair')}</button>
                <button type="button" class="btn btn-sm btn-icon" data-vt="marks" aria-pressed="true" aria-label="Mostra le segnalazioni" data-tip="Evidenzia sulla scansione i campi illeggibili e incerti">${icon('layers')}</button>
              </div>
            </div>
            <div class="card anoms" id="rv-anoms"></div>
          </section>
          <div class="splitter" id="rv-split" role="separator" aria-orientation="vertical" aria-label="Ridimensiona i pannelli" tabindex="0"></div>
          <section class="pane" aria-label="Dati letti" style="position:relative">
            <div class="card hdr-card" id="rv-hdr"></div>
            <div class="card grid-card">
              <div class="grid-card-head">
                <div class="card-title">Tabella giornaliera</div>
                <span class="muted" style="font-size:12px" id="rv-grid-hint">Invio per modificare · Canc per svuotare · Ctrl+Z per annullare</span>
                <div class="legend" aria-label="Legenda">
                  <span class="legend-item"><span class="legend-sw sw-illeg"></span>Illeggibile</span>
                  <span class="legend-item"><span class="legend-sw sw-uncert"></span>Incerto</span>
                  <span class="legend-item"><span class="legend-sw sw-edited"></span>Modificato</span>
                  <span class="legend-item"><span class="legend-sw sw-weekend"></span>Festivo</span>
                </div>
              </div>
              <div id="rv-grid"></div>
            </div>
            <div class="card totals" id="rv-totals"></div>
            <div id="rv-overlay"></div>
          </section>
        </div>
      </div>`;

    const r = this.root;
    $('#rv-prev', r).addEventListener('click', () => this.go(-1));
    $('#rv-next', r).addEventListener('click', () => this.go(1));
    $('#rv-next-flag', r).addEventListener('click', () => this.nextFlag(1));
    $('#rv-confirm', r).addEventListener('click', () => this.toggleVerified());
    $('#rv-reprocess', r).addEventListener('click', () => this.reprocess());
    $('#rv-more', r).addEventListener('click', (e) => this.moreMenu(e.currentTarget));
    $('#rv-save', r).addEventListener('click', (e) => { if (e.target.closest('[data-retry]')) this.flush(); });
    this.initSplitter();

    this.viewer = new ScanViewer($('#rv-canvas', r), {
      onZoom: (s) => { const z = $('#rv-zoom', r); if (z) z.textContent = `${Math.round(s * 100)}%`; },
      onPick: (g, col) => this.onScanPick(g, col),
    });
    r.querySelector('.viewer-tools').addEventListener('click', (e) => {
      const b = e.target.closest('[data-vt]');
      if (!b) return;
      const act = b.dataset.vt;
      if (act === 'in') this.viewer.zoomBy(1.25);
      else if (act === 'out') this.viewer.zoomBy(0.8);
      else if (act === 'page' || act === 'width') this.viewer.fit(act);
      else if (act === 'follow') {
        this.viewer.follow = !this.viewer.follow;
        b.setAttribute('aria-pressed', String(this.viewer.follow));
        if (this.viewer.follow) this.syncScan();
      } else if (act === 'marks') {
        this.viewer.showMarks = !this.viewer.showMarks;
        b.setAttribute('aria-pressed', String(this.viewer.showMarks));
        this.updateMarks();
      }
    });

    this.buildGrid();
    this.hdrEl = $('#rv-hdr', r);
    this.hdrEl.addEventListener('change', (e) => this.onHeaderChange(e));
    this.hdrEl.addEventListener('click', (e) => this.onHeaderClick(e));
    this.hdrEl.addEventListener('focusin', (e) => this.onHeaderFocus(e));
    this.hdrEl.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && e.target.matches('input.input')) { e.preventDefault(); e.target.blur(); this.grid.focus(); }
      if (e.key === 'Escape' && e.target.matches('input.input')) { e.target.value = e.target.dataset.orig ?? ''; e.target.blur(); }
    });
    this.anomsEl = $('#rv-anoms', r);
    this.anomsEl.addEventListener('click', (e) => this.onAnomClick(e));
    this.anomsEl.addEventListener('mouseover', (e) => this.onAnomHover(e));
    this.anomsEl.addEventListener('mouseleave', () => { this.grid.setHoverRow(null); this.viewer.hover(null); });
  }

  initSplitter() {
    const sp = $('#rv-split', this.root);
    const body = $('#rv-body', this.root);
    let dragging = false;
    const set = (pct) => {
      const v = clamp(pct, 24, 62);
      body.style.setProperty('--split', `${v}%`);
      storageSet('sirio.rv.split', Math.round(v * 10) / 10);
    };
    sp.addEventListener('pointerdown', (e) => {
      dragging = true;
      sp.classList.add('is-dragging');
      sp.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    sp.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const r = body.getBoundingClientRect();
      set(((e.clientX - r.left) / r.width) * 100);
    });
    const end = () => { dragging = false; sp.classList.remove('is-dragging'); };
    sp.addEventListener('pointerup', end);
    sp.addEventListener('pointercancel', end);
    sp.addEventListener('keydown', (e) => {
      const cur = parseFloat(body.style.getPropertyValue('--split')) || 37;
      if (e.key === 'ArrowLeft') { e.preventDefault(); set(cur - 2); }
      if (e.key === 'ArrowRight') { e.preventDefault(); set(cur + 2); }
    });
  }

  buildGrid() {
    const host = $('#rv-grid', this.root);
    const columns = [
      { key: 'giorno', label: 'Giorno', type: 'rowhead', width: 56, render: (row) => String(row.giorno), cellTip: (row, info) => info?.tip || '' },
      { key: 'gg', label: 'Gg', type: 'ro', width: 44, align: 'c', cls: 'is-tight', render: (row) => this.renderWeekday(row), tip: 'Giorno della settimana' },
      { key: 'prog_entrata', group: 'Programmato', label: 'Entrata', type: 'time', width: 60 },
      { key: 'prog_uscita', group: 'Programmato', label: 'Uscita', type: 'time', width: 60 },
      { key: 'eff_entrata', group: 'Effettivo', label: 'Entrata', type: 'time', width: 60 },
      { key: 'eff_uscita', group: 'Effettivo', label: 'Uscita', type: 'time', width: 60 },
      { key: 'ore_dichiarate', group: 'Ore', label: 'Dichiar.', type: 'hours', width: 66, tip: 'Ore dichiarate: colonna «Tot. ore effettive» scritta sul foglio' },
      { key: 'ore_calcolate', group: 'Ore', label: 'Calcol.', type: 'calc', width: 64, tip: 'Ore calcolate dagli orari effettivi (uscita − entrata)', render: (row) => this.renderCalc(row) },
      { key: 'assenza_alunno', group: 'Assenza', label: 'Alunno', type: 'bool', variant: 'cross', width: 56 },
      { key: 'assenza_operatore', group: 'Assenza', label: 'Operat.', type: 'bool', variant: 'cross', width: 58, tip: 'Assenza dell\'operatore' },
      { key: 'firma', label: 'Firma', type: 'bool', variant: 'sign', width: 52, tip: 'Firma dell\'operatore' },
      { key: 'note', label: 'Note', type: 'text', minWidth: 96, maxLength: 500 },
      { key: 'esito', label: 'Esito', type: 'ro', width: 112, render: (row) => this.renderEsito(row) },
    ];
    this.grid = new SheetGrid(host, {
      columns,
      rows: [],
      label: 'Tabella giornaliera dei 31 giorni',
      rowKey: (row) => String(row.giorno),
      getValue: (row, col) => this.cellValue(row, col.key),
      cellState: (row, col) => this.cellState(row, col.key),
      rowInfo: (row) => this.rowInfo(row),
      canEdit: () => this.editable(),
      commit: (changes, meta) => this.onGridCommit(changes, meta),
      onActiveChange: (row, col) => { this.focusArea = 'grid'; this.syncScan(row, col); },
      onHoverRow: (row) => this.viewer.hover(row ? this.viewer.rowBox(row.giorno) : null),
      onEditEnd: () => { if (this.needsGridRefresh) { this.needsGridRefresh = false; this.grid.setRows(this.doc.rows); } },
      onSaveShortcut: () => this.saveNow(),
      emptyHtml: '',
    });
    host.addEventListener('focusin', () => { this.focusArea = 'grid'; this.syncScan(); });
  }

  /* ============================================================ caricamento */
  async open(id, query) {
    if (!id) { this.ctx.navigate('#/documenti', { replace: true }); return; }
    if (this.id && this.id !== id) await this.flush();
    this.id = id;
    this.doc = null;
    this.pending = { rows: new Map(), header: {}, extra: {} };
    this.grid.undoStack = [];
    this.grid.redoStack = [];
    this.setSaveState('idle');
    this.renderNav();
    $('#rv-loading', this.root).hidden = false;
    let data;
    try {
      data = await get(`/api/documents/${encodeURIComponent(id)}`);
    } catch (err) {
      if (this.id !== id) return;
      this.renderMissing(err);
      return;
    }
    if (this.id !== id) return;
    this.setDoc(data, { fresh: true });
    try {
      await this.viewer.load(authUrl(data.image_url), data.grid);
    } catch (err) {
      toast({ type: 'error', title: 'Scansione non disponibile', message: err.message });
    }
    $('#rv-loading', this.root).hidden = true;
    if (this.id !== id) return;
    this.updateMarks();
    this.applyQuery(query, true);
  }

  applyQuery(query, initial = false) {
    const g = Number(query?.get?.('g'));
    const c = query?.get?.('c');
    if (c && HEADER_ORDER.includes(c) && !g) { this.focusHeader(c); return; }
    if (g >= 1 && g <= 31) {
      this.grid.selectByKey(String(g), c && DAY_COLS.includes(c) ? c : 'eff_entrata', { focus: true });
      return;
    }
    if (initial) {
      // prima cella da verificare, altrimenti il primo giorno con dati
      const first = this.flaggedCells().find((f) => f.kind === 'cell');
      if (first) this.grid.selectByKey(String(first.g), first.col, { focus: true });
      else {
        const row = this.doc?.rows.find((rr) => rr.prog_entrata || rr.eff_entrata) || this.doc?.rows[0];
        if (row) this.grid.selectByKey(String(row.giorno), 'eff_entrata', { focus: true });
      }
    }
  }

  renderMissing(err) {
    this.broken = true;
    $('#rv-loading', this.root).hidden = true;
    const body = $('#rv-body', this.root);
    setHTML(body, html`<div class="card review-state" style="grid-column:1/-1">${emptyState({
      art: ART.documents,
      title: err.status === 404 ? 'Documento non trovato' : 'Impossibile aprire il documento',
      text: err.status === 404 ? 'Il documento potrebbe essere stato eliminato.' : err.message,
      action: html`<a class="btn btn-primary" href="#/documenti">${icon('arrow-left')}Torna ai documenti</a>`,
    })}</div>`);
    $('#rv-title', this.root).textContent = 'Documento non disponibile';
  }

  /** Imposta il documento (risposta del server) riapplicando le modifiche non ancora salvate. */
  setDoc(data, { fresh = false } = {}) {
    for (const [g, patch] of this.pending.rows) this.applyRowPatch(data, g, patch);
    if (Object.keys(this.pending.header).length) this.applyHeaderPatch(data, this.pending.header);
    this.doc = data;
    this.giorni = new Map((data.giorni || []).map((x) => [x.giorno, x]));
    this.anomByDay = new Map();
    for (const a of data.anomalies || []) {
      if (!a.giorno) continue;
      if (!this.anomByDay.has(a.giorno)) this.anomByDay.set(a.giorno, []);
      this.anomByDay.get(a.giorno).push(a);
    }
    if (!fresh && data.grid && JSON.stringify(data.grid) !== JSON.stringify(this.viewer.grid)) this.viewer.setGrid(data.grid);
    this.grid.readOnly = !this.editable();
    if (this.grid.editing) this.needsGridRefresh = true;
    else this.grid.setRows(data.rows);
    this.renderTitle();
    this.renderHeader();
    this.renderAnoms();
    this.renderTotals();
    this.renderOverlay();
    this.renderFlags();
    this.renderNav();
    if (!fresh) this.updateMarks();
  }

  editable() {
    return !!this.doc && this.doc.status !== 'in_coda' && this.doc.status !== 'in_lavorazione';
  }

  onState(kind) {
    if (kind !== 'state' || !this.doc) return;
    this.renderNav();
    const s = docById(this.id);
    if (!s) return;
    const busyNow = s.status === 'in_coda' || s.status === 'in_lavorazione';
    const wasBusy = this.doc.status === 'in_coda' || this.doc.status === 'in_lavorazione';
    if (busyNow) {
      this.doc.status = s.status;
      this.doc.progress = s.progress;
      this.doc.status_message = s.status_message;
      this.renderOverlay();
      this.renderTitle();
      return;
    }
    if (wasBusy || (s.updated_at !== this.doc.updated_at && !this.saving && !this.hasPending())) {
      // lettura terminata (o documento cambiato altrove): ricarica
      get(`/api/documents/${encodeURIComponent(this.id)}`).then((data) => {
        if (data.id !== this.id || this.saving || this.hasPending()) return;
        this.setDoc(data);
        if (wasBusy) toast({ type: 'success', title: 'Lettura completata', message: 'I dati del documento sono stati aggiornati.', duration: 3000 });
      }).catch(() => {});
    }
  }

  /* ============================================================ intestazione pagina */
  renderTitle() {
    const d = this.doc;
    if (!d) return;
    const h = d.header || {};
    $('#rv-title', this.root).textContent = h.operatore || d.display_name;
    const parts = [];
    if (h.alunno) parts.push(`Alunno: ${h.alunno}`);
    if (h.mese) parts.push(capitalize(fmtPeriodo(h.mese, h.anno)));
    parts.push(`${d.source_file}${d.page_count > 1 ? ` (pag. ${d.source_page}/${d.page_count})` : ''}`);
    $('#rv-sub', this.root).textContent = parts.join(' · ');
    setHTML($('#rv-badge', this.root), statusBadge(d));
    const btn = $('#rv-confirm', this.root);
    const isFF = d.status === 'completato' && d.is_foglio_firma !== false;
    btn.hidden = !isFF && !d.user_verified;
    if (d.user_verified) {
      btn.className = 'btn btn-sm btn-verified';
      setHTML(btn, html`${icon('check-circle')}Confermato`);
      btn.setAttribute('data-tip', 'Documento confermato: clic per annullare la conferma (Ctrl+Invio)');
    } else {
      btn.className = 'btn btn-sm btn-success';
      setHTML(btn, html`${icon('check')}Conferma documento`);
      btn.setAttribute('data-tip', 'Segna il documento come verificato (Ctrl+Invio)');
    }
    btn.disabled = !this.editable();
    $('#rv-reprocess', this.root).disabled = !this.editable() && d.status !== 'errore';
    document.title = `${h.operatore || d.display_name} — Revisione — Sirio OCR`;
  }

  renderNav() {
    const order = reviewOrder();
    const i = order.indexOf(this.id);
    const pos = $('#rv-pos', this.root);
    if (pos) pos.textContent = i >= 0 ? `${i + 1} di ${order.length}` : '';
    const prev = $('#rv-prev', this.root); const next = $('#rv-next', this.root);
    if (prev) prev.disabled = i <= 0;
    if (next) next.disabled = i < 0 || i >= order.length - 1;
  }

  async go(delta) {
    const order = reviewOrder();
    const i = order.indexOf(this.id);
    const target = order[i + delta];
    if (!target) return;
    await this.flush();
    this.ctx.navigate(`#/revisione/${encodeURIComponent(target)}`);
  }

  setSaveState(state, err) {
    this.saveState = state;
    const el = $('#rv-save', this.root);
    if (!el) return;
    el.className = `save-ind is-${state}`;
    if (state === 'saving') setHTML(el, html`<span class="spinner"></span>Salvataggio…`);
    else if (state === 'dirty') setHTML(el, html`${icon('edit')}Modifiche in sospeso`);
    else if (state === 'saved') setHTML(el, html`${icon('check-circle')}<span class="txt">Salvato alle ${fmtTime(this.lastSaved || new Date())}</span>`);
    else if (state === 'error') setHTML(el, html`${icon('alert-circle')}<span data-tip="${err?.message || ''}">Non salvato</span><button type="button" class="link-btn" data-retry>Riprova</button>`);
    else setHTML(el, html`${icon('check')}<span class="txt">Salvataggio automatico</span>`);
    el.setAttribute('data-tip', state === 'saved' ? `Tutte le modifiche sono salvate (${fmtTime(this.lastSaved || new Date())})` : 'Le modifiche vengono salvate automaticamente');
  }

  /* ============================================================ valori e stati delle celle */
  rowByG(g) { return this.doc?.rows.find((r) => r.giorno === g) || null; }

  cellValue(row, key) {
    if (key === 'ore_calcolate') return hoursBetween(row.eff_entrata, row.eff_uscita);
    if (key === 'gg' || key === 'esito') return null;
    if ((key === 'eff_entrata' || key === 'eff_uscita') && emptyVal(row[key]) && row.trattino_effettivo) return '-';
    return row[key] ?? null;
  }

  cellState(row, key) {
    if (!DAY_COLS.includes(key) || !this.doc) return null;
    const k = `rows.${row.giorno}.${key}`;
    if ((row.illeggibili || []).includes(key)) {
      return {
        state: 'illeggibile', tone: 'illeg', tipTitle: 'Illeggibile',
        tip: 'Inserire il valore leggendo la scansione.\nCanc se la cella è effettivamente vuota.',
      };
    }
    if ((row.incerti || []).includes(key)) {
      const conf = row.confidenza?.[key];
      const val = this.cellValue(row, key);
      const shown = typeof val === 'boolean' ? (val ? 'presente' : 'assente') : (key === 'ore_dichiarate' ? fmtOre(val) : (val ?? 'vuoto'));
      return {
        state: 'incerto', tone: 'uncert',
        tipTitle: `Lettura incerta${conf ? ` (affidabilità ${Math.round(conf * 100)}%)` : ''}`,
        tip: `Valore letto: ${shown}. Verificare sulla scansione.\nInvio due volte (o Invio sulle caselle) conferma il valore.`,
      };
    }
    if ((this.doc.user_edited || []).includes(k)) {
      const orig = this.doc.ocr_originali?.[k];
      return { state: 'corretto', tone: 'edited', tipTitle: 'Modificato a mano', tip: `Valore OCR originale: ${fmtOrig(orig ?? null)}` };
    }
    return null;
  }

  rowInfo(row) {
    const info = this.giorni.get(row.giorno);
    const out = { cls: '', tip: '' };
    if (info) {
      if (info.tipo_giorno === 'sabato' || info.tipo_giorno === 'domenica') { out.cls = 'is-weekend'; out.tip = capitalize(info.giorno_settimana === 'sab' ? 'sabato' : 'domenica'); }
      else if (info.tipo_giorno === 'festivo') { out.cls = 'is-holiday'; out.tip = `Festività: ${info.festivita || 'giorno festivo'}`; }
      else if (info.tipo_giorno === 'inesistente') { out.cls = 'is-missing'; out.tip = 'Giorno inesistente nel mese di riferimento'; }
    }
    const anoms = this.anomByDay?.get(row.giorno) || [];
    if (anoms.some((a) => a.gravita === 'errore')) out.cls += ' mark-errore';
    else if (anoms.some((a) => a.gravita === 'attenzione')) out.cls += ' mark-attenzione';
    return out;
  }

  renderWeekday(row) {
    const info = this.giorni.get(row.giorno);
    if (!info || !info.giorno_settimana) return '<span class="gg is-we">—</span>';
    const we = info.tipo_giorno !== 'feriale';
    const fest = info.tipo_giorno === 'festivo' ? `<span class="gg-fest" aria-label="festivo"></span>` : '';
    return `<span class="gg${we ? ' is-we' : ''}">${esc(info.giorno_settimana)}</span>${fest}`;
  }

  renderCalc(row) {
    const calc = hoursBetween(row.eff_entrata, row.eff_uscita);
    if (calc === null) return '';
    const dich = row.ore_dichiarate;
    const bad = dich !== null && dich !== undefined && Math.abs(dich - calc) > 0.01 && !row.assenza_alunno;
    return bad
      ? `<span class="is-bad-value" style="color:var(--danger-text);font-weight:650" data-tip="Le ore dichiarate (${esc(fmtOre(dich))}) non corrispondono all'orario (${esc(fmtOre(calc))})">${esc(fmtOre(calc))}</span>`
      : esc(fmtOre(calc));
  }

  renderEsito(row) {
    const info = this.giorni.get(row.giorno);
    const text = info?.esito || '';
    if (!text) return '';
    let tone = 'tone-ok'; let label = 'OK';
    if (text.startsWith('Errore')) { tone = 'tone-err'; label = 'Errore'; }
    else if (text.startsWith('Da verificare')) { tone = 'tone-warn'; label = 'Da verificare'; }
    else if (text === 'Assenza operatore') { tone = 'tone-neutral'; label = 'Ass. operatore'; }
    else if (text === 'Assenza alunno') { tone = 'tone-info'; label = 'Ass. alunno'; }
    else if (text === 'Non svolto') { tone = 'tone-neutral'; label = 'Non svolto'; }
    const anoms = (this.anomByDay?.get(row.giorno) || []).map((a) => `• ${a.messaggio}`).join('\n');
    const tip = anoms ? `${text}\n${anoms}` : text;
    return `<span class="esito ${tone}" data-tip="${esc(tip)}">${esc(label)}</span>`;
  }

  /* ============================================================ modifica della griglia */
  onGridCommit(changes, meta) {
    if (!this.editable()) {
      toast({ type: 'warning', title: 'Documento in lettura', message: 'Attendere il termine della lettura prima di modificarlo.' });
      return;
    }
    for (const ch of changes) {
      const g = ch.row.giorno;
      const key = ch.col.key;
      const patch = this.pending.rows.get(g) || { giorno: g };
      if (ch.confirm) {
        const row = this.rowByG(g);
        if (!row) continue;
        row.incerti = (row.incerti || []).filter((f) => f !== key);
        row.illeggibili = (row.illeggibili || []).filter((f) => f !== key);
        patch.incerti = [...row.incerti];
        patch.illeggibili = [...row.illeggibili];
        this.pending.rows.set(g, patch);
        toast({ type: 'success', title: 'Valore confermato', message: `Giorno ${g}, ${LABELS[key].toLowerCase()}: la lettura è stata verificata.`, duration: 2200 });
        continue;
      }
      patch[key] = ch.value === undefined ? null : ch.value;
      const local = { [key]: patch[key] };
      if ((key === 'eff_entrata' || key === 'eff_uscita') && patch[key] === null && ch.old === '-') {
        // cancellato il trattino: la prestazione non è più segnata come «non svolta»
        patch.trattino_effettivo = false;
        local.trattino_effettivo = false;
      }
      this.pending.rows.set(g, patch);
      this.applyRowPatch(this.doc, g, local);
    }
    this.markDirty();
    this.renderTotals();
    this.renderFlags();
    this.updateMarks();
    if (meta?.source === 'undo' || meta?.source === 'redo') this.flushSoon();
  }

  /** Applica localmente (ottimistico) una modifica di riga, come fa il server. */
  applyRowPatch(doc, g, patch) {
    const row = doc.rows.find((r) => r.giorno === g);
    if (!row) return;
    if (patch.incerti) row.incerti = [...patch.incerti];
    if (patch.illeggibili) row.illeggibili = [...patch.illeggibili];
    let dash = false;
    for (const key of DAY_COLS) {
      if (!(key in patch)) continue;
      let value = patch[key];
      if ((key.endsWith('entrata') || key.endsWith('uscita')) && value === '-') {
        value = null;
        if (key.startsWith('eff_')) dash = true;
      }
      this.track(doc, row, key, `rows.${g}.${key}`, row[key] ?? null, value);
    }
    if ('trattino_effettivo' in patch) row.trattino_effettivo = !!patch.trattino_effettivo;
    else if (dash) row.trattino_effettivo = true;
    else if ((row.eff_entrata || row.eff_uscita) && ('eff_entrata' in patch || 'eff_uscita' in patch)) row.trattino_effettivo = false;
  }

  applyHeaderPatch(doc, patch) {
    const h = doc.header;
    if (patch.incerti) h.incerti = [...patch.incerti];
    if (patch.illeggibili) h.illeggibili = [...patch.illeggibili];
    for (const key of HEADER_ORDER) {
      if (key in patch) this.track(doc, h, key, `header.${key}`, h[key] ?? null, patch[key]);
    }
  }

  track(doc, obj, key, k, old, value) {
    const same = (emptyVal(old) && emptyVal(value)) || old === value;
    if (same) return;
    obj[key] = value;
    obj.incerti = (obj.incerti || []).filter((f) => f !== key);
    obj.illeggibili = (obj.illeggibili || []).filter((f) => f !== key);
    doc.ocr_originali = doc.ocr_originali || {};
    doc.user_edited = doc.user_edited || [];
    if (!(k in doc.ocr_originali) && !doc.user_edited.includes(k)) doc.ocr_originali[k] = asOriginal(old);
    const orig = doc.ocr_originali[k];
    if (orig !== null && orig !== undefined && asOriginal(value) === orig) {
      delete doc.ocr_originali[k];
      doc.user_edited = doc.user_edited.filter((x) => x !== k);
      return;
    }
    if (!doc.user_edited.includes(k)) doc.user_edited.push(k);
  }

  hasPending() {
    return this.pending.rows.size > 0 || Object.keys(this.pending.header).length > 0 || Object.keys(this.pending.extra).length > 0;
  }

  markDirty() {
    this.setSaveState('dirty');
    this.scheduleSave();
  }

  flushSoon() { this.scheduleSave.cancel(); setTimeout(() => this.flush(), 0); }

  async saveNow() {
    await this.flush();
    if (this.saveState !== 'error') toast({ type: 'success', title: 'Modifiche salvate', duration: 1800 });
  }

  /** Invia le modifiche in sospeso (una richiesta alla volta). */
  async flush() {
    this.scheduleSave.cancel();
    if (this.saving) { await this.saving.catch(() => {}); }
    if (!this.hasPending() || !this.id) return;
    const id = this.id;
    const body = {};
    if (this.pending.rows.size) body.rows = Array.from(this.pending.rows.values());
    if (Object.keys(this.pending.header).length) body.header = { ...this.pending.header };
    Object.assign(body, this.pending.extra);
    this.pending = { rows: new Map(), header: {}, extra: {} };
    this.setSaveState('saving');
    this.saving = put(`/api/documents/${encodeURIComponent(id)}`, body);
    try {
      const data = await this.saving;
      this.saving = null;
      if (this.id !== id) return;
      this.lastSaved = new Date();
      this.setDoc(data);
      this.setSaveState(this.hasPending() ? 'dirty' : 'saved');
      refreshState().catch(() => {});
      if (this.hasPending()) this.scheduleSave();
    } catch (err) {
      this.saving = null;
      if (this.id !== id) return;
      this.setSaveState('error', err);
      toastError(err, 'Modifica non salvata');
      // riallinea con il server (le modifiche rifiutate vengono annullate)
      try {
        const data = await get(`/api/documents/${encodeURIComponent(id)}`);
        if (this.id === id) this.setDoc(data);
      } catch { /* resta lo stato di errore */ }
    }
  }

  /* ============================================================ intestazione */
  headerState(key) {
    const h = this.doc?.header;
    if (!h) return null;
    if ((h.illeggibili || []).includes(key)) return 'illeg';
    if ((h.incerti || []).includes(key)) return 'uncert';
    if ((this.doc.user_edited || []).includes(`header.${key}`)) return 'edited';
    return null;
  }

  headerTip(key, st) {
    if (st === 'illeg') return 'Illeggibile: inserire il valore leggendo la scansione.';
    if (st === 'uncert') return 'Lettura incerta: verificare sulla scansione. Premere ✓ per confermare il valore.';
    if (st === 'edited') return `Modificato a mano · valore OCR originale: ${fmtOrig(this.doc.ocr_originali?.[`header.${key}`] ?? null)}`;
    return '';
  }

  renderHeader() {
    const d = this.doc;
    const h = d.header || {};
    const flagged = HEADER_ORDER.filter((k) => ['illeg', 'uncert'].includes(this.headerState(k)));
    if (flagged.some((k) => HEADER_MORE.has(k))) this.hdrMore = true;
    const active = document.activeElement;
    const focusedKey = this.hdrEl.contains(active) ? active.dataset.key : null;
    // testo che l'utente sta digitando (non ancora confermato): va conservato
    const typing = focusedKey && active.matches('input.input') && active.value !== (active.dataset.orig ?? '')
      ? { value: active.value, start: active.selectionStart, end: active.selectionEnd } : null;
    const ro = !this.editable();
    const label = (key) => {
      const st = this.headerState(key);
      const ic = st === 'illeg' ? raw(`<span class="st-illeg">${icon('alert-triangle')}</span>`)
        : st === 'uncert' ? raw(`<span class="st-uncert">${icon('help')}</span>`)
          : st === 'edited' ? raw(`<span class="st-edited">${icon('edit')}</span>`) : '';
      return html`<label class="field-label" for="hf-${key}" ${st ? raw(`data-tip="${esc(this.headerTip(key, st))}" data-tip-tone="${st === 'illeg' ? 'illeg' : st === 'uncert' ? 'uncert' : 'edited'}"`) : ''}>${ic}${LABELS[key]}</label>`;
    };
    const cls = (key) => { const st = this.headerState(key); return st ? ` is-${st}` : ''; };
    const confirmBtn = (key) => (this.headerState(key) === 'uncert' && !ro
      ? html`<button type="button" class="btn btn-sm btn-ghost btn-icon input-addon" data-confirm="${key}" aria-label="Conferma il valore letto" data-tip="Conferma il valore letto">${icon('check')}</button>` : '');
    const reg = store.registry?.campi || {};
    const text = (key, span, ph = '') => {
      const v = h[key] ?? '';
      const st = this.headerState(key);
      const list = reg[key]?.length ? raw(` list="dl-${key}"`) : '';
      return html`<div class="field span-${span} f-${key}">${label(key)}<div class="input-group">
        <input class="input${cls(key)}" id="hf-${key}" data-key="${key}" data-kind="text" value="${v}" data-orig="${v}" placeholder="${st === 'illeg' ? 'Illeggibile' : ph}" autocomplete="off" spellcheck="false"${list} ${ro ? raw('readonly') : ''}>
        ${confirmBtn(key)}</div></div>`;
    };
    const hours = (key, span) => {
      const v = h[key] === null || h[key] === undefined ? '' : fmtOre(h[key]);
      const st = this.headerState(key);
      return html`<div class="field span-${span} f-${key}">${label(key)}<div class="input-group">
        <input class="input num${cls(key)}" id="hf-${key}" data-key="${key}" data-kind="hours" value="${v}" data-orig="${v}" inputmode="decimal" placeholder="${st === 'illeg' ? 'Illeggibile' : '—'}" autocomplete="off" ${ro ? raw('readonly') : ''}>
        ${confirmBtn(key)}</div></div>`;
    };
    const toggle = (key, span) => {
      const st = this.headerState(key);
      return html`<div class="field span-${span} f-${key}">${label(key)}
        <label class="toggle-field${st ? ` is-${st}` : ''}">
          <span>${h[key] ? 'Presente' : 'Assente'}</span>
          <span class="switch is-sm"><input type="checkbox" id="hf-${key}" data-key="${key}" data-kind="bool" ${h[key] ? raw('checked') : ''} ${ro ? raw('disabled') : ''}><span class="switch-track"></span></span>
        </label></div>`;
    };
    const monthOpts = [html`<option value="">—</option>`, ...MESI.map((m, i) => html`<option value="${i + 1}" ${h.mese === i + 1 ? raw('selected') : ''}>${capitalize(m)}</option>`)];
    const nFlag = flagged.length;
    const summary = [h.operatore, h.alunno, h.istituto, h.mese ? capitalize(fmtPeriodo(h.mese, h.anno)) : '', h.ore_pei != null ? `PEI ${fmtOre(h.ore_pei)} h/sett.` : ''].filter(Boolean).join(' · ');
    this.hdrEl.className = `card hdr-card${this.hdrCollapsed ? ' is-collapsed' : ''}${this.hdrMore ? ' is-expanded' : ''}`;
    setHTML(this.hdrEl, html`
      <div class="hdr-top">
        <div class="card-title">Intestazione</div>
        ${nFlag ? html`<span class="badge badge-sm tone-${flagged.some((k) => this.headerState(k) === 'illeg') ? 'illeg' : 'uncert'}">${plural(nFlag, 'campo da verificare', 'campi da verificare')}</span>` : ''}
        <div class="hdr-summary">${summary}</div>
        <span class="spacer"></span>
        <button type="button" class="btn btn-sm btn-ghost" data-hdr="more" ${this.hdrCollapsed ? raw('hidden') : ''}>${this.hdrMore ? 'Meno campi' : 'Altri campi'}</button>
        <button type="button" class="btn btn-sm btn-ghost btn-icon" data-hdr="collapse" aria-expanded="${String(!this.hdrCollapsed)}" aria-label="${this.hdrCollapsed ? 'Espandi l\'intestazione' : 'Comprimi l\'intestazione'}">${icon(this.hdrCollapsed ? 'chevron-down' : 'chevron-up')}</button>
      </div>
      <div class="hdr-grid">
        ${text('operatore', 3, 'Cognome e nome')}
        ${text('alunno', 3, 'Cognome e nome')}
        ${text('istituto', 3)}
        ${text('ente', 3)}
        <div class="field span-2 f-mese">${label('mese')}<select class="select${cls('mese')}" id="hf-mese" data-key="mese" data-kind="month" ${ro ? raw('disabled') : ''}>${monthOpts}</select></div>
        <div class="field span-1 f-anno">${label('anno')}<input class="input num${cls('anno')}" id="hf-anno" data-key="anno" data-kind="year" value="${h.anno ?? ''}" data-orig="${h.anno ?? ''}" inputmode="numeric" maxlength="4" autocomplete="off" ${ro ? raw('readonly') : ''}></div>
        ${hours('ore_pei', 1)}
        <div class="field span-2 f-sostituzione">${label('sostituzione')}<select class="select${cls('sostituzione')}" id="hf-sostituzione" data-key="sostituzione" data-kind="sost" ${ro ? raw('disabled') : ''}>
          <option value="" ${!h.sostituzione ? raw('selected') : ''}>—</option><option value="NO" ${h.sostituzione === 'NO' ? raw('selected') : ''}>No</option><option value="SI" ${h.sostituzione === 'SI' ? raw('selected') : ''}>Sì</option></select></div>
        ${hours('totale_mensile_dichiarato', 2)}
        ${toggle('firma_coordinatore', 2)}
        ${toggle('timbro_referente', 2)}
        <div class="hdr-more">
          ${text('lotto', 2)}
          ${text('municipalita', 2)}
          ${text('anno_scolastico', 3, 'es. 2025/2026')}
          ${text('data_compilazione', 3, 'gg/mm/aaaa')}
        </div>
      </div>
      ${['operatore', 'alunno', 'istituto', 'ente'].filter((k) => reg[k]?.length).map((k) => html`<datalist id="dl-${k}">${reg[k].map((e) => html`<option value="${e.valore}"></option>`)}</datalist>`)}`);
    if (focusedKey) {
      const el = this.hdrEl.querySelector(`[data-key="${focusedKey}"]`);
      if (el && el !== active) {
        el.focus();
        if (typing && el.matches('input.input')) {
          el.value = typing.value;
          try { el.setSelectionRange(typing.start, typing.end); } catch { /* campo senza selezione */ }
        }
      }
    }
  }

  onHeaderClick(e) {
    const b = e.target.closest('[data-hdr],[data-confirm]');
    if (!b) return;
    if (b.dataset.confirm) {
      const key = b.dataset.confirm;
      const h = this.doc.header;
      h.incerti = (h.incerti || []).filter((f) => f !== key);
      h.illeggibili = (h.illeggibili || []).filter((f) => f !== key);
      this.pending.header.incerti = [...h.incerti];
      this.pending.header.illeggibili = [...h.illeggibili];
      this.markDirty();
      this.renderHeader();
      this.renderFlags();
      toast({ type: 'success', title: 'Valore confermato', message: `${LABELS[key]}: la lettura è stata verificata.`, duration: 2200 });
      return;
    }
    if (b.dataset.hdr === 'collapse') {
      this.hdrCollapsed = !this.hdrCollapsed;
      storageSet('sirio.rv.hdrCollapsed', this.hdrCollapsed);
      this.renderHeader();
    } else if (b.dataset.hdr === 'more') {
      this.hdrMore = !this.hdrMore;
      this.renderHeader();
    }
  }

  onHeaderFocus(e) {
    const key = e.target.dataset?.key;
    if (!key) return;
    this.focusArea = 'header';
    this.headerFocusKey = key;
    const v = this.viewer;
    let box = v.headerBox();
    if (key === 'totale_mensile_dichiarato') box = v.totalBox() || box;
    else if (key === 'firma_coordinatore') box = v.coordinatorBox() || box;
    else if (key === 'timbro_referente' || key === 'data_compilazione') box = v.footerBox() || box;
    v.highlight({ cellBox: box, regionClass: 'hl-region' });
  }

  onHeaderChange(e) {
    const el = e.target;
    const key = el.dataset?.key;
    if (!key || !this.doc || !this.editable()) return;
    const kind = el.dataset.kind;
    let value;
    if (kind === 'bool') value = el.checked;
    else if (kind === 'month') value = el.value ? Number(el.value) : null;
    else if (kind === 'sost') value = el.value || null;
    else if (kind === 'year') {
      const t = el.value.trim();
      if (!t) value = null;
      else if (/^\d{2}$/.test(t)) value = 2000 + Number(t);
      else if (/^\d{4}$/.test(t) && Number(t) >= 2000 && Number(t) <= 2100) value = Number(t);
      else { this.headerError(el, 'Anno non valido: indicare quattro cifre, ad esempio 2026.'); return; }
    } else if (kind === 'hours') {
      const p = parseHours(el.value, key === 'ore_pei' ? 60 : 744);
      if (!p.ok) { this.headerError(el, p.error); return; }
      value = p.value;
    } else {
      const t = el.value.replace(/\s+/g, ' ').trim();
      if (t.length > 200) { this.headerError(el, 'Testo troppo lungo (massimo 200 caratteri).'); return; }
      value = t || null;
    }
    this.pending.header[key] = value;
    this.applyHeaderPatch(this.doc, { [key]: value });
    this.markDirty();
    this.renderHeader();
    this.renderTotals();
    this.renderFlags();
    if (key === 'mese' || key === 'anno' || key === 'totale_mensile_dichiarato') this.flushSoon();
  }

  headerError(el, message) {
    el.classList.add('is-illeg');
    toast({ type: 'warning', title: 'Valore non valido', message });
    el.value = el.dataset.orig ?? '';
    setTimeout(() => el.classList.remove('is-illeg'), 1600);
  }

  focusHeader(key) {
    if (this.hdrCollapsed) { this.hdrCollapsed = false; storageSet('sirio.rv.hdrCollapsed', false); }
    if (HEADER_MORE.has(key)) this.hdrMore = true;
    this.renderHeader();
    const el = this.hdrEl.querySelector(`[data-key="${key}"]`);
    if (el) { el.focus(); if (el.select) el.select(); }
  }

  /* ============================================================ scansione */
  syncScan(row = this.grid.activeRow, col = this.grid.activeCol) {
    if (!row || !col || !this.viewer.ready) return;
    if (this.doc && (this.doc.is_foglio_firma === false || this.doc.status === 'scartato')) {
      // pagina non riconosciuta: nessuna tabella da evidenziare
      this.viewer.highlight({});
      $('#rv-tag', this.root).textContent = '';
      return;
    }
    const g = row.giorno;
    let cellBox = null;
    if (DAY_COLS.includes(col.key)) cellBox = this.viewer.cellBox(g, col.key);
    else if (col.key === 'ore_calcolate') cellBox = this.viewer.spanBox(g, 'eff_entrata', 'eff_uscita');
    else if (col.key === 'giorno' || col.key === 'gg') cellBox = this.viewer.cellBox(g, 'giorno');
    this.viewer.highlight({ rowBox: this.viewer.rowBox(g), cellBox });
    const tag = $('#rv-tag', this.root);
    if (tag) tag.textContent = `Giorno ${g} · ${col.key === 'gg' ? 'giorno della settimana' : (LABELS[col.key] || col.label || '').toLowerCase()}`;
  }

  onScanPick(g, colName) {
    if (!g) return;
    const key = colName === 'giorno' ? 'eff_entrata' : colName;
    this.grid.selectByKey(String(g), key, { focus: true });
  }

  updateMarks() {
    if (!this.doc || !this.viewer.ready) return;
    if (this.doc.is_foglio_firma === false || this.doc.status === 'scartato') { this.viewer.setMarks([]); return; }
    const marks = [];
    for (const row of this.doc.rows) {
      for (const key of row.illeggibili || []) marks.push({ kind: 'illeggibile', box: this.viewer.cellBox(row.giorno, key) });
      for (const key of row.incerti || []) if (!(row.illeggibili || []).includes(key)) marks.push({ kind: 'incerto', box: this.viewer.cellBox(row.giorno, key) });
    }
    const h = this.doc.header || {};
    if ((h.illeggibili || []).includes('totale_mensile_dichiarato')) marks.push({ kind: 'illeggibile', box: this.viewer.totalBox() });
    else if ((h.incerti || []).includes('totale_mensile_dichiarato')) marks.push({ kind: 'incerto', box: this.viewer.totalBox() });
    this.viewer.setMarks(marks);
  }

  /* ============================================================ campi da verificare (F8) */
  flaggedCells() {
    const out = [];
    if (!this.doc) return out;
    const h = this.doc.header || {};
    for (const key of HEADER_ORDER) {
      if ((h.illeggibili || []).includes(key) || (h.incerti || []).includes(key)) out.push({ kind: 'header', key, order: HEADER_ORDER.indexOf(key) });
    }
    const colOrder = this.grid.cols.map((c) => c.key);
    for (const row of this.doc.rows) {
      const set = new Set([...(row.illeggibili || []), ...(row.incerti || [])]);
      for (const key of DAY_COLS) {
        if (set.has(key)) out.push({ kind: 'cell', g: row.giorno, col: key, order: 100 + row.giorno * 100 + colOrder.indexOf(key) });
      }
    }
    return out.sort((a, b) => a.order - b.order);
  }

  renderFlags() {
    const n = this.flaggedCells().length;
    const el = $('#rv-flag-count', this.root);
    if (el) el.textContent = String(n);
    const btn = $('#rv-next-flag', this.root);
    if (btn) {
      btn.classList.toggle('btn-ghost', n === 0);
      btn.disabled = n === 0;
    }
  }

  nextFlag(dir = 1) {
    const list = this.flaggedCells();
    if (!list.length) {
      toast({ type: 'success', title: 'Nessun campo da verificare', message: 'Tutti i campi illeggibili e incerti sono stati controllati.', duration: 3000 });
      return;
    }
    let cur = -1;
    if (this.focusArea === 'header' && this.headerFocusKey) cur = HEADER_ORDER.indexOf(this.headerFocusKey);
    else if (this.grid.active) {
      const colOrder = this.grid.cols.map((c) => c.key);
      cur = 100 + this.grid.activeRow.giorno * 100 + colOrder.indexOf(this.grid.activeCol.key);
    }
    let target = dir > 0 ? list.find((f) => f.order > cur) : [...list].reverse().find((f) => f.order < cur);
    if (!target) target = dir > 0 ? list[0] : list[list.length - 1];
    if (target.kind === 'header') this.focusHeader(target.key);
    else {
      this.focusArea = 'grid';
      this.grid.selectByKey(String(target.g), target.col, { focus: true });
      const r = this.grid.active?.r; const c = this.grid.active?.c;
      if (r !== undefined) this.grid.flash(r, c);
    }
  }

  /* ============================================================ anomalie */
  renderAnoms() {
    const list = this.doc.anomalies || [];
    const counts = Object.fromEntries(SEV.map((s) => [s.id, list.filter((a) => a.gravita === s.id).length]));
    this.anomsEl.classList.toggle('is-collapsed', this.anomsCollapsed);
    const groups = SEV.filter((s) => this.sevFilter.has(s.id) && counts[s.id]).map((s) => {
      const items = list.filter((a) => a.gravita === s.id);
      return html`<div class="anom-group">${s.label} · ${items.length}</div>${items.map((a) => {
        const idx = list.indexOf(a);
        const meta = [];
        if (a.giorno) meta.push(`Giorno ${a.giorno}`);
        if (a.campo && LABELS[a.campo]) meta.push(LABELS[a.campo]);
        return html`<button type="button" class="anom sev-${a.gravita}" data-a="${idx}">
          ${icon(s.icon)}<span class="anom-msg">${a.messaggio}</span>
          <span class="anom-meta">${meta.join(' · ')}${meta.length ? raw(' · ') : ''}<span class="mono">${a.codice}</span></span></button>`;
      })}`;
    });
    const total = list.length;
    setHTML(this.anomsEl, html`
      <div class="anoms-head">
        <button type="button" class="btn btn-sm btn-ghost btn-icon" data-anoms="toggle" aria-expanded="${String(!this.anomsCollapsed)}" aria-label="${this.anomsCollapsed ? 'Mostra le anomalie' : 'Nascondi le anomalie'}" style="margin-left:-8px">${icon(this.anomsCollapsed ? 'chevron-up' : 'chevron-down')}</button>
        <span class="anoms-title">Anomalie</span>
        <span class="muted num" style="font-size:12.5px">${total || ''}</span>
        <div class="anoms-filters">${SEV.map((s) => html`<button type="button" class="anom-chip ${s.chip}" data-sev="${s.id}" aria-pressed="${String(this.sevFilter.has(s.id))}" data-tip="${this.sevFilter.has(s.id) ? 'Nascondi' : 'Mostra'}: ${s.label.toLowerCase()}">${icon(s.icon)}${counts[s.id]}</button>`)}</div>
      </div>
      <div class="anoms-list" role="list">${total
        ? (groups.length ? groups : html`<div class="anoms-empty muted">Nessuna anomalia con i filtri selezionati.</div>`)
        : html`<div class="anoms-empty">${icon('check-circle')}Nessuna anomalia: il documento è coerente.</div>`}</div>`);
  }

  onAnomClick(e) {
    const t = e.target.closest('[data-sev],[data-anoms],[data-a]');
    if (!t) return;
    if (t.dataset.sev) {
      const s = t.dataset.sev;
      if (this.sevFilter.has(s)) this.sevFilter.delete(s); else this.sevFilter.add(s);
      storageSet('sirio.rv.sev', Array.from(this.sevFilter));
      this.renderAnoms();
      return;
    }
    if (t.dataset.anoms) {
      this.anomsCollapsed = !this.anomsCollapsed;
      storageSet('sirio.rv.anomsCollapsed', this.anomsCollapsed);
      this.renderAnoms();
      return;
    }
    const a = (this.doc.anomalies || [])[Number(t.dataset.a)];
    if (!a) return;
    if (a.giorno) {
      let col = a.campo;
      if (col === 'trattino_effettivo') col = 'eff_entrata';
      if (!DAY_COLS.includes(col)) col = a.codice === 'E01_ORE_NON_COERENTI' ? 'ore_dichiarate' : (this.grid.activeCol?.key && DAY_COLS.includes(this.grid.activeCol.key) ? this.grid.activeCol.key : 'eff_entrata');
      this.grid.selectByKey(String(a.giorno), col, { focus: true });
      const r = this.grid.active?.r;
      if (r !== undefined) this.grid.flash(r, this.grid.active.c);
    } else if (a.campo && HEADER_ORDER.includes(a.campo)) {
      this.focusHeader(a.campo);
    } else if (a.codice === 'E02_TOTALE_MENSILE_DIVERSO' || a.codice === 'W10_TOTALE_MENSILE_ASSENTE') {
      this.focusHeader('totale_mensile_dichiarato');
    }
  }

  onAnomHover(e) {
    const t = e.target.closest('[data-a]');
    const a = t ? (this.doc.anomalies || [])[Number(t.dataset.a)] : null;
    if (a?.giorno) {
      this.grid.setHoverRow(String(a.giorno));
      this.viewer.hover(this.viewer.rowBox(a.giorno));
    } else {
      this.grid.setHoverRow(null);
      this.viewer.hover(null);
    }
  }

  /* ============================================================ totali */
  renderTotals() {
    const d = this.doc;
    if (!d) return;
    const rows = d.rows;
    const dirty = this.hasPending() || !!this.saving;
    let dich = 0; let calc = 0; let ric = 0; let lav = 0; let aa = 0; let ao = 0;
    for (const r of rows) {
      const c = hoursBetween(r.eff_entrata, r.eff_uscita);
      if (r.ore_dichiarate !== null && r.ore_dichiarate !== undefined) dich += Number(r.ore_dichiarate) || 0;
      if (c !== null) calc += c;
      const o = r.assenza_operatore ? 0 : (r.ore_dichiarate ?? c ?? 0);
      ric += o;
      if (o > 0 && !r.assenza_operatore) lav++;
      if (r.assenza_alunno) aa++;
      if (r.assenza_operatore) ao++;
    }
    const t = d.totals || {};
    if (!dirty) { ric = t.ore_riconosciute ?? ric; lav = t.giorni_lavorati ?? lav; }
    const tot = d.header?.totale_mensile_dichiarato;
    const diff = tot === null || tot === undefined ? null : Math.round((dich - tot) * 100) / 100;
    const diffCls = diff === null ? '' : (Math.abs(diff) > 0.01 ? 'is-bad' : 'is-good');
    const calcWarn = (d.anomalies || []).some((a) => a.codice === 'E01_ORE_NON_COERENTI');
    setHTML($('#rv-totals', this.root), html`
      <div class="total"><span class="total-label">Ore dichiarate</span><span class="total-value">${fmtOre(dich) || '0'}</span></div>
      <div class="total ${calcWarn ? 'is-warn' : ''}" ${calcWarn ? raw('data-tip="In almeno un giorno le ore dichiarate non corrispondono agli orari effettivi (vedi le anomalie)."') : ''}><span class="total-label">Ore calcolate</span><span class="total-value">${fmtOre(calc) || '0'}</span></div>
      <div class="total" data-tip="Per ogni giorno: ore dichiarate, oppure calcolate dagli orari se mancanti; zero con assenza dell'operatore."><span class="total-label">Ore riconosciute</span><span class="total-value">${fmtOre(ric) || '0'}</span></div>
      <div class="total ${tot === null || tot === undefined ? 'is-warn' : ''}"><span class="total-label">Totale mensile</span><span class="total-value">${tot === null || tot === undefined ? raw('<small>non indicato</small>') : fmtOre(tot)}</span></div>
      <div class="total ${diffCls}" data-tip="Ore dichiarate meno il totale mensile scritto sul foglio"><span class="total-label">Differenza</span><span class="total-value">${diff === null ? '—' : fmtSigned(diff)}${diffCls === 'is-good' ? icon('check-circle', 'icon-sm') : ''}${diffCls === 'is-bad' ? icon('alert-triangle', 'icon-sm') : ''}</span></div>
      <div class="total"><span class="total-label">Giorni lavorati</span><span class="total-value">${lav}</span></div>
      <div class="total" data-tip="${aa} ${aa === 1 ? 'assenza' : 'assenze'} dell'alunno · ${ao} ${ao === 1 ? 'assenza' : 'assenze'} dell'operatore"><span class="total-label">Assenze al. · op.</span><span class="total-value">${aa}<small>·</small>${ao}</span></div>`);
  }

  /* ============================================================ stato del documento */
  renderOverlay() {
    const el = $('#rv-overlay', this.root);
    const d = this.doc;
    if (!el || !d) return;
    this.grid.readOnly = !this.editable();
    if (d.status === 'in_coda' || d.status === 'in_lavorazione') {
      const pct = Math.round((d.progress || 0) * 100);
      setHTML(el, html`<div class="review-overlay"><div class="card review-overlay-card">
        <span class="spinner" style="color:var(--accent);width:22px;height:22px"></span>
        <div style="font-weight:650">${d.status === 'in_coda' ? 'Documento in coda di lettura' : 'Lettura in corso…'}</div>
        <div class="muted" style="font-size:13px">${d.status_message || 'Attendere: i dati compariranno al termine.'}</div>
        <div class="progress ${d.status === 'in_lavorazione' ? 'is-active' : 'is-indeterminate'}" style="width:100%"><span style="width:${pct}%"></span></div>
      </div></div>`);
      return;
    }
    if (d.status === 'errore') {
      setHTML(el, html`<div class="review-overlay"><div class="card review-overlay-card">
        <div class="dialog-icon" style="background:var(--danger-soft);color:var(--danger)">${icon('x-circle')}</div>
        <div style="font-weight:650">Lettura non riuscita</div>
        <div class="muted" style="font-size:13px">${d.error || d.status_message || 'Errore durante la lettura.'}</div>
        <button type="button" class="btn btn-primary btn-sm" data-ov="reprocess">${icon('refresh')}Riprova la lettura</button>
      </div></div>`);
      el.querySelector('[data-ov]').addEventListener('click', () => this.reprocess(true));
      return;
    }
    if (d.status === 'scartato' || d.is_foglio_firma === false) {
      setHTML(el, html`<div class="review-overlay"><div class="card review-overlay-card">
        <div class="dialog-icon" style="background:var(--neutral-soft);color:var(--text-2)">${icon('slash')}</div>
        <div style="font-weight:650">Pagina non riconosciuta come foglio firma</div>
        <div class="muted" style="font-size:13px">La pagina è esclusa dall'Excel. Se si tratta di un foglio firma, includila e completa i dati a mano oppure rielaborala.</div>
        <div class="row" style="justify-content:center"><button type="button" class="btn btn-sm" data-ov="reprocess">${icon('refresh')}Rielabora</button>
        <button type="button" class="btn btn-sm btn-primary" data-ov="include">${icon('check')}È un foglio firma</button></div>
      </div></div>`);
      el.querySelector('[data-ov="reprocess"]').addEventListener('click', () => this.reprocess());
      el.querySelector('[data-ov="include"]').addEventListener('click', () => this.setFoglioFirma(true));
      return;
    }
    el.innerHTML = '';
  }

  async setFoglioFirma(value) {
    this.pending.extra.is_foglio_firma = value;
    await this.flush();
    toast({ type: 'success', title: value ? 'Documento incluso' : 'Documento escluso', message: value ? 'La pagina verrà inclusa nell\'Excel.' : 'La pagina non verrà inclusa nell\'Excel.', duration: 3000 });
  }

  async toggleVerified() {
    if (!this.doc || !this.editable()) return;
    const verify = !this.doc.user_verified;
    if (verify) {
      const illeg = this.flaggedCells().filter((f) => {
        if (f.kind === 'header') return (this.doc.header.illeggibili || []).includes(f.key);
        return (this.rowByG(f.g)?.illeggibili || []).includes(f.col);
      }).length;
      const errs = (this.doc.anomalies || []).filter((a) => a.gravita === 'errore').length;
      if (illeg || errs) {
        const parts = [];
        if (illeg) parts.push(`<li><b>${plural(illeg, 'campo illeggibile', 'campi illeggibili')}</b> ancora da compilare</li>`);
        if (errs) parts.push(`<li><b>${plural(errs, 'errore', 'errori')}</b> di coerenza</li>`);
        const ok = await confirmDialog({
          title: 'Confermare comunque il documento?',
          message: raw(`<p>Il documento presenta ancora:</p><ul>${parts.join('')}</ul><p>La conferma indica che i dati sono stati controllati sulla scansione.</p>`),
          confirmLabel: 'Conferma comunque',
          tone: 'warning',
        });
        if (!ok) return;
      }
    }
    this.pending.extra.user_verified = verify;
    this.doc.user_verified = verify;
    this.renderTitle();
    await this.flush();
    if (this.saveState === 'error') return;
    loadRegistry(true);
    if (verify) {
      const next = this.nextToVerify();
      toast({
        type: 'success',
        title: 'Documento confermato',
        message: next ? 'Puoi passare al prossimo foglio da verificare.' : 'Tutti i fogli sono stati verificati: puoi generare l\'Excel.',
        actions: next
          ? [{ label: 'Vai al successivo', onClick: () => this.ctx.navigate(`#/revisione/${encodeURIComponent(next)}`) }]
          : [{ label: 'Apri l\'anteprima Excel', onClick: () => this.ctx.navigate('#/anteprima') }],
      });
    } else {
      toast({ type: 'info', title: 'Conferma annullata', duration: 2500 });
    }
  }

  nextToVerify() {
    const order = reviewOrder();
    const i = order.indexOf(this.id);
    const seq = [...order.slice(i + 1), ...order.slice(0, Math.max(0, i))];
    return seq.find((id) => {
      const d = docById(id);
      return d && d.status === 'completato' && d.is_foglio_firma && !d.user_verified;
    }) || null;
  }

  async reprocess(skipConfirm = false) {
    if (!this.doc) return;
    if (!skipConfirm) {
      const edited = (this.doc.user_edited || []).length;
      const ok = await confirmDialog({
        title: 'Rielaborare il documento?',
        message: raw(`<p>La scansione verrà letta di nuovo con il motore attuale.</p>${edited || this.doc.user_verified
          ? `<p><b>Le ${edited ? plural(edited, 'correzione manuale', 'correzioni manuali') : 'conferme'}</b> verranno sostituite dalla nuova lettura.</p>` : ''}`),
        confirmLabel: 'Rielabora',
        tone: 'warning',
      });
      if (!ok) return;
    }
    this.pending = { rows: new Map(), header: {}, extra: {} };
    this.scheduleSave.cancel();
    try {
      const s = await post(`/api/documents/${encodeURIComponent(this.id)}/reprocess`);
      this.doc.status = s.status || 'in_coda';
      this.doc.progress = 0;
      this.doc.status_message = s.status_message || 'In coda';
      this.renderOverlay();
      this.renderTitle();
      this.setSaveState('idle');
      pokePolling();
    } catch (err) { toastError(err, 'Rielaborazione non riuscita'); }
  }

  moreMenu(anchor) {
    if (!this.doc) return;
    const items = [
      { label: 'Anteprima Excel di questo documento', icon: 'excel', onClick: () => this.ctx.navigate(`#/anteprima?ids=${encodeURIComponent(this.id)}`) },
      { label: 'Scorciatoie da tastiera', icon: 'keyboard', onClick: () => this.shortcuts() },
    ];
    if (this.doc.status === 'completato' && this.doc.is_foglio_firma) {
      items.push('sep', { label: 'Escludi: non è un foglio firma', icon: 'slash', onClick: () => this.setFoglioFirma(false) });
    }
    items.push('sep', { label: 'Elimina documento', icon: 'trash', danger: true, onClick: () => this.remove() });
    showMenu(anchor, items);
  }

  async remove() {
    const ok = await confirmDialog({
      title: 'Eliminare il documento?',
      message: 'I dati letti, le correzioni e l\'immagine della scansione verranno rimossi da Sirio OCR.',
      confirmLabel: 'Elimina',
      danger: true,
    });
    if (!ok) return;
    const order = reviewOrder();
    const i = order.indexOf(this.id);
    const next = order[i + 1] || order[i - 1] || null;
    this.pending = { rows: new Map(), header: {}, extra: {} };
    try {
      await del(`/api/documents/${encodeURIComponent(this.id)}`);
      toast({ type: 'success', title: 'Documento eliminato', duration: 2500 });
      await refreshState(true).catch(() => {});
      this.ctx.navigate(next ? `#/revisione/${encodeURIComponent(next)}` : '#/documenti');
    } catch (err) { toastError(err); }
  }

  shortcuts() {
    const rows = [
      ['Modifica la cella', '<kbd>Invio</kbd> <kbd>F2</kbd> o digitare'],
      ['Conferma e scendi / a destra', '<kbd>Invio</kbd> / <kbd>Tab</kbd>'],
      ['Annulla la modifica in corso', '<kbd>Esc</kbd>'],
      ['Svuota la cella', '<kbd>Canc</kbd>'],
      ['Spunta / togli la spunta', '<kbd>Spazio</kbd>'],
      ['Annulla / ripeti', '<kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd>'],
      ['Copia / incolla', '<kbd>Ctrl</kbd>+<kbd>C</kbd> / <kbd>Ctrl</kbd>+<kbd>V</kbd>'],
      ['Prossimo campo da verificare', '<kbd>F8</kbd> (<kbd>Maiusc</kbd>+<kbd>F8</kbd> indietro)'],
      ['Salva subito', '<kbd>Ctrl</kbd>+<kbd>S</kbd>'],
      ['Conferma il documento', '<kbd>Ctrl</kbd>+<kbd>Invio</kbd>'],
      ['Documento precedente / successivo', '<kbd>Alt</kbd>+<kbd>←</kbd> / <kbd>Alt</kbd>+<kbd>→</kbd>'],
      ['Orari accettati', '8 · 8.30 · 830 · 8:30 · – (trattino)'],
    ];
    dialog({
      title: 'Scorciatoie da tastiera',
      icon: 'keyboard',
      wide: true,
      message: raw(`<div class="shortcut-list">${rows.map(([a, b]) => `<span>${a}</span><span>${b}</span>`).join('')}</div>`),
      buttons: [{ label: 'Chiudi', value: true, kind: 'primary' }],
    });
  }

  /* ============================================================ tastiera */
  onKey(e) {
    if (document.querySelector('dialog[open]')) return;
    const ctrl = e.ctrlKey || e.metaKey;
    if (e.altKey && !ctrl && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      e.preventDefault();
      this.go(e.key === 'ArrowLeft' ? -1 : 1);
      return;
    }
    if (e.key === 'F8') {
      e.preventDefault();
      if (this.grid.editing) { if (!this.grid.commitEdit()) return; }
      this.nextFlag(e.shiftKey ? -1 : 1);
      return;
    }
    if (ctrl && !e.altKey && e.key.toLowerCase() === 's') {
      e.preventDefault();
      if (isTypingTarget(document.activeElement) && this.hdrEl.contains(document.activeElement)) document.activeElement.blur();
      this.saveNow();
      return;
    }
    if (ctrl && e.key === 'Enter') {
      e.preventDefault();
      if (this.grid.editing && !this.grid.commitEdit()) return;
      this.toggleVerified();
    }
  }
}

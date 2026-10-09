// Anteprima Excel: Riepilogo, Dettaglio giornaliero (modificabile) e Anomalie in griglie stile
// foglio di calcolo, poi generazione del file .xlsx ed esportazioni precedenti.

import { downloadFile, get, post, put } from '../api.js';
import { SheetGrid } from '../grid.js';
import { ART, icon } from '../icons.js';
import { refreshState, saveSettings, store } from '../state.js';
import { emptyState, statusKey, statusLabel, toast, toastError } from '../ui.js';
import {
  $, capitalize, colLetter, debounce, esc, fmtBytes, fmtDateISO, fmtDateTime, fmtOre, fmtPeriodo, fmtSigned,
  hoursBetween, html, plural, raw, setHTML, storageGet, storageSet,
} from '../util.js';

const TABS = [
  { id: 'riepilogo', label: 'Riepilogo' },
  { id: 'dettaglio', label: 'Dettaglio giornaliero' },
  { id: 'anomalie', label: 'Anomalie' },
];
const DAY_FIELDS = ['prog_entrata', 'prog_uscita', 'eff_entrata', 'eff_uscita', 'ore_dichiarate', 'assenza_alunno', 'assenza_operatore', 'firma', 'note'];
const SEV_TONE = { errore: 'tone-err', attenzione: 'tone-warn', info: 'tone-info' };
const SEV_LABEL = { errore: 'Errore', attenzione: 'Attenzione', info: 'Informazione' };
const STATO_TONE = { ok: 'tone-ok', da_verificare: 'tone-warn', errori: 'tone-err', errore: 'tone-err', scartato: 'tone-neutral', in_coda: 'tone-neutral', in_lavorazione: 'tone-info' };

export function mount(root, ctx) {
  const v = new Preview(root, ctx);
  return {
    unmount: () => v.destroy(),
    onState: (kind) => v.onState(kind),
    beforeLeave: () => v.flush(),
    update: (params, query) => { v.setScope(query); return true; },
  };
}

class Preview {
  constructor(root, ctx) {
    this.root = root;
    this.ctx = ctx;
    this.tab = storageGet('sirio.xl.tab', 'riepilogo');
    this.data = null;
    this.grid = null;
    this.ids = null;
    this.onlyVerified = false;
    this.pending = new Map();      // doc_id -> Map(giorno -> patch)
    this.saving = null;
    this.result = null;
    this.exports = [];
    this.lastRevision = null;
    this.scheduleSave = debounce(() => this.flush(), 450);
    this.scheduleReload = debounce(() => {
      if (this.grid?.editing || this.pending.size || this.saving) { this.scheduleReload(); return; }
      this.load();
    }, 500);
    this.build();
    this.setScope(ctx.query);
    this.loadExports();
  }

  destroy() {
    this.scheduleSave.cancel();
    this.scheduleReload.cancel();
    this.grid?.destroy();
  }

  get options() {
    const s = store.settings?.settings || {};
    return { fogli_per_documento: s.export_fogli_per_documento ?? true, giorni_vuoti: s.export_giorni_vuoti ?? false };
  }

  build() {
    this.root.innerHTML = `
      <div class="xl">
        <section class="card xl-main" aria-label="Anteprima del file Excel">
          <div class="xl-head">
            <div class="xl-file"><span class="xl-file-icon">${icon('excel')}</span>
              <div style="min-width:0"><div class="card-title">Rendicontazione · anteprima</div><div class="card-sub" id="xl-sub">Caricamento…</div></div></div>
            <div id="xl-scope" style="margin-left:auto"></div>
          </div>
          <div class="formula-bar" aria-live="polite"><div class="formula-ref" id="xl-ref">—</div><div class="formula-fx">fx</div><div class="formula-val" id="xl-val"></div></div>
          <div id="xl-grid"></div>
          <div class="xl-tabs" role="tablist" id="xl-tabs"></div>
        </section>
        <aside class="xl-side">
          <div id="xl-result"></div>
          <section class="card" aria-labelledby="xl-gen-title">
            <div class="card-head"><div><div class="card-title" id="xl-gen-title">Genera Excel</div><div class="card-sub">File .xlsx con formule, pronto per la rendicontazione</div></div></div>
            <div class="card-body xl-options" id="xl-options"></div>
          </section>
          <section class="card" aria-labelledby="xl-prev-title">
            <div class="card-head"><div style="flex:1"><div class="card-title" id="xl-prev-title">Esportazioni precedenti</div></div>
              <button type="button" class="btn btn-sm btn-ghost" data-act="open-dir">${icon('folder-open')}Apri cartella</button></div>
            <div class="exports-list" id="xl-exports"></div>
          </section>
        </aside>
      </div>`;
    this.root.addEventListener('click', (e) => this.onClick(e));
    this.root.addEventListener('change', (e) => this.onChange(e));
  }

  setScope(query) {
    const ids = (query?.get?.('ids') || '').split(',').map((s) => s.trim()).filter(Boolean);
    this.ids = ids.length ? ids : null;
    this.load();
  }

  scopeIds() {
    if (this.ids) return this.ids;
    if (this.onlyVerified) {
      return store.documents.filter((d) => d.status === 'completato' && d.is_foglio_firma && d.user_verified).map((d) => d.id);
    }
    return null;
  }

  async load() {
    const ids = this.scopeIds();
    const params = new URLSearchParams();
    if (ids) params.set('ids', ids.join(',') || '-');
    if (this.options.giorni_vuoti) params.set('giorni_vuoti', 'true');
    this.lastRevision = store.revision;
    let data;
    try {
      data = await get(`/api/preview?${params}`);
      if (ids && !ids.length) { data.documenti = []; data.dettaglio = []; data.anomalie = []; }
    } catch (err) {
      toastError(err, 'Anteprima non disponibile');
      data = this.data || { documenti: [], dettaglio: [], anomalie: [], kpi: {} };
    }
    if (this.grid?.editing) {
      // si aggiorna al termine della modifica in corso
      this.deferred = data;
      return;
    }
    this.data = data;
    this.renderAll();
  }

  applyDeferred() {
    if (!this.deferred) return;
    this.data = this.deferred;
    this.deferred = null;
    this.renderAll();
  }

  onState(kind) {
    if (kind === 'settings') { this.renderOptions(); return; }
    if (kind !== 'state') return;
    if (this.onlyVerified || this.ids) this.renderOptions();
    // aggiorna l'anteprima se i documenti sono cambiati (non durante le modifiche)
    if (store.revision !== this.lastRevision && !this.pending.size && !this.saving && !this.grid?.editing) this.scheduleReload();
  }

  renderAll() {
    const d = this.data;
    const nDocs = d.documenti.length;
    $('#xl-sub', this.root).textContent = nDocs
      ? `${plural(nDocs, 'foglio firma', 'fogli firma')} · ${plural(d.dettaglio.length, 'riga di dettaglio', 'righe di dettaglio')} · ${plural(d.anomalie.length, 'anomalia', 'anomalie')}`
      : 'Nessun foglio firma da esportare';
    const scope = $('#xl-scope', this.root);
    if (this.ids) {
      setHTML(scope, html`<span class="xl-scope">${icon('filter', 'icon-sm')}Limitata a ${plural(this.ids.length, 'documento selezionato', 'documenti selezionati')}<button type="button" class="link-btn" data-act="all">Mostra tutti</button></span>`);
    } else scope.innerHTML = '';
    this.renderTabs();
    this.renderGrid();
    this.renderOptions();
  }

  renderTabs() {
    const d = this.data;
    const counts = { riepilogo: d.documenti.length, dettaglio: d.dettaglio.length, anomalie: d.anomalie.length };
    setHTML($('#xl-tabs', this.root), html`${TABS.map((t) => html`<button type="button" class="xl-tab" role="tab" data-tab="${t.id}" aria-selected="${String(this.tab === t.id)}">${t.id === 'riepilogo' ? icon('sheet', 'icon-sm') : ''}${t.label}<span class="count">${counts[t.id]}</span></button>`)}
      <div class="xl-status" id="xl-status"></div>`);
  }

  /* ============================================================ griglie */
  renderGrid(force = false) {
    const host = $('#xl-grid', this.root);
    const d = this.data;
    if (d.documenti.length && this.grid && this.gridTab === this.tab && !force) {
      // stessi fogli e stessa scheda: aggiorna i dati mantenendo selezione e scorrimento
      const cfg = this.tab === 'dettaglio' ? this.detailConfig() : this.tab === 'anomalie' ? this.anomConfig() : this.summaryConfig();
      Object.assign(this.grid.opts, cfg);
      this.grid.setRows(cfg.rows);
      this.updateFormula();
      this.renderStatus();
      return;
    }
    this.grid?.destroy();
    this.grid = null;
    this.gridTab = null;
    host.className = '';
    host.innerHTML = '';
    if (!d.documenti.length) {
      setHTML(host, html`<div style="height:100%;display:grid;place-items:center">${emptyState({
        art: ART.sheet,
        title: 'Nessun foglio firma pronto per l\'Excel',
        text: this.onlyVerified
          ? 'Nessun documento è stato ancora confermato: disattiva «Solo documenti confermati» oppure conferma i documenti nella revisione.'
          : 'L\'anteprima mostra i fogli firma letti completamente. Importa i PDF e attendi il termine della lettura.',
        action: html`<a class="btn btn-primary" href="#/dashboard">${icon('upload')}Vai all'importazione</a>`,
      })}</div>`);
      $('#xl-ref', this.root).textContent = '—';
      $('#xl-val', this.root).textContent = '';
      this.renderStatus();
      return;
    }
    const cfg = this.tab === 'dettaglio' ? this.detailConfig() : this.tab === 'anomalie' ? this.anomConfig() : this.summaryConfig();
    this.grid = new SheetGrid(host, {
      excel: true,
      zebra: true,
      rowHeight: 30,
      label: TABS.find((t) => t.id === this.tab)?.label,
      onActiveChange: () => this.updateFormula(),
      onSaveShortcut: () => this.flush(),
      onEditEnd: () => setTimeout(() => this.applyDeferred(), 0),
      ...cfg,
    });
    this.gridTab = this.tab;
    if (this.activeKey && this.activeKey.tab === this.tab) {
      this.grid.selectByKey(this.activeKey.row, this.activeKey.col, { scroll: true });
    } else this.grid.select(0, 1, { scroll: false });
    this.updateFormula();
    this.renderStatus();
  }

  updateFormula() {
    const g = this.grid;
    if (!g?.active) return;
    const { r, c } = g.active;
    const col = g.cols[c];
    const row = g.rows[r];
    $('#xl-ref', this.root).textContent = `${colLetter(c - 1)}${r + 2}`;
    let text = '';
    if (col.formula) text = col.formula(row);
    else {
      const v = g.valueOf(row, col);
      text = g.copyText(row, col);
      if (col.type === 'bool') text = v ? 'Sì' : 'No';
    }
    $('#xl-val', this.root).textContent = text;
    this.activeKey = { tab: this.tab, row: g.rowKey(row, r), col: col.key };
    this.renderStatus();
  }

  renderStatus() {
    const el = $('#xl-status', this.root);
    if (!el) return;
    const d = this.data;
    if (this.tab === 'riepilogo') {
      const ore = d.documenti.reduce((s, x) => s + (x.totals?.ore_riconosciute || 0), 0);
      const ver = d.documenti.filter((x) => x.user_verified).length;
      setHTML(el, html`<span>Fogli: <b>${d.documenti.length}</b></span><span>Confermati: <b>${ver}</b></span><span>Ore riconosciute: <b>${fmtOre(ore) || '0'}</b></span>`);
    } else if (this.tab === 'dettaglio') {
      const ore = d.dettaglio.reduce((s, x) => s + (x.ore_riconosciute || 0), 0);
      const ill = d.dettaglio.reduce((s, x) => s + Object.values(x.stati || {}).filter((v) => v === 'illeggibile').length, 0);
      setHTML(el, html`<span>Righe: <b>${d.dettaglio.length}</b></span>${ill ? html`<span style="color:var(--illeg)">Illeggibili: <b style="color:inherit">${ill}</b></span>` : ''}<span>Ore riconosciute: <b>${fmtOre(ore) || '0'}</b></span>`);
    } else {
      const c = (s) => d.anomalie.filter((a) => a.gravita === s).length;
      setHTML(el, html`<span>Errori: <b>${c('errore')}</b></span><span>Attenzioni: <b>${c('attenzione')}</b></span><span>Informazioni: <b>${c('info')}</b></span>`);
    }
  }

  summaryConfig() {
    const docs = this.data.documenti;
    const rows = docs.map((x, i) => ({ ...x, __i: i }));
    const sum = (f) => docs.reduce((s, x) => s + (Number(f(x)) || 0), 0);
    const total = {
      __total: true, id: '__total',
      header: { operatore: 'TOTALE' },
      totals: {
        giorni_lavorati: sum((x) => x.totals?.giorni_lavorati),
        ore_dichiarate: sum((x) => x.totals?.ore_dichiarate),
        ore_calcolate: sum((x) => x.totals?.ore_calcolate),
        ore_riconosciute: sum((x) => x.totals?.ore_riconosciute),
        totale_mensile_dichiarato: sum((x) => x.totals?.totale_mensile_dichiarato),
        differenza_totale: sum((x) => x.totals?.differenza_totale),
        n_errori: sum((x) => x.totals?.n_errori),
        n_attenzioni: sum((x) => x.totals?.n_attenzioni),
        campi_illeggibili: sum((x) => x.totals?.campi_illeggibili),
        campi_incerti: sum((x) => x.totals?.campi_incerti),
      },
    };
    if (rows.length > 1) rows.push(total);
    const T = (k) => (row) => row.totals?.[k];
    const H = (k) => (row) => row.header?.[k];
    const num = (k, label, width, tip) => ({ key: k, label, width, type: 'calc', tip, cls: 'num', get: T(k) });
    const count = (k, label, width, cls) => ({
      key: k, label, width, type: 'ro', align: 'r', get: T(k),
      render: (row) => { const v = row.totals?.[k] || 0; return v ? `<span class="counter ${cls}">${v}</span>` : '<span class="muted">0</span>'; },
    });
    const columns = [
      { key: 'operatore', label: 'Operatore', width: 170, type: 'ro', get: (row) => row.header?.operatore || (row.__total ? '' : row.display_name), render: (row) => (row.__total ? '<b>TOTALE</b>' : esc(row.header?.operatore || row.display_name)) },
      { key: 'alunno', label: 'Alunno', width: 150, type: 'ro', get: H('alunno') },
      { key: 'istituto', label: 'Istituto', width: 150, type: 'ro', get: H('istituto') },
      { key: 'periodo', label: 'Mese', width: 128, type: 'ro', get: (row) => (row.__total ? '' : capitalize(fmtPeriodo(row.header?.mese, row.header?.anno, ''))) },
      { key: 'ore_pei', label: 'Ore PEI', width: 70, type: 'calc', get: H('ore_pei'), tip: 'Ore settimanali previste dal PEI' },
      { key: 'giorni_lavorati', label: 'Giorni lav.', width: 82, type: 'ro', align: 'r', get: T('giorni_lavorati') },
      num('ore_dichiarate', 'Ore dich.', 82, 'Somma della colonna «Tot. ore effettive»'),
      num('ore_calcolate', 'Ore calc.', 82, 'Somma delle ore calcolate dagli orari effettivi'),
      { ...num('ore_riconosciute', 'Ore ric.', 82, 'Ore riconosciute: dichiarate, oppure calcolate se mancanti'), render: (row) => `<b>${esc(fmtOre(row.totals?.ore_riconosciute) || '0')}</b>` },
      num('totale_mensile_dichiarato', 'Tot. mensile', 92, 'Totale ore effettive mensili scritto sul foglio'),
      {
        key: 'differenza_totale', label: 'Differenza', width: 88, type: 'calc', get: T('differenza_totale'),
        render: (row) => { const v = row.totals?.differenza_totale; if (v === null || v === undefined) return ''; return Math.abs(v) > 0.01 ? `<span class="is-bad-value" style="color:var(--danger-text);font-weight:650">${esc(fmtSigned(v))}</span>` : '0'; },
      },
      count('n_errori', 'Errori', 64, 'is-err'),
      count('n_attenzioni', 'Attenz.', 64, 'is-uncert'),
      count('campi_illeggibili', 'Illegg.', 64, 'is-illeg'),
      count('campi_incerti', 'Incerti', 64, 'is-uncert'),
      {
        key: 'stato', label: 'Stato', width: 116, type: 'ro', get: (row) => (row.__total ? '' : statusLabel(statusKey(row))),
        render: (row) => (row.__total ? '' : `<span class="esito ${STATO_TONE[statusKey(row)] || 'tone-neutral'}">${esc(statusLabel(statusKey(row)))}</span>`),
      },
      {
        key: 'verificato', label: 'Confermato', width: 92, type: 'ro', align: 'c', get: (row) => (row.__total ? '' : (row.user_verified ? 'Sì' : 'No')),
        render: (row) => (row.__total ? '' : (row.user_verified ? `<span class="verified-tick" style="width:18px;height:18px;box-shadow:none">${icon('check')}</span>` : '<span class="muted">No</span>')),
      },
      { key: 'file', label: 'File d\'origine', width: 200, type: 'ro', get: (row) => (row.__total ? '' : `${row.source_file}${row.page_count > 1 ? ` (pag. ${row.source_page})` : ''}`) },
    ];
    return {
      columns,
      rows,
      rowKey: (row) => row.id,
      readOnly: true,
      getValue: (row, col) => (col.get ? col.get(row) : row[col.key]),
      rowInfo: (row) => (row.__total ? { cls: 'is-total' } : null),
      onReadOnlyActivate: (row) => { if (!row.__total) this.ctx.navigate(`#/revisione/${encodeURIComponent(row.id)}`); },
    };
  }

  detailConfig() {
    const rows = this.data.dettaglio;
    const stato = (row, key) => row.stati?.[key] || '';
    const columns = [
      { key: 'operatore', label: 'Operatore', width: 150, type: 'ro' },
      { key: 'alunno', label: 'Alunno', width: 136, type: 'ro' },
      { key: 'istituto', label: 'Istituto', width: 130, type: 'ro' },
      { key: 'data', label: 'Data', width: 100, type: 'ro', align: 'c', get: (row) => fmtDateISO(row.data) || `g. ${row.giorno}` },
      { key: 'giorno_settimana', label: 'Gg', width: 48, type: 'ro', align: 'c', cls: 'is-tight', render: (row) => `<span class="gg${row.tipo_giorno && row.tipo_giorno !== 'feriale' ? ' is-we' : ''}">${esc(row.giorno_settimana || '—')}</span>${row.tipo_giorno === 'festivo' ? '<span class="gg-fest"></span>' : ''}` },
      { key: 'prog_entrata', label: 'Prog. entrata', width: 92, type: 'time' },
      { key: 'prog_uscita', label: 'Prog. uscita', width: 88, type: 'time' },
      { key: 'eff_entrata', label: 'Eff. entrata', width: 86, type: 'time' },
      { key: 'eff_uscita', label: 'Eff. uscita', width: 84, type: 'time' },
      { key: 'ore_programmate', label: 'Ore progr.', width: 78, type: 'calc' },
      { key: 'ore_calcolate', label: 'Ore calc.', width: 76, type: 'calc' },
      { key: 'ore_dichiarate', label: 'Ore dich.', width: 76, type: 'hours' },
      {
        key: 'differenza', label: 'Differenza', width: 84, type: 'calc',
        render: (row) => { const v = row.differenza; if (v === null || v === undefined) return ''; return Math.abs(v) > 0.01 ? `<span style="color:var(--danger-text);font-weight:650">${esc(fmtSigned(v))}</span>` : '0'; },
      },
      { key: 'assenza_alunno', label: 'Ass. alunno', width: 84, type: 'bool', variant: 'cross' },
      { key: 'assenza_operatore', label: 'Ass. operat.', width: 88, type: 'bool', variant: 'cross' },
      { key: 'firma', label: 'Firma', width: 60, type: 'bool', variant: 'sign' },
      { key: 'note', label: 'Note', width: 150, type: 'text', maxLength: 500 },
      {
        key: 'esito', label: 'Esito', width: 200, type: 'ro',
        render: (row) => {
          const t = row.esito || '';
          if (!t) return '';
          const tone = t.startsWith('Errore') ? 'tone-err' : t.startsWith('Da verificare') ? 'tone-warn' : t === 'OK' ? 'tone-ok' : (t.startsWith('Assenza') ? 'tone-info' : 'tone-neutral');
          return `<span class="esito ${tone}" data-tip="${esc(t)}">${esc(t)}</span>`;
        },
      },
      { key: 'anomalie', label: 'Anomalie', width: 340, type: 'ro', cellTip: (row) => row.anomalie || '' },
    ];
    return {
      columns,
      rows,
      rowKey: (row) => `${row.doc_id}:${row.giorno}`,
      getValue: (row, col) => {
        if (col.get) return col.get(row);
        if ((col.key === 'eff_entrata' || col.key === 'eff_uscita') && !row[col.key] && row.trattino_effettivo) return '-';
        return row[col.key] ?? null;
      },
      cellState: (row, col) => {
        if (!DAY_FIELDS.includes(col.key)) return null;
        const st = stato(row, col.key);
        if (st === 'illeggibile') return { state: st, tone: 'illeg', tipTitle: 'Illeggibile', tip: 'Inserire il valore leggendo la scansione (doppio clic sulla riga «Esito» per aprire la revisione).' };
        if (st === 'incerto') return { state: st, tone: 'uncert', tipTitle: 'Lettura incerta', tip: 'Verificare il valore sulla scansione; Invio due volte conferma il valore.' };
        if (st === 'corretto') return { state: st, tone: 'edited', tipTitle: 'Modificato a mano', tip: 'Valore corretto rispetto alla lettura OCR.' };
        return null;
      },
      rowInfo: (row) => {
        let cls = '';
        if (row.tipo_giorno === 'sabato' || row.tipo_giorno === 'domenica') cls = 'is-weekend';
        else if (row.tipo_giorno === 'festivo') cls = 'is-holiday';
        else if (row.tipo_giorno === 'inesistente') cls = 'is-missing';
        if ((row.esito || '').startsWith('Errore')) cls += ' mark-errore';
        else if ((row.esito || '').startsWith('Da verificare')) cls += ' mark-attenzione';
        return { cls };
      },
      canEdit: (row) => {
        const d = store.documents.find((x) => x.id === row.doc_id);
        return !d || (d.status !== 'in_coda' && d.status !== 'in_lavorazione');
      },
      commit: (changes) => this.onDetailCommit(changes),
      onReadOnlyActivate: (row, col) => {
        this.ctx.navigate(`#/revisione/${encodeURIComponent(row.doc_id)}?g=${row.giorno}&c=${DAY_FIELDS.includes(col.key) ? col.key : 'eff_entrata'}`);
      },
    };
  }

  anomConfig() {
    const rows = this.data.anomalie.map((a, i) => ({ ...a, __i: i }));
    const columns = [
      { key: 'documento', label: 'Documento', width: 190, type: 'ro' },
      { key: 'giorno', label: 'Giorno', width: 64, type: 'ro', align: 'c', get: (row) => row.giorno ?? '' },
      {
        key: 'gravita', label: 'Gravità', width: 112, type: 'ro', get: (row) => SEV_LABEL[row.gravita] || row.gravita,
        render: (row) => `<span class="esito ${SEV_TONE[row.gravita] || 'tone-neutral'}">${esc(SEV_LABEL[row.gravita] || row.gravita)}</span>`,
      },
      { key: 'codice', label: 'Codice', width: 230, type: 'ro', render: (row) => `<span class="mono">${esc(row.codice)}</span>` },
      { key: 'messaggio', label: 'Descrizione', width: 460, type: 'ro', cellTip: (row) => row.messaggio },
      { key: 'campo', label: 'Campo', width: 130, type: 'ro' },
      { key: 'valore_letto', label: 'Valore letto', width: 100, type: 'ro' },
      { key: 'valore_atteso', label: 'Valore atteso', width: 104, type: 'ro' },
    ];
    return {
      columns,
      rows,
      rowKey: (row) => String(row.__i),
      readOnly: true,
      getValue: (row, col) => (col.get ? col.get(row) : row[col.key] ?? null),
      rowInfo: (row) => ({ cls: row.gravita === 'errore' ? 'mark-errore' : row.gravita === 'attenzione' ? 'mark-attenzione' : '' }),
      onReadOnlyActivate: (row) => {
        const q = row.giorno ? `?g=${row.giorno}${row.campo ? `&c=${encodeURIComponent(row.campo)}` : ''}` : (row.campo ? `?c=${encodeURIComponent(row.campo)}` : '');
        this.ctx.navigate(`#/revisione/${encodeURIComponent(row.doc_id)}${q}`);
      },
    };
  }

  /* ============================================================ modifica del dettaglio */
  onDetailCommit(changes) {
    for (const ch of changes) {
      const row = ch.row;
      const key = ch.col.key;
      const byDoc = this.pending.get(row.doc_id) || new Map();
      const patch = byDoc.get(row.giorno) || { giorno: row.giorno };
      row.stati = { ...(row.stati || {}) };
      if (ch.confirm) {
        delete row.stati[key];
        patch.incerti = Object.keys(row.stati).filter((k) => row.stati[k] === 'incerto');
        patch.illeggibili = Object.keys(row.stati).filter((k) => row.stati[k] === 'illeggibile');
      } else {
        patch[key] = ch.value;
        if ((key === 'eff_entrata' || key === 'eff_uscita') && ch.value === '-') { row[key] = null; row.trattino_effettivo = true; }
        else {
          row[key] = ch.value;
          if ((key === 'eff_entrata' || key === 'eff_uscita') && ch.value === null && ch.old === '-') {
            patch.trattino_effettivo = false;
            row.trattino_effettivo = false;
          }
        }
        row.stati[key] = 'corretto';
        row.ore_calcolate = hoursBetween(row.eff_entrata, row.eff_uscita);
        row.ore_programmate = hoursBetween(row.prog_entrata, row.prog_uscita);
        row.differenza = row.ore_dichiarate !== null && row.ore_dichiarate !== undefined && row.ore_calcolate !== null
          ? Math.round((row.ore_dichiarate - row.ore_calcolate) * 100) / 100 : null;
      }
      byDoc.set(row.giorno, patch);
      this.pending.set(row.doc_id, byDoc);
    }
    this.updateFormula();
    this.scheduleSave();
  }

  async flush() {
    this.scheduleSave.cancel();
    if (this.saving) await this.saving.catch(() => {});
    if (!this.pending.size) return;
    const work = this.pending;
    this.pending = new Map();
    const run = (async () => {
      let saved = 0;
      for (const [docId, byDay] of work) {
        try {
          await put(`/api/documents/${encodeURIComponent(docId)}`, { rows: Array.from(byDay.values()) });
          saved += byDay.size;
        } catch (err) {
          toastError(err, 'Modifica non salvata');
        }
      }
      return saved;
    })();
    this.saving = run;
    try {
      const saved = await run;
      if (saved) toast({ type: 'success', title: 'Modifica salvata nel documento', message: 'Esito, anomalie e totali sono stati ricalcolati.', duration: 2200 });
    } finally {
      this.saving = null;
    }
    refreshState().catch(() => {});
    if (!this.pending.size) await this.load();
  }

  /* ============================================================ generazione */
  renderOptions() {
    const o = this.options;
    const docs = this.data?.documenti || [];
    const notVerified = docs.filter((d) => !d.user_verified).length;
    const withErr = docs.filter((d) => d.totals?.stato === 'errori').length;
    const illeg = docs.reduce((s, d) => s + (d.totals?.campi_illeggibili || 0), 0);
    const busy = !!this.generating;
    const notes = [];
    if (illeg) notes.push(html`<li><span style="color:var(--illeg);font-weight:650">${plural(illeg, 'campo illeggibile', 'campi illeggibili')}</span> verranno evidenziati nel file</li>`);
    if (withErr) notes.push(html`<li>${plural(withErr, 'foglio con errori', 'fogli con errori')} di coerenza</li>`);
    if (notVerified) notes.push(html`<li>${plural(notVerified, 'foglio non ancora confermato', 'fogli non ancora confermati')}</li>`);
    setHTML($('#xl-options', this.root), html`
      <label class="opt"><span><span class="opt-title">Una scheda per ogni foglio firma</span><br><span class="opt-text">Replica del modulo con le celle evidenziate</span></span>
        <span class="switch"><input type="checkbox" data-opt="fogli_per_documento" ${o.fogli_per_documento ? raw('checked') : ''}><span class="switch-track"></span></span></label>
      <label class="opt"><span><span class="opt-title">Includi i giorni senza dati</span><br><span class="opt-text">Anche weekend e giorni vuoti nel dettaglio</span></span>
        <span class="switch"><input type="checkbox" data-opt="giorni_vuoti" ${o.giorni_vuoti ? raw('checked') : ''}><span class="switch-track"></span></span></label>
      ${this.ids ? '' : html`<label class="opt"><span><span class="opt-title">Solo documenti confermati</span><br><span class="opt-text">Esclude i fogli non ancora verificati</span></span>
        <span class="switch"><input type="checkbox" data-opt="solo_confermati" ${this.onlyVerified ? raw('checked') : ''}><span class="switch-track"></span></span></label>`}
      ${notes.length && docs.length ? html`<div class="banner is-info" style="padding:10px 12px;align-items:flex-start"><div style="font-size:12.5px;color:var(--text-2)"><b style="color:var(--text)">Prima di generare</b><ul style="margin:4px 0 0;padding-left:16px">${notes}</ul></div></div>` : ''}
      <button type="button" class="btn btn-primary btn-lg" data-act="generate" style="width:100%" ${!docs.length || busy ? raw('disabled') : ''}>
        ${busy ? html`<span class="spinner"></span>Generazione in corso…` : html`${icon('excel')}Genera Excel`}</button>
      <div class="muted" style="font-size:12px;text-align:center">${docs.length ? `${plural(docs.length, 'foglio firma', 'fogli firma')} nel file` : 'Nessun foglio firma da esportare'}</div>`);
  }

  async generate() {
    if (this.generating) return;
    await this.flush();
    const docs = this.data?.documenti || [];
    if (!docs.length) return;
    this.generating = true;
    this.renderOptions();
    const body = { options: { ...this.options } };
    const ids = this.scopeIds();
    if (ids) body.ids = docs.map((d) => d.id);
    try {
      const res = await post('/api/export', body);
      this.result = res;
      this.renderResult();
      $('#xl-result .export-done', this.root)?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      toast({ type: 'success', title: 'Excel generato', message: res.filename });
      this.loadExports();
    } catch (err) {
      toastError(err, 'Generazione dell\'Excel non riuscita');
    } finally {
      this.generating = false;
      this.renderOptions();
    }
  }

  renderResult() {
    const r = this.result;
    const el = $('#xl-result', this.root);
    if (!r) { el.innerHTML = ''; return; }
    setHTML(el, html`<section class="card export-done" aria-live="polite" tabindex="-1">
      <div class="export-done-head">
        <span class="export-done-icon">${icon('check')}</span>
        <div style="min-width:0"><div class="export-done-name">${r.filename}</div>
          <div class="export-done-meta">${fmtBytes(r.size)} · ${plural(r.documenti ?? (this.data?.documenti.length || 0), 'foglio firma', 'fogli firma')} · ${fmtDateTime(r.created_at)}</div></div>
      </div>
      <div class="export-done-actions">
        <button type="button" class="btn btn-success" data-act="open-file" data-file="${r.filename}">${icon('external')}Apri il file</button>
        <button type="button" class="btn" data-act="open-dir">${icon('folder-open')}Apri cartella</button>
        <button type="button" class="btn" data-act="download" data-file="${r.filename}" data-url="${r.url}">${icon('download')}Scarica</button>
      </div>
      <div class="export-path" title="${r.path}">${r.path}</div>
    </section>`);
  }

  async loadExports() {
    try {
      this.exports = await get('/api/exports');
    } catch (err) {
      this.exports = [];
      if (err.status !== 404) toastError(err, 'Elenco delle esportazioni non disponibile');
    }
    this.renderExports();
  }

  renderExports() {
    const el = $('#xl-exports', this.root);
    if (!el) return;
    if (!this.exports.length) {
      setHTML(el, html`<div class="muted" style="padding:4px 18px 18px;font-size:13px">Nessun file generato finora.</div>`);
      return;
    }
    setHTML(el, this.exports.slice(0, 8).map((x) => html`<div class="export-item">
      <span class="export-item-icon">${icon('excel')}</span>
      <div style="min-width:0"><div class="export-item-name" title="${x.filename}">${x.filename}</div><div class="export-item-meta">${fmtDateTime(x.created_at)} · ${fmtBytes(x.size)}</div></div>
      <div class="export-item-actions">
        <button type="button" class="btn btn-sm btn-ghost btn-icon" data-act="open-file" data-file="${x.filename}" aria-label="Apri ${x.filename}" data-tip="Apri">${icon('external')}</button>
        <button type="button" class="btn btn-sm btn-ghost btn-icon" data-act="download" data-file="${x.filename}" data-url="${x.url}" aria-label="Scarica ${x.filename}" data-tip="Scarica">${icon('download')}</button>
      </div></div>`));
  }

  /* ============================================================ eventi */
  async onClick(e) {
    const tab = e.target.closest('[data-tab]');
    if (tab) {
      if (this.grid?.editing && !this.grid.commitEdit()) return;
      this.tab = tab.dataset.tab;
      storageSet('sirio.xl.tab', this.tab);
      this.renderTabs();
      this.renderGrid(true);
      this.grid?.focus();
      return;
    }
    const b = e.target.closest('[data-act]');
    if (!b) return;
    const act = b.dataset.act;
    if (act === 'all') { this.ctx.navigate('#/anteprima'); return; }
    if (act === 'generate') { this.generate(); return; }
    if (act === 'open-dir') {
      try { await post('/api/open', { target: 'export_dir' }); toast({ type: 'info', title: 'Cartella delle esportazioni aperta', duration: 2500 }); } catch (err) { toastError(err, 'Impossibile aprire la cartella'); }
      return;
    }
    if (act === 'open-file') {
      try { await post('/api/open', { target: 'file', filename: b.dataset.file }); toast({ type: 'info', title: 'Apertura del file in corso', message: b.dataset.file, duration: 2500 }); } catch (err) { toastError(err, 'Impossibile aprire il file'); }
      return;
    }
    if (act === 'download') {
      try { await downloadFile(b.dataset.url, b.dataset.file); } catch (err) { toastError(err, 'Download non riuscito'); }
    }
  }

  async onChange(e) {
    const inp = e.target.closest('[data-opt]');
    if (!inp) return;
    const opt = inp.dataset.opt;
    if (opt === 'solo_confermati') {
      this.onlyVerified = inp.checked;
      await this.load();
      return;
    }
    const key = opt === 'fogli_per_documento' ? 'export_fogli_per_documento' : 'export_giorni_vuoti';
    try {
      await saveSettings({ [key]: inp.checked });
      if (opt === 'giorni_vuoti') await this.load();
    } catch (err) {
      inp.checked = !inp.checked;
      toastError(err, 'Impostazione non salvata');
    }
  }
}

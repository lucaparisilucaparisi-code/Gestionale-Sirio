// Documenti: elenco filtrabile e ordinabile, selezione multipla e azioni di gruppo.

import { authUrl, del, post } from '../api.js';
import { ART, icon } from '../icons.js';
import { pokePolling, refreshState, store } from '../state.js';
import { confirmDialog, emptyState, showMenu, skeletonRows, statusBadge, statusKey, toast, toastError } from '../ui.js';
import { pickFiles } from '../upload.js';
import { $, $$, debounce, esc, fmtOre, fmtPeriodo, html, normalizeText, plural, raw, setHTML, storageGet, storageSet } from '../util.js';

const FILTERS = [
  { id: 'tutti', label: 'Tutti', test: () => true },
  { id: 'da_verificare', label: 'Da verificare', test: (d) => isDone(d) && d.totals?.stato !== 'ok' && !d.user_verified },
  { id: 'errori', label: 'Con errori', test: (d) => (isDone(d) && d.totals?.stato === 'errori') || d.status === 'errore' },
  { id: 'illeggibili', label: 'Illeggibili', test: (d) => isDone(d) && (d.totals?.campi_illeggibili || 0) > 0 },
  { id: 'ok', label: 'OK', test: (d) => isDone(d) && d.totals?.stato === 'ok' },
  { id: 'verificati', label: 'Verificati', test: (d) => isDone(d) && d.user_verified },
  { id: 'in_lettura', label: 'In lettura', test: (d) => d.status === 'in_coda' || d.status === 'in_lavorazione' },
  { id: 'scartati', label: 'Scartati', test: (d) => d.status === 'scartato' || (d.status === 'completato' && d.is_foglio_firma === false) },
];
const STATUS_RANK = { errore: 0, errori: 1, da_verificare: 2, in_lavorazione: 3, in_coda: 4, ok: 5, scartato: 6 };
const SORTS = {
  operatore: (d) => normalizeText(d.header?.operatore || d.display_name),
  alunno: (d) => normalizeText(d.header?.alunno || ''),
  istituto: (d) => normalizeText(d.header?.istituto || ''),
  periodo: (d) => (d.header?.anno || 0) * 100 + (d.header?.mese || 0),
  ore: (d) => d.totals?.ore_riconosciute || 0,
  stato: (d) => STATUS_RANK[statusKey(d)] ?? 9,
  importato: (d) => d.created_at || '',
};

function isDone(d) { return d.status === 'completato' && d.is_foglio_firma !== false; }

/** Elenco degli id nell'ordine mostrato (per «precedente/successivo» nella revisione). */
export function reviewOrder() {
  const all = store.documents.filter((d) => d.status !== 'in_coda' && d.status !== 'in_lavorazione');
  const list = (store.navList || []).filter((id) => all.some((d) => d.id === id));
  return list.length ? list : all.map((d) => d.id);
}

export function mount(root, { navigate, query }) {
  const state = {
    filter: FILTERS.some((f) => f.id === query.get('stato')) ? query.get('stato') : 'tutti',
    q: query.get('q') || '',
    periodo: 'tutti',
    sort: storageGet('sirio.docs.sort', { key: 'importato', dir: 1 }),
    view: storageGet('sirio.docs.view', 'table'),
    selected: new Set(),
    lastVisible: [],
  };

  root.innerHTML = `
    <div class="docs">
      <div class="toolbar">
        <div class="input-wrap">${icon('search')}<input class="input" type="search" id="docs-q" placeholder="Filtra per nome, istituto, file…" aria-label="Filtra i documenti"></div>
        <div class="segmented" id="docs-filters" role="group" aria-label="Filtra per stato"></div>
        <select class="select select-sm" id="docs-period" aria-label="Filtra per periodo" style="width:auto;min-width:150px"></select>
        <span class="spacer"></span>
        <div class="segmented" role="group" aria-label="Modalità di visualizzazione">
          <button type="button" data-view="table" aria-label="Elenco" data-tip="Elenco">${icon('list', 'icon-sm')}</button>
          <button type="button" data-view="cards" aria-label="Schede" data-tip="Schede con anteprima">${icon('cards', 'icon-sm')}</button>
        </div>
      </div>
      <div id="docs-bulk"></div>
      <div id="docs-body"></div>
    </div>`;

  const qInput = $('#docs-q', root);
  qInput.value = state.q;
  const applyQ = debounce(() => { state.q = qInput.value.trim(); render(); }, 120);
  qInput.addEventListener('input', applyQ);

  $('#docs-period', root).addEventListener('change', (e) => { state.periodo = e.target.value; render(); });

  root.addEventListener('click', async (e) => {
    const f = e.target.closest('[data-filter]');
    if (f) { state.filter = f.dataset.filter; render(); return; }
    const v = e.target.closest('[data-view]');
    if (v) { state.view = v.dataset.view; storageSet('sirio.docs.view', state.view); render(); return; }
    const s = e.target.closest('[data-sort]');
    if (s) {
      const key = s.dataset.sort;
      state.sort = state.sort.key === key ? { key, dir: -state.sort.dir } : { key, dir: key === 'ore' ? -1 : 1 };
      storageSet('sirio.docs.sort', state.sort);
      render();
      return;
    }
    const act = e.target.closest('[data-act]');
    if (act) { e.stopPropagation(); await handleAction(act.dataset.act, act); return; }
    if (e.target.closest('.check, [data-check]')) return;
    const row = e.target.closest('[data-id]');
    if (row) navigate(`#/revisione/${encodeURIComponent(row.dataset.id)}`);
  });

  root.addEventListener('change', (e) => {
    const cb = e.target.closest('[data-check]');
    if (!cb) return;
    const id = cb.dataset.check;
    if (id === '__all') {
      if (cb.checked) state.lastVisible.forEach((d) => state.selected.add(d.id));
      else state.selected.clear();
    } else if (cb.checked) state.selected.add(id);
    else state.selected.delete(id);
    render();
  });

  root.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target.matches('tr[data-id], .doc-card[data-id]')) {
      e.preventDefault();
      navigate(`#/revisione/${encodeURIComponent(e.target.dataset.id)}`);
    }
  });

  async function handleAction(act, el) {
    const ids = Array.from(state.selected);
    if (act === 'clear') { state.selected.clear(); render(); return; }
    if (act === 'import') { pickFiles(); return; }
    if (act === 'reset') { state.filter = 'tutti'; state.q = ''; qInput.value = ''; state.periodo = 'tutti'; render(); return; }
    if (act === 'menu') {
      const id = el.dataset.id;
      showMenu(el, [
        { label: 'Apri la revisione', icon: 'edit', onClick: () => navigate(`#/revisione/${encodeURIComponent(id)}`) },
        { label: 'Anteprima Excel del documento', icon: 'excel', onClick: () => navigate(`#/anteprima?ids=${encodeURIComponent(id)}`) },
        { label: 'Rielabora', icon: 'refresh', onClick: () => reprocess([id]) },
        'sep',
        { label: 'Elimina', icon: 'trash', danger: true, onClick: () => remove([id]) },
      ]);
      return;
    }
    if (!ids.length) return;
    if (act === 'reprocess') await reprocess(ids);
    else if (act === 'delete') await remove(ids);
    else if (act === 'export') navigate(`#/anteprima?ids=${ids.map(encodeURIComponent).join(',')}`);
  }

  async function reprocess(ids) {
    const edited = ids.map((id) => store.documents.find((d) => d.id === id)).filter((d) => d && (d.n_modifiche || d.user_verified));
    const ok = await confirmDialog({
      title: ids.length === 1 ? 'Rielaborare il documento?' : `Rielaborare ${ids.length} documenti?`,
      message: raw(`<p>La scansione verrà letta di nuovo con il motore attuale.</p>${edited.length
        ? `<p><b>${edited.length === 1 ? 'Un documento contiene' : `${edited.length} documenti contengono`} correzioni manuali o conferme</b> che verranno sostituite dalla nuova lettura.</p>` : ''}`),
      confirmLabel: 'Rielabora',
      tone: 'warning',
    });
    if (!ok) return;
    let failed = 0;
    for (const id of ids) {
      try { await post(`/api/documents/${encodeURIComponent(id)}/reprocess`); } catch { failed++; }
    }
    if (failed) toast({ type: 'error', title: 'Rielaborazione non riuscita', message: `${plural(failed, 'documento non è stato rimesso', 'documenti non sono stati rimessi')} in coda.` });
    else toast({ type: 'info', title: ids.length === 1 ? 'Documento rimesso in coda' : `${ids.length} documenti rimessi in coda`, duration: 3000 });
    state.selected.clear();
    pokePolling();
  }

  async function remove(ids) {
    const ok = await confirmDialog({
      title: ids.length === 1 ? 'Eliminare il documento?' : `Eliminare ${ids.length} documenti?`,
      message: 'I dati letti, le correzioni e l\'immagine della scansione verranno rimossi da Sirio OCR. I file originali e gli Excel già generati non vengono toccati.',
      confirmLabel: 'Elimina',
      danger: true,
    });
    if (!ok) return;
    let failed = 0;
    for (const id of ids) {
      try { await del(`/api/documents/${encodeURIComponent(id)}`); } catch (err) { if (err.status !== 404) failed++; }
      state.selected.delete(id);
    }
    if (failed) toast({ type: 'error', title: 'Eliminazione non completata', message: `${plural(failed, 'documento non è stato eliminato', 'documenti non sono stati eliminati')}.` });
    else toast({ type: 'success', title: ids.length === 1 ? 'Documento eliminato' : `${ids.length} documenti eliminati`, duration: 3000 });
    try { await refreshState(true); } catch (err) { toastError(err); }
  }

  function filtered() {
    const f = FILTERS.find((x) => x.id === state.filter) || FILTERS[0];
    const terms = normalizeText(state.q).split(/\s+/).filter(Boolean);
    let list = store.documents.filter((d) => f.test(d));
    if (state.periodo !== 'tutti') list = list.filter((d) => periodKey(d) === state.periodo);
    if (terms.length) {
      list = list.filter((d) => {
        const h = d.header || {};
        const hay = normalizeText([d.display_name, d.source_file, h.operatore, h.alunno, h.istituto, h.ente, fmtPeriodo(h.mese, h.anno, '')].join(' '));
        return terms.every((t) => hay.includes(t));
      });
    }
    const key = SORTS[state.sort.key] || SORTS.importato;
    const dir = state.sort.dir || 1;
    return list.map((d, i) => ({ d, i })).sort((a, b) => {
      const x = key(a.d); const y = key(b.d);
      if (x < y) return -dir;
      if (x > y) return dir;
      return a.i - b.i;
    }).map((x) => x.d);
  }

  function render() {
    // pulizia della selezione (documenti eliminati)
    for (const id of Array.from(state.selected)) if (!store.documents.some((d) => d.id === id)) state.selected.delete(id);
    renderFilters();
    renderPeriods();
    $$('[data-view]', root).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.view === state.view)));
    const list = filtered();
    state.lastVisible = list;
    store.navList = list.map((d) => d.id);
    renderBulk();
    const body = $('#docs-body', root);
    if (!store.loaded) {
      setHTML(body, html`<div class="card table-card"><table class="table"><tbody>${skeletonRows(6, ['18px', '36px', '60%', '50%', '50%', '40%', '30%', '40%'])}</tbody></table></div>`);
      return;
    }
    if (!store.documents.length) {
      setHTML(body, html`<div class="card">${emptyState({
        art: ART.documents,
        title: 'Nessun foglio firma importato',
        text: 'Importa i PDF scansionati dei fogli firma: Sirio OCR leggerà intestazione e tabella giornaliera di ogni pagina.',
        action: html`<button type="button" class="btn btn-primary" data-act="import">${icon('upload')}Importa PDF</button>`,
      })}</div>`);
      return;
    }
    if (!list.length) {
      setHTML(body, html`<div class="card">${emptyState({
        art: ART.documents,
        title: 'Nessun documento corrisponde ai filtri',
        text: 'Modifica la ricerca o lo stato selezionato per vedere altri documenti.',
        action: html`<button type="button" class="btn" data-act="reset">${icon('x')}Azzera i filtri</button>`,
      })}</div>`);
      return;
    }
    if (state.view === 'cards') renderCards(body, list);
    else renderTable(body, list);
  }

  function renderFilters() {
    const counts = Object.fromEntries(FILTERS.map((f) => [f.id, store.documents.filter((d) => f.test(d)).length]));
    setHTML($('#docs-filters', root), FILTERS
      .filter((f) => f.id === 'tutti' || counts[f.id] > 0 || f.id === state.filter || ['da_verificare', 'ok'].includes(f.id))
      .map((f) => html`<button type="button" data-filter="${f.id}" aria-pressed="${String(state.filter === f.id)}">${f.label}<span class="count">${counts[f.id]}</span></button>`));
  }

  function renderPeriods() {
    const sel = $('#docs-period', root);
    const keys = Array.from(new Set(store.documents.map(periodKey).filter(Boolean))).sort().reverse();
    if (state.periodo !== 'tutti' && !keys.includes(state.periodo)) state.periodo = 'tutti';
    const opts = [`<option value="tutti">Tutti i periodi</option>`, ...keys.map((k) => {
      const [a, m] = k.split('-').map(Number);
      return `<option value="${k}">${esc(capital(fmtPeriodo(m, a)))}</option>`;
    })].join('');
    if (sel.dataset.sig !== opts) { sel.innerHTML = opts; sel.dataset.sig = opts; }
    sel.value = state.periodo;
    sel.hidden = keys.length < 2;
  }

  function renderBulk() {
    const n = state.selected.size;
    const el = $('#docs-bulk', root);
    if (!n) { el.innerHTML = ''; return; }
    setHTML(el, html`<div class="bulkbar" role="region" aria-label="Azioni sui documenti selezionati">
      <span class="bulkbar-count">${plural(n, 'documento selezionato', 'documenti selezionati')}</span>
      <span class="spacer"></span>
      <button type="button" class="btn btn-sm" data-act="export">${icon('excel')}Esporta selezionati</button>
      <button type="button" class="btn btn-sm" data-act="reprocess">${icon('refresh')}Rielabora</button>
      <button type="button" class="btn btn-sm btn-danger-soft" data-act="delete">${icon('trash')}Elimina</button>
      <button type="button" class="btn btn-sm btn-icon" data-act="clear" aria-label="Annulla la selezione" data-tip="Annulla la selezione">${icon('x')}</button>
    </div>`);
  }

  function sortHead(key, label, cls = '') {
    const on = state.sort.key === key;
    const arrow = on && state.sort.dir < 0 ? 'arrow-down' : 'arrow-up';
    return html`<th class="is-sortable${on ? ' is-sorted' : ''} ${cls}" data-sort="${key}" aria-sort="${on ? (state.sort.dir > 0 ? 'ascending' : 'descending') : 'none'}">${label}<span class="sort-ind">${icon(arrow, 'icon-xs')}</span></th>`;
  }

  function renderTable(body, list) {
    const allOn = list.length > 0 && list.every((d) => state.selected.has(d.id));
    const someOn = list.some((d) => state.selected.has(d.id));
    setHTML(body, html`<div class="card table-card"><div class="table-wrap"><table class="table">
      <thead><tr>
        <th class="col-check"><label class="check" aria-label="Seleziona tutti"><input type="checkbox" data-check="__all" ${allOn ? raw('checked') : ''}><span class="check-box">${raw('<svg viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>')}</span></label></th>
        <th class="col-thumb"></th>
        ${sortHead('operatore', 'Operatore')}
        ${sortHead('alunno', 'Alunno')}
        ${sortHead('istituto', 'Istituto')}
        ${sortHead('periodo', 'Periodo')}
        ${sortHead('ore', 'Ore', 'ta-r')}
        <th>Segnalazioni</th>
        ${sortHead('stato', 'Stato')}
        <th class="ta-c">Verificato</th>
        <th style="width:52px"></th>
      </tr></thead>
      <tbody>${list.map((d) => tableRow(d))}</tbody>
    </table></div></div>`);
    const all = body.querySelector('[data-check="__all"]');
    if (all) all.indeterminate = someOn && !allOn;
  }

  function tableRow(d) {
    const h = d.header || {};
    const t = d.totals || {};
    const done = isDone(d);
    const sel = state.selected.has(d.id);
    return html`<tr data-id="${d.id}" tabindex="0" class="${sel ? 'is-selected' : ''}" aria-label="${d.display_name}">
      <td class="col-check"><label class="check" aria-label="Seleziona ${d.display_name}"><input type="checkbox" data-check="${d.id}" ${sel ? raw('checked') : ''}><span class="check-box">${raw('<svg viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>')}</span></label></td>
      <td class="col-thumb"><img class="thumb" src="${authUrl(d.thumb_url)}" alt="" loading="lazy" width="36" height="50"></td>
      <td><div class="doc-name">${h.operatore || (d.status === 'completato' ? 'Operatore non indicato' : d.display_name)}</div>
        <div class="doc-file" title="${d.source_file}">${d.source_file}${d.page_count > 1 ? ` · pag. ${d.source_page}/${d.page_count}` : ''}</div></td>
      <td><div class="cell-clip">${h.alunno || raw('<span class="muted">—</span>')}</div></td>
      <td><div class="cell-clip" title="${h.istituto || ''}">${h.istituto || raw('<span class="muted">—</span>')}</div></td>
      <td style="white-space:nowrap">${h.mese ? capital(fmtPeriodo(h.mese, h.anno)) : raw('<span class="muted">—</span>')}</td>
      <td class="ta-r ore-cell">${done ? fmtOre(t.ore_riconosciute || 0) || '0' : raw('<span class="muted">—</span>')}</td>
      <td>${done ? flags(t) : (d.status === 'errore' ? html`<span class="muted cell-clip" style="display:block;max-width:180px" data-tip="${d.error || d.status_message}">${d.error || d.status_message}</span>` : '')}</td>
      <td>${statusBadge(d)}</td>
      <td class="ta-c">${d.user_verified ? html`<span class="verified-tick" data-tip="Documento confermato">${icon('check')}</span>` : (done ? raw('<span class="verified-empty" aria-label="Non confermato"></span>') : '')}</td>
      <td><div class="row-actions"><button type="button" class="btn btn-sm btn-ghost btn-icon" data-act="menu" data-id="${d.id}" aria-label="Altre azioni">${icon('more')}</button></div></td>
    </tr>`;
  }

  function flags(t) {
    const items = [];
    items.push(html`<span class="counter ${t.n_errori ? 'is-err' : 'is-zero'}" data-tip="${plural(t.n_errori || 0, 'errore', 'errori')}">${icon('x-circle')}${t.n_errori || 0}</span>`);
    items.push(html`<span class="counter ${t.campi_illeggibili ? 'is-illeg' : 'is-zero'}" data-tip="${plural(t.campi_illeggibili || 0, 'campo illeggibile', 'campi illeggibili')}">${icon('eye-off')}${t.campi_illeggibili || 0}</span>`);
    items.push(html`<span class="counter ${t.campi_incerti ? 'is-uncert' : 'is-zero'}" data-tip="${plural(t.campi_incerti || 0, 'lettura incerta', 'letture incerte')}">${icon('help')}${t.campi_incerti || 0}</span>`);
    return html`<div class="flags">${items}</div>`;
  }

  function renderCards(body, list) {
    setHTML(body, html`<div class="doc-cards">${list.map((d) => {
      const h = d.header || {};
      const t = d.totals || {};
      const sel = state.selected.has(d.id);
      return html`<article class="card doc-card${sel ? ' is-selected' : ''}" data-id="${d.id}" tabindex="0" aria-label="${d.display_name}">
        <div class="doc-card-img"><img src="${authUrl(d.thumb_url)}" alt="" loading="lazy"></div>
        <label class="check doc-card-check" aria-label="Seleziona ${d.display_name}"><input type="checkbox" data-check="${d.id}" ${sel ? raw('checked') : ''}><span class="check-box">${raw('<svg viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>')}</span></label>
        <div class="doc-card-badge">${statusBadge(d, true)}</div>
        <div class="doc-card-body">
          <div class="row" style="gap:8px"><div class="doc-name" style="flex:1">${h.operatore || d.display_name}</div>${d.user_verified ? html`<span class="verified-tick" data-tip="Documento confermato">${icon('check')}</span>` : ''}</div>
          <div class="doc-file">${[h.alunno, h.mese ? capital(fmtPeriodo(h.mese, h.anno)) : ''].filter(Boolean).join(' · ') || d.source_file}</div>
          <div class="row" style="justify-content:space-between;margin-top:4px">
            <span class="ore-cell">${isDone(d) ? `${fmtOre(t.ore_riconosciute || 0) || '0'} ore` : ''}</span>
            ${isDone(d) ? flags(t) : ''}
          </div>
        </div>
      </article>`;
    })}</div>`);
  }

  render();
  return {
    onState: (kind) => { if (kind === 'state') render(); },
    update: (params, q) => {
      const st = q.get('stato');
      if (st && FILTERS.some((f) => f.id === st)) state.filter = st;
      if (q.has('q')) { state.q = q.get('q') || ''; qInput.value = state.q; }
      render();
      return true;
    },
  };
}

function periodKey(d) {
  const h = d.header || {};
  return h.mese && h.anno ? `${h.anno}-${String(h.mese).padStart(2, '0')}` : '';
}
function capital(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

// Dashboard: indicatori, area di importazione, coda di elaborazione e flusso di lavoro.

import { authUrl, post } from '../api.js';
import { ART, icon } from '../icons.js';
import { pokePolling, saveSettings, store } from '../state.js';
import { statusBadge, statusKey, toast, toastError } from '../ui.js';
import { filesFromDataTransfer, onUpload, pickFiles, pickFolder, upload, uploadItems, uploadSummaryText } from '../upload.js';
import { $, esc, fmtInt, fmtOre, fmtUSD, html, plural, raw, setHTML } from '../util.js';

export function mount(root, { navigate }) {
  root.innerHTML = `
    <div class="dash">
      <div id="dash-banner"></div>
      <div class="kpi-grid" id="kpis" aria-label="Indicatori"></div>
      <div class="dash-main">
        <div class="dropzone" id="dropzone" tabindex="0" role="button"
             aria-label="Importa fogli firma: trascina qui PDF o immagini, oppure premi Invio per sceglierli">
          <div class="dropzone-inner">
            ${ART.upload}
            <div class="dropzone-title">Trascina qui i fogli firma</div>
            <p class="dropzone-text">PDF scansionati (anche multipagina), foto o intere cartelle: ogni pagina diventa un documento da leggere e verificare.</p>
            <div class="dropzone-actions">
              <button type="button" class="btn btn-primary" data-act="files">${icon('upload')}Scegli file</button>
              <button type="button" class="btn" data-act="folder">${icon('folder-open')}Scegli cartella</button>
            </div>
            <div class="dropzone-formats">${['PDF', 'JPG', 'PNG', 'TIFF', 'BMP', 'WEBP'].map((f) => `<span class="format-chip">${f}</span>`).join('')}</div>
          </div>
          <div class="upload-status" id="upload-status" hidden></div>
        </div>
        <section class="card queue" aria-labelledby="queue-title">
          <div class="card-head">
            <div style="min-width:0;flex:1">
              <div class="card-title" id="queue-title">Coda di elaborazione</div>
              <div class="card-sub" id="queue-sub"></div>
            </div>
            <div class="queue-summary" id="queue-summary"></div>
          </div>
          <div class="queue-list" id="queue-list"></div>
        </section>
      </div>
      <section class="card steps" id="steps" aria-label="Flusso di lavoro"></section>
    </div>`;

  const dz = $('#dropzone', root);
  let depth = 0;
  dz.addEventListener('dragenter', (e) => { e.preventDefault(); depth++; dz.classList.add('is-dragover'); });
  dz.addEventListener('dragover', (e) => { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; });
  dz.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) dz.classList.remove('is-dragover'); });
  dz.addEventListener('drop', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    depth = 0;
    dz.classList.remove('is-dragover');
    document.body.classList.remove('is-dragging-files');
    const items = await filesFromDataTransfer(e.dataTransfer);
    if (items.length) uploadItems(items);
  });
  dz.addEventListener('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'folder') { pickFolder(); return; }
    if (e.target.closest('.upload-status')) return;
    pickFiles();
  });
  dz.addEventListener('keydown', (e) => {
    if ((e.key === 'Enter' || e.key === ' ') && e.target === dz) { e.preventDefault(); pickFiles(); }
  });

  root.addEventListener('click', async (e) => {
    const t = e.target.closest('[data-nav],[data-open],[data-reprocess],[data-engine]');
    if (!t) return;
    if (t.dataset.nav) navigate(t.dataset.nav);
    else if (t.dataset.reprocess) {
      e.stopPropagation();
      try {
        await post(`/api/documents/${encodeURIComponent(t.dataset.reprocess)}/reprocess`);
        toast({ type: 'info', title: 'Documento rimesso in coda', duration: 2500 });
        pokePolling();
      } catch (err) { toastError(err); }
    } else if (t.dataset.engine) {
      try {
        await saveSettings({ engine: t.dataset.engine });
        toast({ type: 'success', title: 'Motore locale attivato', message: 'I documenti in attesa vengono letti sul computer, senza costi.' });
        pokePolling();
      } catch (err) { toastError(err); }
    } else if (t.dataset.open) navigate(`#/revisione/${encodeURIComponent(t.dataset.open)}`);
  });

  root.addEventListener('keydown', (e) => {
    const item = e.target.closest?.('.queue-item[data-open]');
    if (item && e.target === item && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      navigate(`#/revisione/${encodeURIComponent(item.dataset.open)}`);
    }
  });

  const offUpload = onUpload(() => renderUpload(root));
  renderAll(root);
  renderUpload(root);
  return {
    onState: (kind) => { if (kind === 'state' || kind === 'settings') renderAll(root); },
    unmount: () => offUpload(),
  };
}

function renderAll(root) {
  renderBanner(root);
  renderKpis(root);
  renderQueue(root);
  renderSteps(root);
}

/* ------------------------------------------------------------ banner */
function renderBanner(root) {
  const el = $('#dash-banner', root);
  const q = store.queue;
  const s = store.settings;
  const waiting = q && !q.motore_pronto && q.in_coda > 0;
  if (!waiting) { el.innerHTML = ''; el.hidden = true; return; }
  el.hidden = false;
  const claude = (q.motore || s?.settings?.engine) === 'claude';
  if (claude && s && !s.api_key_set) {
    setHTML(el, html`<div class="banner" role="status">
      <div class="banner-icon">${icon('key')}</div>
      <div style="flex:1;min-width:0">
        <div class="banner-title">Configura la chiave API di Anthropic per avviare la lettura</div>
        <div class="banner-text">${plural(q.in_coda, 'documento è in attesa', 'documenti sono in attesa')}: Claude Vision richiede una chiave API. In alternativa puoi usare il motore locale, gratuito e senza invio di dati.</div>
      </div>
      <button type="button" class="btn" data-engine="locale">${icon('cpu')}Usa il motore locale</button>
      <button type="button" class="btn btn-primary" data-nav="#/impostazioni">${icon('key')}Configura la chiave</button>
    </div>`);
    return;
  }
  setHTML(el, html`<div class="banner" role="status">
    <div class="banner-icon">${icon('hourglass')}</div>
    <div style="flex:1;min-width:0">
      <div class="banner-title">Lettura in attesa</div>
      <div class="banner-text">${q.messaggio || 'Il motore di lettura non è ancora pronto.'}</div>
    </div>
    <button type="button" class="btn" data-nav="#/impostazioni">${icon('settings')}Apri impostazioni</button>
  </div>`);
}

/* ------------------------------------------------------------ indicatori */
function kpiCard({ tone, ic, label, value, unit = '', foot, nav, tip }) {
  const tag = nav ? 'button' : 'div';
  const attrs = (nav ? ` type="button" data-nav="${esc(nav)}"` : '') + (tip ? ` data-tip="${esc(tip)}"` : '');
  return html`<${raw(tag)} class="kpi ${tone}"${raw(attrs)}>
    <div class="kpi-head"><span class="kpi-icon">${icon(ic)}</span>${label}</div>
    <div class="kpi-value">${value}${unit ? html`<small>${unit}</small>` : ''}</div>
    <div class="kpi-foot">${foot}</div>
  </${raw(tag)}>`;
}

function renderKpis(root) {
  const k = store.kpi || {};
  const engine = store.queue?.motore || store.settings?.settings?.engine || 'locale';
  const inLettura = (k.in_coda || 0) + (k.in_lavorazione || 0);
  const cards = [
    kpiCard({
      tone: 'k-blue', ic: 'documents', label: 'Fogli firma', value: fmtInt(k.documenti || 0),
      foot: inLettura ? `${plural(k.completati || 0, 'letto', 'letti')} · ${fmtInt(inLettura)} in lettura` : `${plural(k.completati || 0, 'letto', 'letti')}${k.scartati ? ` · ${plural(k.scartati, 'scartato', 'scartati')}` : ''}`,
      nav: '#/documenti',
    }),
    kpiCard({
      tone: 'k-green', ic: 'clock', label: 'Ore riconosciute', value: fmtOre(k.ore_totali || 0) || '0', unit: 'ore',
      foot: k.completati ? `su ${plural(k.completati, 'foglio completato', 'fogli completati')}` : 'nessun foglio completato',
      tip: 'Per ogni giorno: ore dichiarate sul foglio, oppure calcolate dagli orari se mancanti.',
    }),
    kpiCard({
      tone: 'k-amber', ic: 'flag', label: 'Da verificare', value: fmtInt(k.da_verificare || 0),
      foot: `${fmtInt(k.verificati || 0)} confermati`, nav: '#/documenti?stato=da_verificare',
    }),
    kpiCard({
      tone: 'k-illeg', ic: 'eye-off', label: 'Campi illeggibili', value: fmtInt(k.illeggibili || 0),
      foot: `${plural(k.incerti || 0, 'lettura incerta', 'letture incerte')}`, nav: '#/documenti?stato=illeggibili',
    }),
    kpiCard({
      tone: 'k-red', ic: 'x-circle', label: 'Errori', value: fmtInt(k.errori || 0),
      foot: k.errori_elaborazione
        ? `${plural(k.errori_elaborazione, 'lettura non riuscita', 'letture non riuscite')}`
        : `in ${plural(k.documenti_con_errori || 0, 'foglio', 'fogli')}`,
      nav: '#/documenti?stato=errori',
    }),
    kpiCard({
      tone: 'k-slate', ic: 'coin', label: 'Costo stimato', value: fmtUSD(k.costo_usd || 0),
      foot: engine === 'claude' ? 'Letture con Claude Vision' : 'Motore locale gratuito',
      tip: 'Stima del costo delle letture con Claude Vision (prezzi pubblici di Anthropic). Il motore locale non ha costi.',
    }),
  ];
  setHTML($('#kpis', root), cards);
}

/* ------------------------------------------------------------ coda */
function renderQueue(root) {
  const docs = store.documents;
  const q = store.queue || {};
  const order = { in_lavorazione: 0, in_coda: 1, errore: 2 };
  const active = docs.filter((d) => d.status in order).sort((a, b) => order[a.status] - order[b.status]);
  const done = docs.filter((d) => !(d.status in order))
    .sort((a, b) => String(b.updated_at).localeCompare(String(a.updated_at)))
    .slice(0, Math.max(0, 40 - active.length));
  const list = [...active, ...done];

  const sub = $('#queue-sub', root);
  if (q.in_lavorazione || q.in_coda) sub.textContent = `${q.in_lavorazione || 0} in lettura · ${q.in_coda || 0} in coda`;
  else if (docs.length) sub.textContent = q.motore_pronto === false ? (q.messaggio || 'Motore non pronto') : 'Nessuna lettura in corso';
  else sub.textContent = 'In attesa di documenti';

  const summary = [];
  if (q.completati) summary.push(html`<span class="badge badge-sm tone-ok">${plural(q.completati, 'letto', 'letti')}</span>`);
  if (q.errori) summary.push(html`<span class="badge badge-sm tone-err">${plural(q.errori, 'errore', 'errori')}</span>`);
  setHTML($('#queue-summary', root), summary);

  const el = $('#queue-list', root);
  if (!list.length) {
    setHTML(el, html`<div class="empty is-compact">${raw(ART.queue)}
      <div class="empty-title">Nessun documento in coda</div>
      <p class="empty-text">I fogli importati compaiono qui con l'avanzamento della lettura.</p></div>`);
    return;
  }
  setHTML(el, list.map((d) => queueItem(d)));
}

function queueItem(d) {
  const key = statusKey(d);
  const pct = d.status === 'completato' || d.status === 'scartato' ? 100 : Math.round((d.progress || 0) * 100);
  let barCls = 'is-idle';
  if (d.status === 'in_lavorazione') barCls = 'is-active';
  else if (d.status === 'errore') barCls = 'is-err';
  else if (d.status === 'completato') barCls = key === 'errori' ? 'is-err' : 'is-ok';
  let msg = d.status_message || '';
  if (d.status === 'completato' && d.is_foglio_firma) {
    const t = d.totals || {};
    const parts = [`${fmtOre(t.ore_riconosciute || 0) || '0'} ore`];
    if (t.campi_illeggibili) parts.push(`${t.campi_illeggibili} illeggibili`);
    if (t.n_errori) parts.push(plural(t.n_errori, 'errore', 'errori'));
    else if (t.n_attenzioni) parts.push(plural(t.n_attenzioni, 'avviso', 'avvisi'));
    msg = parts.join(' · ');
  } else if (d.status === 'errore') {
    msg = d.error || d.status_message || 'Lettura non riuscita';
  }
  const canOpen = d.status === 'completato' || d.status === 'scartato' || d.status === 'errore';
  return html`<div class="queue-item" ${canOpen ? raw(`data-open="${esc(d.id)}" role="button" tabindex="0"`) : ''} title="${d.source_file}${d.page_count > 1 ? ` — pagina ${d.source_page} di ${d.page_count}` : ''}">
    <img class="queue-thumb" src="${authUrl(d.thumb_url)}" alt="" loading="lazy">
    <div class="queue-name">${d.display_name}</div>
    <div>${d.status === 'errore' ? html`<button type="button" class="btn btn-sm btn-ghost" data-reprocess="${d.id}">${icon('refresh')}Riprova</button>` : statusBadge(d, true)}</div>
    <div class="queue-meta">
      <div class="progress ${barCls}"><span style="width:${pct}%"></span></div>
      <span class="queue-msg" ${d.status === 'errore' ? raw(`data-tip="${esc(msg)}"`) : ''}>${msg}</span>
      <span class="queue-pct">${d.status === 'in_coda' ? '—' : `${pct}%`}</span>
    </div>
  </div>`;
}

/* ------------------------------------------------------------ caricamento */
function renderUpload(root) {
  const el = $('#upload-status', root);
  if (!el) return;
  if (!upload.active) { el.hidden = true; el.innerHTML = ''; return; }
  el.hidden = false;
  const pct = upload.total ? Math.round((upload.loaded / upload.total) * 100) : 0;
  setHTML(el, html`<div class="upload-status-row"><span class="spinner" style="color:var(--accent)"></span>
      <strong>Caricamento di ${plural(upload.files, 'file', 'file')}</strong><span class="spacer"></span><span class="num muted">${pct}%</span></div>
    <div class="progress is-active"><span style="width:${pct}%"></span></div>
    <div class="muted" style="font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${upload.current ? `${upload.current} · ` : ''}${uploadSummaryText()}</div>`);
}

/* ------------------------------------------------------------ flusso di lavoro */
function renderSteps(root) {
  const k = store.kpi || {};
  const docs = store.documents;
  const imported = (k.documenti || 0) > 0;
  const toVerify = k.da_verificare || 0;
  const completed = k.completati || 0;
  const reading = (k.in_coda || 0) + (k.in_lavorazione || 0);
  const s1 = imported ? 'is-done' : 'is-current';
  const s2 = !completed ? '' : (toVerify ? 'is-current' : 'is-done');
  const s3 = completed && !toVerify && !reading ? 'is-current' : '';
  const firstToVerify = docs.find((d) => d.status === 'completato' && d.is_foglio_firma && d.totals?.stato !== 'ok' && !d.user_verified)
    || docs.find((d) => d.status === 'completato' && d.is_foglio_firma && !d.user_verified);
  setHTML($('#steps', root), html`
    <div class="step ${s1}">
      <div class="step-num">${s1 === 'is-done' ? icon('check') : '1'}</div>
      <div class="step-title">Importa i fogli firma</div>
      <div class="step-text">${imported ? `${plural(k.documenti, 'documento importato', 'documenti importati')}${reading ? `, ${fmtInt(reading)} in lettura` : ''}.` : 'Trascina i PDF scansionati nell\'area qui sopra o usa «Importa PDF».'}</div>
    </div>
    <div class="step ${s2}">
      <div class="step-num">${s2 === 'is-done' ? icon('check') : '2'}</div>
      <div class="step-title">Verifica i campi segnalati</div>
      <div class="step-text">${completed ? (toVerify ? `${plural(toVerify, 'foglio da verificare', 'fogli da verificare')} · ${plural(k.illeggibili || 0, 'campo illeggibile', 'campi illeggibili')}.` : 'Tutti i fogli sono stati verificati o non hanno segnalazioni.') : 'Confronta la scansione con i dati letti e correggi le celle evidenziate.'}</div>
      ${firstToVerify ? html`<button type="button" class="btn btn-sm${s2 === 'is-current' ? ' btn-primary' : ''}" data-open="${firstToVerify.id}">${icon('arrow-right')}${toVerify ? 'Inizia la revisione' : 'Apri la revisione'}</button>` : ''}
    </div>
    <div class="step ${s3}">
      <div class="step-num">3</div>
      <div class="step-title">Genera l'Excel</div>
      <div class="step-text">${completed ? 'Controlla l\'anteprima e genera la rendicontazione con riepilogo, dettaglio, schede e anomalie.' : 'Disponibile quando almeno un foglio firma è stato letto.'}</div>
      ${completed ? html`<button type="button" class="btn btn-sm${s3 ? ' btn-primary' : ''}" data-nav="#/anteprima">${icon('excel')}Apri l'anteprima Excel</button>` : ''}
    </div>`);
}

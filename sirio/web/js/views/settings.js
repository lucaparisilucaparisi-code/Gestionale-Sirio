// Impostazioni: motore di lettura, Claude Vision (chiave, modello, accuratezza), aspetto,
// esportazione, cartelle, privacy e manutenzione.

import { del, post } from '../api.js';
import { icon } from '../icons.js';
import { loadSettings, refreshState, saveSettings, store } from '../state.js';
import { confirmDialog, toast, toastError } from '../ui.js';
import { $, esc, fmtUSD, html, raw, setHTML } from '../util.js';

const EFFORTS = [
  { id: 'low', label: 'Rapida', text: 'Lettura veloce, adatta a moduli molto chiari.' },
  { id: 'medium', label: 'Standard', text: 'Buon equilibrio tra tempi e precisione.' },
  { id: 'high', label: 'Alta (consigliata)', text: 'Ragionamento accurato sulle cifre scritte a mano.' },
  { id: 'xhigh', label: 'Molto alta', text: 'Più tempo di analisi sulle celle difficili.' },
  { id: 'max', label: 'Massima', text: 'Il massimo della precisione, con tempi e costi maggiori.' },
];
const SOURCE_TEXT = {
  env: 'dalla variabile d\'ambiente ANTHROPIC_API_KEY',
  keyring: 'nel portachiavi di sistema',
  file: 'in un file protetto nella cartella dei dati',
};
// token indicativi per foglio (pagina intera + 2 ritagli + risposta strutturata, verifica incrociata inclusa)
const TOKENS_PER_SHEET = { input: 14000, output: 4000 };

export function mount(root) {
  const view = { root, keyVisible: false, testing: false, testResult: null, saving: false };
  root.addEventListener('click', (e) => onClick(view, e));
  root.addEventListener('change', (e) => onChange(view, e));
  root.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.target.id === 'set-key') { e.preventDefault(); saveKey(view); }
  });
  render(view);
  loadSettings().then(() => render(view)).catch((err) => toastError(err, 'Impostazioni non disponibili'));
  return { onState: (kind) => { if (kind === 'settings') render(view); } };
}

function render(view) {
  const data = store.settings;
  const root = view.root;
  if (!data) {
    setHTML(root, html`<div class="settings"><div class="card settings-skel">${[1, 2, 3].map(() => html`<div class="skeleton" style="height:18px;width:40%"></div><div class="skeleton" style="height:90px"></div>`)}</div></div>`);
    return;
  }
  // preserva quanto digitato nel campo della chiave
  const typed = $('#set-key', root)?.value || '';
  const s = data.settings;
  const claude = s.engine === 'claude';
  const eng = data.engines || {};
  const model = (data.models || []).find((m) => m.id === s.claude_model) || data.models?.[0];
  const perSheet = model ? (TOKENS_PER_SHEET.input * model.input + TOKENS_PER_SHEET.output * model.output) / 1e6 : 0;

  const engineCard = (id, cfg) => {
    const st = eng[id] || {};
    const on = s.engine === id;
    return html`<button type="button" class="engine-card" role="radio" aria-checked="${String(on)}" data-engine="${id}">
      <span class="engine-card-check">${icon('check')}</span>
      <div class="engine-card-top">
        <span class="engine-card-icon ${cfg.cls}">${icon(cfg.icon)}</span>
        <div><div class="engine-card-title">${cfg.title}</div>
          <div class="engine-card-tags">${cfg.tags.map(([t, tone]) => html`<span class="badge badge-sm no-dot ${tone}">${t}</span>`)}</div></div>
      </div>
      <ul>${cfg.pros.map((p) => html`<li>${icon('check')}<span>${p}</span></li>`)}${cfg.cons.map((p) => html`<li class="is-minus">${icon('minus')}<span>${p}</span></li>`)}</ul>
      <div class="engine-card-status"><span class="dot ${st.available ? 'is-ok' : 'is-warn'}"></span><span>${st.message || (st.available ? 'Pronto' : 'Non disponibile')}</span></div>
    </button>`;
  };

  setHTML(root, html`<div class="settings">
    <section class="card">
      <div class="set-section">
        <div class="set-intro"><h3>${icon('scan')}Motore di lettura</h3>
          <p>Come vengono letti i fogli firma scritti a mano. Puoi cambiare motore in qualsiasi momento e rielaborare i documenti.</p></div>
        <div class="set-controls">
          <div class="engine-cards" role="radiogroup" aria-label="Motore di lettura">
            ${engineCard('locale', {
              title: 'Motore locale offline', icon: 'cpu', cls: 'is-local',
              tags: [['Predefinito', 'tone-info'], ['Gratuito', 'tone-ok'], ['Privacy totale', 'tone-neutral']],
              pros: ['Nessuna chiave API e nessun costo', 'I documenti non lasciano mai il computer', 'Funziona anche senza Internet'],
              cons: ['Precisione inferiore sulla grafia difficile: più campi da verificare'],
            })}
            ${engineCard('claude', {
              title: 'Claude Vision', icon: 'sparkle', cls: 'is-cloud',
              tags: [['Facoltativo', 'tone-neutral'], ['Massima precisione', 'tone-info'], ['A consumo', 'tone-warn']],
              pros: ['Lettura più accurata della scrittura a mano', 'Verifica incrociata delle righe dubbie', 'Più documenti letti in parallelo'],
              cons: ['Richiede una chiave API di Anthropic e invia le scansioni al servizio'],
            })}
          </div>
          ${privacyNote(claude)}
        </div>
      </div>

      ${claude ? html`
      <div class="set-section" id="claude-section">
        <div class="set-intro"><h3>${icon('key')}Claude Vision</h3>
          <p>Chiave API, modello e accuratezza della lettura. La chiave si crea su <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener noreferrer">console.anthropic.com</a> → <i>API Keys</i> (serve un credito prepagato).</p></div>
        <div class="set-controls">
          <div class="field">
            <label class="field-label" for="set-key">Chiave API di Anthropic</label>
            ${data.api_key_set ? html`<div class="key-status">${icon('check-circle', 'icon-sm')}<span>Chiave configurata <span class="mono">${data.api_key_hint || ''}</span>, salvata ${SOURCE_TEXT[data.api_key_source] || ''}.</span></div>` : html`<div class="key-status" style="background:var(--warning-soft);border-color:var(--warning-border)">${icon('alert-triangle', 'icon-sm')}<span>Nessuna chiave configurata: i documenti restano in coda finché non viene inserita.</span></div>`}
            <div class="row" style="gap:8px;align-items:stretch">
              <div class="input-group" style="flex:1">
                <input class="input mono" id="set-key" type="${view.keyVisible ? 'text' : 'password'}" placeholder="${data.api_key_set ? 'Incolla una nuova chiave per sostituirla' : 'sk-ant-…'}" autocomplete="off" spellcheck="false" ${data.api_key_source === 'env' ? raw('disabled') : ''}>
                <button type="button" class="btn btn-sm btn-ghost btn-icon input-addon" data-act="toggle-key" aria-label="${view.keyVisible ? 'Nascondi la chiave' : 'Mostra la chiave'}" data-tip="${view.keyVisible ? 'Nascondi' : 'Mostra'}">${icon(view.keyVisible ? 'eye-off' : 'eye')}</button>
              </div>
              <button type="button" class="btn btn-primary" data-act="save-key" ${data.api_key_source === 'env' ? raw('disabled') : ''}>${icon('save')}Salva</button>
              <button type="button" class="btn" data-act="test-key" ${view.testing ? raw('disabled') : ''}>${view.testing ? html`<span class="spinner"></span>Verifica…` : html`${icon('shield')}Verifica chiave`}</button>
            </div>
            ${data.api_key_source === 'env' ? html`<div class="field-hint">La chiave è impostata dalla variabile d'ambiente ANTHROPIC_API_KEY e ha la precedenza sulle altre.</div>` : ''}
            ${data.api_key_set && data.api_key_source !== 'env' ? html`<div class="field-hint"><button type="button" class="link-btn" data-act="remove-key" style="color:var(--danger-text)">Rimuovi la chiave salvata</button></div>` : ''}
            ${view.testResult ? html`<div class="banner ${view.testResult.ok ? 'is-ok' : 'is-err'}" style="padding:10px 12px" role="status">
              <div class="banner-icon" style="width:28px;height:28px">${icon(view.testResult.ok ? 'check-circle' : 'x-circle', 'icon-sm')}</div>
              <div class="banner-text" style="color:var(--text)">${view.testResult.message}</div></div>` : ''}
          </div>

          <div class="field">
            <label class="field-label" for="set-model">Modello</label>
            <select class="select" id="set-model" data-set="claude_model">${(data.models || []).map((m) => html`<option value="${m.id}" ${m.id === s.claude_model ? raw('selected') : ''}>${m.label}</option>`)}</select>
            ${model ? html`<div class="field-hint">${model.description || ''}</div>
            <div class="model-info">
              <div><div class="model-info-label">Input · per milione di token</div><div class="model-info-value">${fmtUSD(model.input)}</div></div>
              <div><div class="model-info-label">Output · per milione di token</div><div class="model-info-value">${fmtUSD(model.output)}</div></div>
              <div data-tip="Stima indicativa: circa ${TOKENS_PER_SHEET.input.toLocaleString('it-IT')} token in ingresso e ${TOKENS_PER_SHEET.output.toLocaleString('it-IT')} in uscita per foglio, verifica incrociata inclusa. Il costo effettivo è mostrato in Dashboard."><div class="model-info-label">Costo indicativo per foglio</div><div class="model-info-value">≈ ${fmtUSD(perSheet)}</div></div>
            </div>` : ''}
          </div>

          <div class="field">
            <label class="field-label" for="set-effort">Accuratezza della lettura</label>
            <select class="select" id="set-effort" data-set="claude_effort">${EFFORTS.map((e) => html`<option value="${e.id}" ${e.id === s.claude_effort ? raw('selected') : ''}>${e.label}</option>`)}</select>
            <div class="field-hint">${EFFORTS.find((e) => e.id === s.claude_effort)?.text || ''}</div>
          </div>

          <div class="set-row">
            <div class="set-row-text"><div class="set-row-title">Verifica incrociata</div>
              <div class="set-row-sub">Rilegge ingrandite le righe con incoerenze o campi incerti (consigliata).</div></div>
            <label class="switch"><input type="checkbox" data-set="verifica_incrociata" ${s.verifica_incrociata ? raw('checked') : ''} aria-label="Verifica incrociata"><span class="switch-track"></span></label>
          </div>

          <div class="set-row">
            <div class="set-row-text"><div class="set-row-title">Documenti letti in parallelo</div>
              <div class="set-row-sub">Più letture contemporanee accelerano le grandi importazioni (da 1 a 8).</div></div>
            <div class="stepper" role="group" aria-label="Documenti in parallelo">
              <button type="button" data-act="conc" data-d="-1" aria-label="Diminuisci">${icon('minus', 'icon-sm')}</button>
              <output class="num" aria-live="polite">${s.concorrenza}</output>
              <button type="button" data-act="conc" data-d="1" aria-label="Aumenta">${icon('plus', 'icon-sm')}</button>
            </div>
          </div>
        </div>
      </div>` : ''}

      <div class="set-section">
        <div class="set-intro"><h3>${icon('palette')}Aspetto</h3><p>Tema dell'interfaccia. «Automatico» segue l'impostazione del sistema operativo.</p></div>
        <div class="set-controls">
          <div class="segmented" role="group" aria-label="Tema">
            ${[['auto', 'Automatico', 'monitor'], ['chiaro', 'Chiaro', 'sun'], ['scuro', 'Scuro', 'moon']].map(([id, label, ic]) => html`<button type="button" data-theme-opt="${id}" aria-pressed="${String(s.tema === id)}">${icon(ic, 'icon-sm')}${label}</button>`)}
          </div>
        </div>
      </div>

      <div class="set-section">
        <div class="set-intro"><h3>${icon('excel')}Excel</h3><p>Opzioni predefinite del file generato (modificabili anche dall'Anteprima Excel).</p></div>
        <div class="set-controls">
          <div class="set-row">
            <div class="set-row-text"><div class="set-row-title">Una scheda per ogni foglio firma</div><div class="set-row-sub">Replica del modulo con celle illeggibili, incerte e corrette evidenziate.</div></div>
            <label class="switch"><input type="checkbox" data-set="export_fogli_per_documento" ${s.export_fogli_per_documento ? raw('checked') : ''} aria-label="Una scheda per ogni foglio firma"><span class="switch-track"></span></label>
          </div>
          <div class="set-row">
            <div class="set-row-text"><div class="set-row-title">Giorni senza dati nel dettaglio</div><div class="set-row-sub">Include nel dettaglio giornaliero anche weekend e giorni vuoti.</div></div>
            <label class="switch"><input type="checkbox" data-set="export_giorni_vuoti" ${s.export_giorni_vuoti ? raw('checked') : ''} aria-label="Giorni senza dati nel dettaglio"><span class="switch-track"></span></label>
          </div>
        </div>
      </div>

      <div class="set-section">
        <div class="set-intro"><h3>${icon('folder')}Cartelle</h3><p>Dove Sirio OCR conserva i documenti letti e dove salva i file Excel.</p></div>
        <div class="set-controls">
          <div class="field"><span class="field-label">File Excel generati</span>
            <div class="path-row"><div class="path-box" title="${data.export_dir}">${data.export_dir}</div>
              <button type="button" class="btn" data-act="open-export">${icon('folder-open')}Apri cartella export</button></div></div>
          <div class="field"><span class="field-label">Dati dell'applicazione</span>
            <div class="path-row"><div class="path-box" title="${data.data_dir}">${data.data_dir}</div>
              <button type="button" class="btn" data-act="copy" data-text="${data.data_dir}">${icon('copy')}Copia percorso</button></div></div>
        </div>
      </div>

      <div class="set-section">
        <div class="set-intro"><h3>${icon('trash')}Manutenzione</h3><p>Operazioni sui documenti importati.</p></div>
        <div class="set-controls">
          <div class="set-row">
            <div class="set-row-text"><div class="set-row-title">Elimina tutti i documenti</div>
              <div class="set-row-sub">Svuota l'archivio di Sirio OCR (scansioni, letture e correzioni). Gli Excel generati restano nella cartella export.</div></div>
            <button type="button" class="btn btn-danger-soft" data-act="clear-all">${icon('trash')}Elimina tutto</button>
          </div>
        </div>
      </div>
    </section>
    <p class="muted" style="text-align:center;font-size:12px">Sirio OCR ${data.version ? `versione ${data.version}` : ''} · Cooperativa Sociale Sirio · i dati restano sul computer, salvo l'uso facoltativo di Claude Vision.</p>
  </div>`);
  const keyInput = $('#set-key', root);
  if (keyInput && typed) keyInput.value = typed;
}

function privacyNote(claude) {
  if (claude) {
    return html`<div class="privacy is-cloud"><div class="privacy-icon">${icon('cloud')}</div><div>
      <div class="privacy-title">Nota sulla privacy — Claude Vision</div>
      <div class="privacy-text">Le scansioni dei fogli firma, che contengono <b>dati personali di minori</b>, vengono inviate all'API di Anthropic per la lettura. Per impostazione predefinita Anthropic non usa i dati ricevuti tramite API per addestrare i propri modelli. Verificare che l'uso sia coerente con le informative e le nomine privacy della cooperativa.</div>
    </div></div>`;
  }
  return html`<div class="privacy is-local"><div class="privacy-icon">${icon('shield')}</div><div>
    <div class="privacy-title">Privacy totale — motore locale</div>
    <div class="privacy-text">La lettura avviene interamente su questo computer: scansioni e dati personali (anche di minori) <b>non vengono inviati a nessun servizio esterno</b>. Alla prima lettura viene scaricato una sola volta il modello di riconoscimento della scrittura.</div>
  </div></div>`;
}

async function save(patch, message = 'Impostazioni salvate') {
  try {
    await saveSettings(patch);
    toast({ type: 'success', title: message, duration: 1800 });
    refreshState().catch(() => {});
    return true;
  } catch (err) {
    toastError(err, 'Impostazioni non salvate');
    try { await loadSettings(); } catch { /* ignorato */ }
    return false;
  }
}

async function saveKey(view) {
  const input = $('#set-key', view.root);
  const key = (input?.value || '').trim();
  if (!key) { toast({ type: 'warning', title: 'Inserire la chiave', message: 'Incollare la chiave API copiata dalla console di Anthropic.' }); input?.focus(); return; }
  view.testResult = null;
  if (await save({ api_key: key }, 'Chiave API salvata')) {
    if (input) input.value = '';
    await testKey(view);
  }
}

async function testKey(view) {
  const typed = ($('#set-key', view.root)?.value || '').trim();
  view.testing = true;
  view.testResult = null;
  render(view);
  try {
    view.testResult = await post('/api/settings/test', typed ? { api_key: typed } : {});
  } catch (err) {
    view.testResult = { ok: false, message: err.message };
  } finally {
    view.testing = false;
    if ($('#set-key', view.root)) $('#set-key', view.root).value = typed;
    render(view);
  }
}

async function onClick(view, e) {
  const engine = e.target.closest('[data-engine]');
  if (engine) {
    const id = engine.dataset.engine;
    if (store.settings?.settings?.engine === id) return;
    if (await save({ engine: id }, id === 'claude' ? 'Claude Vision selezionato' : 'Motore locale selezionato')) {
      if (id === 'claude' && !store.settings?.api_key_set) setTimeout(() => $('#set-key', view.root)?.focus(), 50);
    }
    return;
  }
  const theme = e.target.closest('[data-theme-opt]');
  if (theme) { await save({ tema: theme.dataset.themeOpt }, 'Tema aggiornato'); return; }
  const b = e.target.closest('[data-act]');
  if (!b) return;
  const act = b.dataset.act;
  if (act === 'toggle-key') {
    view.keyVisible = !view.keyVisible;
    const input = $('#set-key', view.root);
    if (input) input.type = view.keyVisible ? 'text' : 'password';
    b.innerHTML = String(icon(view.keyVisible ? 'eye-off' : 'eye'));
    b.setAttribute('aria-label', view.keyVisible ? 'Nascondi la chiave' : 'Mostra la chiave');
  } else if (act === 'save-key') saveKey(view);
  else if (act === 'test-key') testKey(view);
  else if (act === 'remove-key') {
    const ok = await confirmDialog({ title: 'Rimuovere la chiave API?', message: 'Claude Vision non potrà leggere nuovi documenti finché non verrà inserita una nuova chiave.', confirmLabel: 'Rimuovi', danger: true });
    if (ok) { view.testResult = null; await save({ api_key: '' }, 'Chiave API rimossa'); }
  } else if (act === 'conc') {
    const cur = store.settings?.settings?.concorrenza || 3;
    const next = Math.max(1, Math.min(8, cur + Number(b.dataset.d)));
    if (next !== cur) await save({ concorrenza: next });
  } else if (act === 'open-export') {
    try { await post('/api/open', { target: 'export_dir' }); toast({ type: 'info', title: 'Cartella export aperta', duration: 2200 }); } catch (err) { toastError(err, 'Impossibile aprire la cartella'); }
  } else if (act === 'copy') {
    try {
      await navigator.clipboard.writeText(b.dataset.text || '');
      toast({ type: 'success', title: 'Percorso copiato', duration: 1800 });
    } catch {
      toast({ type: 'warning', title: 'Copia non riuscita', message: 'Selezionare e copiare il percorso manualmente.' });
    }
  } else if (act === 'clear-all') {
    const n = store.documents.length;
    const ok = await confirmDialog({
      title: 'Eliminare tutti i documenti?',
      message: raw(`<p>Verranno eliminati <b>${esc(String(n))} documenti</b> con le scansioni, le letture e le correzioni manuali. L'operazione non è reversibile.</p><p>Gli Excel già generati restano nella cartella export.</p>`),
      confirmLabel: 'Elimina tutto',
      danger: true,
    });
    if (!ok) return;
    try {
      await del('/api/documents');
      toast({ type: 'success', title: 'Archivio svuotato', duration: 2500 });
      await refreshState(true);
    } catch (err) { toastError(err); }
  }
}

async function onChange(view, e) {
  const el = e.target.closest('[data-set]');
  if (!el) return;
  const key = el.dataset.set;
  const value = el.type === 'checkbox' ? el.checked : el.value;
  await save({ [key]: value });
}

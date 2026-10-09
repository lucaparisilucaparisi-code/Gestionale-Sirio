// Sirio OCR — avvio dell'interfaccia: struttura, navigazione, ricerca globale, ciclo di vita.

import { post, sendBye, authUrl } from './js/api.js';
import { icon } from './js/icons.js';
import { loadSettings, refreshState, startPolling, store, subscribe } from './js/state.js';
import { closeMenu, hideTooltip, initMenus, initTooltips, statusBadge, toast } from './js/ui.js';
import { filesFromDataTransfer, filesFromInput, pickFiles, uploadItems } from './js/upload.js';
import { $, esc, fmtPeriodo, html, isTypingTarget, normalizeText, raw, setHTML, storageGet, storageSet } from './js/util.js';
import * as dashboard from './js/views/dashboard.js';
import * as documenti from './js/views/documents.js';
import * as revisione from './js/views/review.js';
import * as anteprima from './js/views/preview.js';
import * as impostazioni from './js/views/settings.js';

const ROUTES = {
  dashboard: { view: dashboard, title: 'Dashboard', sub: 'Panoramica dei fogli firma e della lettura in corso', nav: 'dashboard' },
  documenti: { view: documenti, title: 'Documenti', sub: 'Tutti i fogli firma importati', nav: 'documenti' },
  revisione: { view: revisione, title: 'Revisione documento', sub: 'Confronto tra scansione e dati letti', nav: 'documenti', rail: true, fill: true },
  anteprima: { view: anteprima, title: 'Anteprima Excel', sub: 'Controllo e modifica dei dati prima della generazione', nav: 'anteprima', fill: true },
  impostazioni: { view: impostazioni, title: 'Impostazioni', sub: 'Motore di lettura, chiave API, aspetto e cartelle', nav: 'impostazioni' },
};

const app = { current: null, route: null, name: null, railPref: storageGet('sirio.rail', false), navToken: 0 };

/* ================================================================== tema */
const media = window.matchMedia('(prefers-color-scheme: dark)');
export function applyTheme(tema) {
  const t = tema || 'auto';
  storageSet('sirio.tema', t);
  try { localStorage.setItem('sirio.tema', t); } catch { /* ignorato */ }
  const dark = t === 'scuro' || (t === 'auto' && media.matches);
  document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
}
media.addEventListener('change', () => applyTheme(store.settings?.settings?.tema || readTheme()));
function readTheme() {
  try { return localStorage.getItem('sirio.tema') || 'auto'; } catch { return 'auto'; }
}

/* ================================================================== struttura */
function renderShell() {
  setHTML($('#sidebar'), html`
    <div class="brand">
      <img class="brand-mark" src="/static/assets/logo.svg" alt="" width="38" height="38">
      <div class="brand-text">
        <div class="brand-name">Sirio <span>OCR</span></div>
        <div class="brand-sub">Fogli firma · Comune di Napoli</div>
      </div>
    </div>
    <nav class="nav" aria-label="Sezioni">
      <div class="nav-label">Area di lavoro</div>
      ${navItem('dashboard', 'Dashboard', 'dashboard')}
      ${navItem('documenti', 'Documenti', 'documents')}
      ${navItem('anteprima', 'Anteprima Excel', 'excel')}
      <div class="nav-label">Sistema</div>
      ${navItem('impostazioni', 'Impostazioni', 'settings')}
    </nav>
    <div class="sidebar-foot">
      <button type="button" class="engine-pill" id="engine-pill" aria-label="Stato del motore di lettura">
        <span class="dot" id="engine-dot"></span>
        <span class="engine-pill-title" id="engine-title">Motore di lettura</span>
        <span class="engine-pill-sub" id="engine-sub">Verifica in corso…</span>
      </button>
      <div class="sidebar-meta">
        <span class="sidebar-version" id="app-version"></span>
        <button type="button" class="rail-toggle" id="rail-toggle" aria-label="Riduci o espandi la barra laterale" data-tip="Riduci/espandi la barra laterale">${icon('sidebar', 'icon-sm')}</button>
      </div>
    </div>`);

  setHTML($('#topbar'), html`
    <div class="page-head">
      <div class="page-head-text">
        <h1 class="page-title" id="page-title">Sirio OCR</h1>
        <div class="page-sub" id="page-sub"></div>
      </div>
    </div>
    <div class="search" id="search">
      ${icon('search')}
      <input type="search" id="search-input" placeholder="Cerca operatore, alunno, istituto…" autocomplete="off"
             aria-label="Cerca nei documenti" aria-controls="search-pop" aria-expanded="false">
      <span class="kbd-hint"><kbd>Ctrl</kbd> <kbd>K</kbd></span>
    </div>
    <div class="topbar-actions">
      <button type="button" class="btn btn-primary" id="btn-import">${icon('upload')}Importa PDF</button>
    </div>`);

  $('#btn-import').addEventListener('click', pickFiles);
  $('#engine-pill').addEventListener('click', () => navigate('#/impostazioni'));
  $('#rail-toggle').addEventListener('click', () => {
    app.railPref = !document.getElementById('app').classList.contains('is-rail');
    storageSet('sirio.rail', app.railPref);
    app.railOverride = true;
    applyRail();
  });
  initSearch();
}

function navItem(route, label, ic) {
  return html`<a class="nav-item" href="#/${route}" data-route="${route}" data-tip-label="${label}">
    ${raw(icon(ic))}<span class="nav-text">${label}</span><span class="nav-badge" data-badge="${route}"></span><span class="nav-badge-dot" data-dot="${route}" hidden></span></a>`;
}

function applyRail() {
  const shell = document.getElementById('app');
  const narrow = window.innerWidth < 1240;
  const wantRail = app.route?.rail ? !app.railOverride || app.railPref : (narrow || app.railPref);
  shell.classList.toggle('is-rail', !!wantRail);
  document.querySelectorAll('.nav-item').forEach((a) => {
    if (wantRail) a.setAttribute('data-tip', a.dataset.tipLabel || '');
    else a.removeAttribute('data-tip');
  });
}
window.addEventListener('resize', () => applyRail());

function updateChrome() {
  const k = store.kpi;
  const q = store.queue;
  const docsBadge = document.querySelector('[data-badge="documenti"]');
  const dashBadge = document.querySelector('[data-badge="dashboard"]');
  const dot = document.querySelector('[data-dot="documenti"]');
  if (docsBadge) {
    const n = k?.da_verificare || 0;
    docsBadge.textContent = n ? String(n) : (k?.documenti ? String(k.documenti) : '');
    docsBadge.classList.toggle('is-warn', n > 0);
    docsBadge.setAttribute('aria-label', n ? `${n} da verificare` : `${k?.documenti || 0} documenti`);
    if (dot) dot.hidden = !n;
  }
  if (dashBadge) {
    const busy = (q?.in_coda || 0) + (q?.in_lavorazione || 0);
    dashBadge.textContent = busy ? String(busy) : '';
  }
  // stato del motore
  const s = store.settings;
  const engine = q?.motore || s?.settings?.engine || 'locale';
  const name = engine === 'claude' ? 'Claude Vision' : 'Motore locale';
  let tone = 'is-warn';
  let sub = 'Verifica in corso…';
  if (q) {
    if (!q.motore_pronto) { tone = q.in_coda ? 'is-err' : 'is-warn'; sub = q.messaggio || 'Non disponibile'; }
    else if (q.in_lavorazione || q.in_coda) { tone = 'is-busy'; sub = `In lettura: ${q.in_lavorazione} · in coda: ${q.in_coda}`; }
    else { tone = 'is-ok'; sub = 'Pronto'; }
  }
  const dotEl = $('#engine-dot');
  if (dotEl) dotEl.className = `dot ${tone}`;
  const title = $('#engine-title'); if (title) title.textContent = name;
  const subEl = $('#engine-sub'); if (subEl) subEl.textContent = sub;
  const pill = $('#engine-pill');
  if (pill) {
    pill.setAttribute('data-tip', `${name}: ${sub}`);
    pill.setAttribute('aria-label', `${name}: ${sub}. Apri le impostazioni`);
  }
  const ver = $('#app-version');
  if (ver && s?.version) ver.textContent = `Versione ${s.version}`;
}

/* ================================================================== navigazione */
export function navigate(hash, { replace = false } = {}) {
  if (replace) history.replaceState(null, '', hash);
  else if (location.hash !== hash) { location.hash = hash; return; }
  route();
}

function parseHash() {
  const h = location.hash.replace(/^#\/?/, '');
  const [path, qs] = h.split('?');
  const parts = path.split('/').filter(Boolean);
  const name = ROUTES[parts[0]] ? parts[0] : 'dashboard';
  return { name, params: parts.slice(1).map(decodeURIComponent), query: new URLSearchParams(qs || '') };
}

async function route() {
  const { name, params, query } = parseHash();
  const def = ROUTES[name];
  const token = ++app.navToken;
  // la vista corrente può chiedere di completare il salvataggio prima di uscire
  if (app.current?.beforeLeave) {
    try { await app.current.beforeLeave(); } catch { /* il salvataggio segnala da sé l'errore */ }
  }
  if (token !== app.navToken) return;
  if (app.current && app.name === name && app.current.update && app.current.update(params, query) !== false) {
    return;
  }
  closeMenu();
  hideTooltip();
  try { app.current?.unmount?.(); } catch (err) { console.warn(err); }
  app.current = null;
  app.route = def;
  app.name = name;
  app.railOverride = false;
  applyRail();
  const view = $('#view');
  view.className = `view${def.fill ? ' is-fill' : ''}`;
  view.innerHTML = '';
  view.scrollTop = 0;
  setTitle(def.title, def.sub);
  document.querySelectorAll('.nav-item').forEach((a) => {
    const on = a.dataset.route === def.nav;
    a.classList.toggle('is-active', on);
    if (on) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  const host = document.createElement('div');
  host.className = 'view-enter';
  host.style.cssText = def.fill ? 'display:flex;flex:1;min-width:0;min-height:0' : '';
  view.appendChild(host);
  app.current = def.view.mount(host, { params, query, navigate, setTitle }) || {};
}

export function setTitle(title, sub) {
  const t = $('#page-title'); const s = $('#page-sub');
  if (t) t.textContent = title || 'Sirio OCR';
  if (s) s.textContent = sub || '';
  document.title = title && title !== 'Dashboard' ? `${title} — Sirio OCR` : 'Sirio OCR';
}

/* ================================================================== ricerca */
function initSearch() {
  const input = $('#search-input');
  const box = $('#search');
  let pop = null;
  let idx = -1;
  let results = [];

  const close = () => { pop?.remove(); pop = null; idx = -1; input.setAttribute('aria-expanded', 'false'); };
  const open = (id) => { close(); input.blur(); navigate(`#/revisione/${encodeURIComponent(id)}`); };
  const showAll = () => {
    const q = input.value.trim();
    close();
    input.blur();
    navigate(`#/documenti${q ? `?q=${encodeURIComponent(q)}` : ''}`);
  };
  const render = () => {
    const q = normalizeText(input.value.trim());
    if (!q) { close(); return; }
    const terms = q.split(/\s+/);
    results = store.documents.filter((d) => {
      const h = d.header || {};
      const hay = normalizeText([d.display_name, d.source_file, h.operatore, h.alunno, h.istituto, h.ente, fmtPeriodo(h.mese, h.anno, '')].join(' '));
      return terms.every((t) => hay.includes(t));
    }).slice(0, 8);
    if (!pop) {
      pop = document.createElement('div');
      pop.className = 'search-pop';
      pop.id = 'search-pop';
      pop.setAttribute('role', 'listbox');
      box.appendChild(pop);
      pop.addEventListener('mousedown', (e) => e.preventDefault());
      pop.addEventListener('click', (e) => {
        const it = e.target.closest('[data-id]');
        if (it) open(it.dataset.id);
        else if (e.target.closest('[data-all]')) showAll();
      });
    }
    input.setAttribute('aria-expanded', 'true');
    idx = results.length ? 0 : -1;
    pop.innerHTML = results.length
      ? results.map((d, i) => `<button type="button" class="search-item${i === idx ? ' is-active' : ''}" role="option" data-id="${esc(d.id)}">
          <img src="${esc(authUrl(d.thumb_url))}" alt="" loading="lazy">
          <span style="min-width:0"><div class="search-item-title">${esc(d.header?.operatore || d.display_name)}</div>
          <div class="search-item-sub">${esc([d.header?.alunno, d.header?.istituto, fmtPeriodo(d.header?.mese, d.header?.anno, '')].filter(Boolean).join(' · ') || d.source_file)}</div></span>
          ${statusBadge(d, true)}</button>`).join('')
        + `<div class="search-foot"><button type="button" class="link-btn" data-all>Mostra tutti i risultati in Documenti</button><span><kbd>↵</kbd> apri</span></div>`
      : `<div class="search-empty">Nessun documento corrisponde a «${esc(input.value.trim())}».</div><div class="search-foot"><button type="button" class="link-btn" data-all>Cerca in Documenti</button><span></span></div>`;
  };
  const highlight = () => {
    pop?.querySelectorAll('.search-item').forEach((el, i) => el.classList.toggle('is-active', i === idx));
  };
  input.addEventListener('input', render);
  input.addEventListener('focus', () => { if (input.value.trim()) render(); });
  input.addEventListener('blur', () => setTimeout(close, 120));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown' && results.length) { e.preventDefault(); idx = (idx + 1) % results.length; highlight(); }
    else if (e.key === 'ArrowUp' && results.length) { e.preventDefault(); idx = (idx - 1 + results.length) % results.length; highlight(); }
    else if (e.key === 'Enter') { e.preventDefault(); if (idx >= 0 && results[idx]) open(results[idx].id); else showAll(); }
    else if (e.key === 'Escape') { e.preventDefault(); input.value = ''; close(); input.blur(); }
  });
}

/* ================================================================== trascinamento globale */
function initGlobalDrop() {
  let depth = 0;
  const hasFiles = (e) => Array.from(e.dataTransfer?.types || []).includes('Files');
  window.addEventListener('dragenter', (e) => { if (hasFiles(e)) { depth++; document.body.classList.add('is-dragging-files'); } });
  window.addEventListener('dragleave', (e) => { if (hasFiles(e)) { depth = Math.max(0, depth - 1); if (!depth) document.body.classList.remove('is-dragging-files'); } });
  window.addEventListener('dragover', (e) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } });
  window.addEventListener('drop', async (e) => {
    depth = 0;
    document.body.classList.remove('is-dragging-files');
    if (!hasFiles(e)) return;
    e.preventDefault();
    const items = await filesFromDataTransfer(e.dataTransfer);
    if (items.length) {
      uploadItems(items);
      if (app.name !== 'dashboard') navigate('#/dashboard');
    }
  });
  for (const id of ['file-input', 'folder-input']) {
    const input = document.getElementById(id);
    input.addEventListener('change', () => {
      const items = filesFromInput(input);
      input.value = '';
      if (items.length) {
        uploadItems(items);
        if (app.name !== 'dashboard' && app.name !== 'documenti') navigate('#/dashboard');
      }
    });
  }
}

/* ================================================================== tastiera globale */
function initKeyboard() {
  document.addEventListener('keydown', (e) => {
    const ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && !e.shiftKey && !e.altKey && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      $('#search-input')?.focus();
      $('#search-input')?.select();
      return;
    }
    if (ctrl && e.key.toLowerCase() === 'o' && !e.shiftKey) {
      e.preventDefault();
      pickFiles();
      return;
    }
    if (e.key === '/' && !isTypingTarget(e.target) && !document.querySelector('dialog[open]') && !e.target.closest?.('.sheet')) {
      e.preventDefault();
      $('#search-input')?.focus();
    }
  });
}

/* ================================================================== ciclo di vita */
function initLifecycle() {
  const beat = () => post('/api/heartbeat').catch(() => {});
  beat();
  setInterval(beat, 20000);
  window.addEventListener('pagehide', () => sendBye());
}

let offlineEl = null;
function onConnection() {
  if (store.connection === 'lost' && !offlineEl) {
    offlineEl = document.createElement('div');
    offlineEl.className = 'offline';
    offlineEl.setAttribute('role', 'alertdialog');
    offlineEl.setAttribute('aria-label', 'Connessione persa');
    offlineEl.innerHTML = `<div class="card offline-card">
      <img src="/static/assets/logo.svg" alt="">
      <h2 style="font-size:17px">Connessione con Sirio OCR interrotta</h2>
      <p class="muted" style="font-size:13.5px">Il programma potrebbe essere stato chiuso. Sto riprovando a collegarmi…
      Se il problema persiste, chiudere questa finestra e riavviare Sirio OCR.</p>
      <div class="progress is-indeterminate" style="width:180px;margin-top:6px"><span></span></div>
    </div>`;
    document.body.appendChild(offlineEl);
  } else if (store.connection === 'ok' && offlineEl) {
    offlineEl.remove();
    offlineEl = null;
    toast({ type: 'success', title: 'Connessione ripristinata', duration: 2500 });
  }
}

/* ================================================================== avvio */
async function boot() {
  applyTheme(readTheme());
  renderShell();
  initTooltips();
  initMenus();
  initGlobalDrop();
  initKeyboard();
  initLifecycle();
  subscribe((kind) => {
    if (kind === 'connection') onConnection();
    if (kind === 'state' || kind === 'settings' || kind === 'upload') updateChrome();
    if (kind === 'settings' && store.settings?.settings?.tema) applyTheme(store.settings.settings.tema);
    try { app.current?.onState?.(kind); } catch (err) { console.warn('Aggiornamento della vista non riuscito', err); }
  });
  window.addEventListener('hashchange', route);
  try {
    await Promise.all([refreshState(true), loadSettings()]);
  } catch (err) {
    toast({ type: 'error', title: 'Avvio incompleto', message: err.message });
  }
  updateChrome();
  route();
  startPolling();
  const splash = $('#splash');
  splash?.classList.add('is-hidden');
  setTimeout(() => splash?.remove(), 400);
}

window.addEventListener('unhandledrejection', (e) => {
  const err = e.reason;
  if (err && err.name === 'AbortError') { e.preventDefault(); return; }
  if (err && err.name === 'ApiError') {
    e.preventDefault();
    toast({ type: 'error', title: 'Operazione non riuscita', message: err.message });
  }
});

boot();

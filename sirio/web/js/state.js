// Stato condiviso dell'applicazione e sincronizzazione con il server (polling a revisione).

import { ApiError, get, put } from './api.js';

const listeners = new Set();

export const store = {
  revision: null,
  documents: [],
  kpi: null,
  queue: null,
  settings: null,          // risposta completa di GET /api/settings
  loaded: false,
  uploading: false,
  connection: 'ok',        // 'ok' | 'lost'
};

export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

function emit(kind) {
  for (const fn of Array.from(listeners)) {
    try { fn(kind); } catch (err) { console.warn('Aggiornamento della vista non riuscito', err); }
  }
}

export const docById = (id) => store.documents.find((d) => d.id === id) || null;

export function isBusy() {
  const q = store.queue;
  return store.uploading || !!(q && (q.in_coda > 0 || q.in_lavorazione > 0));
}

/* ------------------------------------------------------------------ stato */
let failures = 0;
let pollTimer = null;
let inflight = null;

export async function refreshState(force = false) {
  if (inflight) return inflight;
  const since = !force && store.revision !== null ? `?since=${store.revision}` : '';
  inflight = (async () => {
    try {
      const data = await get(`/api/state${since}`);
      failures = 0;
      if (store.connection !== 'ok') { store.connection = 'ok'; emit('connection'); }
      if (data.unchanged) return false;
      store.revision = data.revision;
      store.documents = data.documents || [];
      store.kpi = data.kpi || null;
      store.queue = data.queue || null;
      store.loaded = true;
      emit('state');
      return true;
    } catch (err) {
      if (err instanceof ApiError && err.status === 0) {
        failures += 1;
        if (failures >= 3 && store.connection !== 'lost') { store.connection = 'lost'; emit('connection'); }
      }
      throw err;
    } finally {
      inflight = null;
    }
  })();
  return inflight;
}

function nextDelay() {
  if (store.connection === 'lost') return 2500;
  if (isBusy()) return 1000;
  return document.hidden ? 12000 : 4000;
}

export function startPolling() {
  const tick = async () => {
    try { await refreshState(); } catch { /* gestito tramite store.connection */ }
    pollTimer = setTimeout(tick, nextDelay());
  };
  clearTimeout(pollTimer);
  tick();
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) { clearTimeout(pollTimer); tick(); }
  });
}

/** Forza un aggiornamento ravvicinato (es. dopo un caricamento o una rielaborazione). */
export function pokePolling() {
  clearTimeout(pollTimer);
  pollTimer = setTimeout(async function tick() {
    try { await refreshState(); } catch { /* ignorato */ }
    pollTimer = setTimeout(tick, nextDelay());
  }, 150);
}

/* ------------------------------------------------------------------ impostazioni */
export async function loadSettings() {
  const data = await get('/api/settings');
  store.settings = data;
  emit('settings');
  return data;
}

export async function saveSettings(patch) {
  const data = await put('/api/settings', patch);
  store.settings = data;
  emit('settings');
  return data;
}

export function setUploading(v) {
  store.uploading = v;
  emit('upload');
}

export function emitLocal(kind) { emit(kind); }

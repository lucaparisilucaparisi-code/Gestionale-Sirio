// Accesso all'API locale di Sirio OCR (token di sessione in ogni richiesta).

export const TOKEN = document.querySelector('meta[name="sirio-token"]')?.getAttribute('content') || '';

export class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.data = data;
  }
}

const NETWORK_MESSAGE = 'Impossibile comunicare con Sirio OCR: il programma potrebbe essere stato chiuso.';
const STATUS_MESSAGES = {
  400: 'Richiesta non valida.',
  403: 'Sessione non valida: chiudere la finestra e riaprire Sirio OCR.',
  404: 'Risorsa non trovata.',
  409: 'Operazione non possibile in questo momento.',
  413: 'Il contenuto inviato è troppo grande.',
  422: 'Dati non validi.',
  500: 'Errore interno di Sirio OCR (dettagli nel registro sirio.log).',
  503: 'Servizio momentaneamente non disponibile.',
};

function authHeaders(extra = {}) {
  return { 'X-Sirio-Token': TOKEN, ...extra };
}

async function errorFrom(res) {
  let detail = '';
  let data = null;
  try {
    data = await res.json();
    if (data && typeof data.detail === 'string') detail = data.detail;
  } catch { /* corpo non JSON */ }
  return new ApiError(detail || STATUS_MESSAGES[res.status] || `Errore ${res.status}.`, res.status, data);
}

/**
 * Richiesta JSON. `body` viene serializzato; con `raw: true` restituisce la Response.
 */
export async function api(path, { method = 'GET', body, signal, raw = false, keepalive = false } = {}) {
  const init = { method, headers: authHeaders({ Accept: 'application/json' }), signal, keepalive, cache: 'no-store' };
  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, init);
  } catch (err) {
    if (err && err.name === 'AbortError') throw err;
    throw new ApiError(NETWORK_MESSAGE, 0);
  }
  if (!res.ok) throw await errorFrom(res);
  if (raw) return res;
  if (res.status === 204) return null;
  try {
    return await res.json();
  } catch {
    throw new ApiError('Risposta non valida da Sirio OCR.', res.status);
  }
}

export const get = (path, opts) => api(path, { ...opts, method: 'GET' });
export const post = (path, body, opts) => api(path, { ...opts, method: 'POST', body });
export const put = (path, body, opts) => api(path, { ...opts, method: 'PUT', body });
export const del = (path, opts) => api(path, { ...opts, method: 'DELETE' });

/** URL di una risorsa letta da <img> (non può inviare intestazioni): token come parametro. */
export function authUrl(path) {
  if (!path) return '';
  return `${path}${path.includes('?') ? '&' : '?'}t=${encodeURIComponent(TOKEN)}`;
}

/**
 * Carica file con XMLHttpRequest per avere l'avanzamento del trasferimento.
 * `onProgress(loaded, total)`; restituisce la risposta JSON del server.
 */
export function uploadFiles(files, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    for (const f of files) form.append('files', f.file || f, f.name || f.file?.name || 'file');
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.setRequestHeader('X-Sirio-Token', TOKEN);
    xhr.setRequestHeader('Accept', 'application/json');
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => { if (onProgress) onProgress(e.loaded, e.lengthComputable ? e.total : 0); };
    xhr.onload = () => {
      const data = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300 && data) resolve(data);
      else reject(new ApiError((data && data.detail) || STATUS_MESSAGES[xhr.status] || 'Caricamento non riuscito.', xhr.status, data));
    };
    xhr.onerror = () => reject(new ApiError(NETWORK_MESSAGE, 0));
    xhr.onabort = () => reject(new ApiError('Caricamento annullato.', 0));
    xhr.send(form);
  });
}

/** Scarica un file protetto dal token e lo salva tramite un URL blob. */
export async function downloadFile(path, filename) {
  const res = await api(path, { raw: true });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename || 'download';
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

/** Avvisa il server che la finestra si sta chiudendo (fetch keepalive con token). */
export function sendBye() {
  try {
    fetch('/api/bye', { method: 'POST', keepalive: true, headers: authHeaders() }).catch(() => {});
  } catch { /* la pagina si sta chiudendo */ }
}

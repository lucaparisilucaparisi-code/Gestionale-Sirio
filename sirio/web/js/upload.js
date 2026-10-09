// Importazione dei file: selezione, trascinamento (anche cartelle intere) e caricamento a lotti.

import { uploadFiles as xhrUpload } from './api.js';
import { pokePolling, refreshState, setUploading, store } from './state.js';
import { toast } from './ui.js';
import { esc, fmtBytes, plural, raw } from './util.js';

export const SUPPORTED = ['.pdf', '.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp'];
const BATCH_FILES = 40;
const BATCH_BYTES = 64 * 1024 * 1024;

export const upload = { active: false, files: 0, done: 0, loaded: 0, total: 0, current: '' };
const listeners = new Set();
export const onUpload = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
const notify = () => listeners.forEach((fn) => { try { fn(upload); } catch { /* vista smontata */ } });

const ext = (name) => {
  const i = name.lastIndexOf('.');
  return i >= 0 ? name.slice(i).toLowerCase() : '';
};
export const isSupported = (name) => SUPPORTED.includes(ext(name));

/* ------------------------------------------------------------ cartelle */
function readEntries(reader) {
  return new Promise((resolve) => reader.readEntries(resolve, () => resolve([])));
}

async function walk(entry, out, path = '') {
  if (!entry) return;
  if (entry.isFile) {
    const file = await new Promise((resolve) => entry.file(resolve, () => resolve(null)));
    if (file) out.push({ file, name: file.name, path: path + file.name });
    return;
  }
  if (entry.isDirectory) {
    const reader = entry.createReader();
    // readEntries restituisce i risultati a blocchi: si legge finché non torna vuoto
    for (;;) {
      const batch = await readEntries(reader);
      if (!batch.length) break;
      for (const child of batch) await walk(child, out, `${path}${entry.name}/`);
    }
  }
}

/** File (anche dentro cartelle) da un evento di trascinamento. */
export async function filesFromDataTransfer(dt) {
  const out = [];
  const items = Array.from(dt.items || []);
  const entries = items.map((it) => (it.kind === 'file' && it.webkitGetAsEntry ? it.webkitGetAsEntry() : null));
  if (entries.some(Boolean)) {
    for (const entry of entries) await walk(entry, out);
    return out;
  }
  for (const file of Array.from(dt.files || [])) out.push({ file, name: file.name, path: file.name });
  return out;
}

export function filesFromInput(input) {
  return Array.from(input.files || []).map((file) => ({ file, name: file.name, path: file.webkitRelativePath || file.name }));
}

export function pickFiles() { document.getElementById('file-input')?.click(); }
export function pickFolder() { document.getElementById('folder-input')?.click(); }

/* ------------------------------------------------------------ caricamento */
function batches(items) {
  const out = [];
  let cur = []; let size = 0;
  for (const it of items) {
    const s = it.file.size || 0;
    if (cur.length && (cur.length >= BATCH_FILES || size + s > BATCH_BYTES)) { out.push(cur); cur = []; size = 0; }
    cur.push(it); size += s;
  }
  if (cur.length) out.push(cur);
  return out;
}

/**
 * Carica i file indicati (oggetti {file, name}). Restituisce il riepilogo dei risultati.
 */
export async function uploadItems(items, { onDone } = {}) {
  if (upload.active) {
    toast({ type: 'info', title: 'Caricamento già in corso', message: 'Attendere il termine del caricamento attuale.' });
    return null;
  }
  const supported = items.filter((it) => isSupported(it.name) && !it.name.startsWith('.'));
  const ignored = items.length - supported.length;
  if (!supported.length) {
    toast({
      type: 'warning',
      title: 'Nessun file da importare',
      message: items.length
        ? 'I file selezionati non sono PDF o immagini supportate (JPG, PNG, TIFF, BMP, WEBP).'
        : 'La selezione è vuota.',
    });
    return null;
  }
  const total = supported.reduce((s, it) => s + (it.file.size || 0), 0);
  Object.assign(upload, { active: true, files: supported.length, done: 0, loaded: 0, total, current: '' });
  setUploading(true);
  notify();
  const result = { documents: [], errors: [], duplicates: [] };
  let sent = 0;
  try {
    for (const group of batches(supported)) {
      upload.current = group.length === 1 ? group[0].name : `${group[0].name} e altri ${group.length - 1}`;
      notify();
      try {
        const res = await xhrUpload(group, (loaded) => {
          upload.loaded = Math.min(total, sent + loaded);
          notify();
        });
        result.documents.push(...(res.documents || []));
        result.errors.push(...(res.errors || []));
        result.duplicates.push(...(res.duplicates || []));
      } catch (err) {
        for (const it of group) result.errors.push({ file: it.name, message: err.message });
      }
      sent += group.reduce((s, it) => s + (it.file.size || 0), 0);
      upload.done += group.length;
      upload.loaded = sent;
      notify();
      pokePolling();
    }
  } finally {
    Object.assign(upload, { active: false, current: '' });
    setUploading(false);
    notify();
    try { await refreshState(true); } catch { /* il polling riproverà */ }
  }
  reportUpload(result, ignored);
  onDone?.(result);
  return result;
}

function reportUpload(result, ignored) {
  const nDocs = result.documents.length;
  const fresh = result.documents.filter((d) => d.status === 'in_coda' || d.status === 'in_lavorazione').length;
  const parts = [];
  if (nDocs) {
    const engine = store.queue?.motore === 'claude' ? 'Claude Vision' : 'il motore locale';
    parts.push(fresh ? `La lettura con ${engine} è in corso: i risultati compaiono nella coda.` : 'I documenti sono pronti per la revisione.');
  }
  if (result.duplicates.length) parts.push(`${plural(result.duplicates.length, 'file era già stato importato', 'file erano già stati importati')}: nessun duplicato creato.`);
  if (ignored) parts.push(`${plural(ignored, 'file ignorato', 'file ignorati')} perché non supportati.`);
  if (nDocs) {
    toast({ type: 'success', title: `${plural(nDocs, 'foglio firma importato', 'fogli firma importati')}`, message: parts.join(' ') });
  } else if (result.duplicates.length && !result.errors.length) {
    toast({ type: 'info', title: 'Nessun nuovo documento', message: parts.join(' ') });
  }
  if (result.errors.length) {
    const list = result.errors.slice(0, 4).map((e) => `<li><b>${esc(e.file)}</b>: ${esc(e.message)}</li>`).join('');
    const more = result.errors.length > 4 ? `<li>…e altri ${result.errors.length - 4}.</li>` : '';
    toast({
      type: 'error',
      title: result.errors.length === 1 ? 'Un file non è stato importato' : `${result.errors.length} file non importati`,
      message: raw(`<ul>${list}${more}</ul>`),
      duration: 12000,
    });
  }
}

export function uploadSummaryText() {
  if (!upload.active) return '';
  const pct = upload.total ? Math.round((upload.loaded / upload.total) * 100) : 0;
  return `${upload.done}/${upload.files} file · ${fmtBytes(upload.loaded)} di ${fmtBytes(upload.total)} (${pct}%)`;
}

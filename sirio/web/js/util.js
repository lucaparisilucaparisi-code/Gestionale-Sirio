// Utilità condivise: HTML sicuro, formattazione all'italiana, interpretazione di orari e ore.

/* ------------------------------------------------------------------ HTML */
export class Raw {
  constructor(s) { this.s = String(s); }
  toString() { return this.s; }
}
export const raw = (s) => new Raw(s);

const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
export function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) => ESC[c]);
}

function renderVal(v) {
  if (v === null || v === undefined || v === false) return '';
  if (v instanceof Raw) return v.s;
  if (Array.isArray(v)) return v.map(renderVal).join('');
  return esc(v);
}

/** Template HTML con escape automatico dei valori interpolati. */
export function html(strings, ...vals) {
  let out = strings[0];
  for (let i = 0; i < vals.length; i++) out += renderVal(vals[i]) + strings[i + 1];
  return new Raw(out);
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function setHTML(el, content) {
  el.innerHTML = content instanceof Raw ? content.s : renderVal(content);
}

/** Crea un elemento da una stringa HTML. */
export function fromHTML(content) {
  const t = document.createElement('template');
  t.innerHTML = (content instanceof Raw ? content.s : String(content)).trim();
  return t.content.firstElementChild;
}

/* ------------------------------------------------------------------ tempo */
export function debounce(fn, ms) {
  let t = null;
  const d = (...args) => {
    clearTimeout(t);
    t = setTimeout(() => { t = null; fn(...args); }, ms);
  };
  d.cancel = () => { clearTimeout(t); t = null; };
  d.flush = (...args) => { if (t !== null) { clearTimeout(t); t = null; fn(...args); } };
  d.pending = () => t !== null;
  return d;
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ------------------------------------------------------------------ numeri */
const nf2 = new Intl.NumberFormat('it-IT', { maximumFractionDigits: 2 });
const nf1 = new Intl.NumberFormat('it-IT', { maximumFractionDigits: 1 });
const nfInt = new Intl.NumberFormat('it-IT', { maximumFractionDigits: 0 });

/** Ore all'italiana: 1.5 -> "1,5"; 3 -> "3"; null -> "". */
export function fmtOre(x) {
  if (x === null || x === undefined || x === '' || !Number.isFinite(Number(x))) return '';
  const v = Math.round(Number(x) * 100) / 100 + 0;
  return nf2.format(Object.is(v, -0) ? 0 : v);
}
export function fmtOreLabel(x) {
  const t = fmtOre(x);
  if (!t) return '';
  return t === '1' ? '1 ora' : `${t} ore`;
}
export const fmtInt = (x) => nfInt.format(Number(x) || 0);
export const fmtDec1 = (x) => nf1.format(Number(x) || 0);
export function fmtSigned(x) {
  if (x === null || x === undefined || !Number.isFinite(Number(x))) return '';
  const v = Math.round(Number(x) * 100) / 100;
  if (Math.abs(v) < 0.005) return '0';
  return (v > 0 ? '+' : '−') + fmtOre(Math.abs(v));
}

export function fmtUSD(x) {
  const v = Number(x) || 0;
  const digits = v > 0 && v < 1 ? 3 : 2;
  return new Intl.NumberFormat('it-IT', {
    style: 'currency', currency: 'USD', currencyDisplay: 'narrowSymbol',
    minimumFractionDigits: 2, maximumFractionDigits: digits,
  }).format(v);
}

export function fmtBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${fmtInt(v)} byte`;
  if (v < 1024 * 1024) return `${fmtDec1(v / 1024)} KB`;
  return `${fmtDec1(v / 1024 / 1024)} MB`;
}

export function plural(n, one, many) {
  return `${fmtInt(n)} ${Number(n) === 1 ? one : many}`;
}

/* ------------------------------------------------------------------ date */
export const MESI = ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 'luglio', 'agosto',
  'settembre', 'ottobre', 'novembre', 'dicembre'];
export const GIORNI_BREVI = ['dom', 'lun', 'mar', 'mer', 'gio', 'ven', 'sab'];

export function fmtPeriodo(mese, anno, fallback = '—') {
  if (mese >= 1 && mese <= 12 && anno) return `${MESI[mese - 1]} ${anno}`;
  if (anno) return String(anno);
  return fallback;
}
export function capitalize(s) {
  const t = String(s ?? '');
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function fmtDateTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const now = new Date();
  const time = d.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' });
  if (d.toDateString() === now.toDateString()) return `oggi, ${time}`;
  const y = new Date(now); y.setDate(now.getDate() - 1);
  if (d.toDateString() === y.toDateString()) return `ieri, ${time}`;
  const date = d.toLocaleDateString('it-IT', { day: 'numeric', month: 'short', year: d.getFullYear() === now.getFullYear() ? undefined : 'numeric' });
  return `${date}, ${time}`;
}
export function fmtTime(d = new Date()) {
  return d.toLocaleTimeString('it-IT', { hour: '2-digit', minute: '2-digit' });
}
export function fmtDateISO(iso) {
  if (!iso) return '';
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  return m ? `${m[3]}/${m[2]}/${m[1]}` : iso;
}

/* ------------------------------------------------------------------ orari */
const DASH_RE = /^[\s\-–—−_]+$/;
export const DASH = '-';

/**
 * Interpreta un orario digitato: "8", "8.30", "830", "8:30", "08,30", "8h30" -> "08:30".
 * Restituisce {ok, value, dash, error}. Vuoto -> value null; trattino -> dash true.
 */
export function parseTime(input) {
  const s0 = String(input ?? '').trim();
  if (!s0) return { ok: true, value: null, dash: false };
  if (DASH_RE.test(s0)) return { ok: true, value: null, dash: true };
  let s = s0.toLowerCase().replace(/^ore\s*/, '').replace(/\s+/g, '');
  s = s.replace(/[.,;h']/g, ':').replace(/:+$/, '');
  let h; let m;
  let mm;
  if ((mm = /^(\d{1,2})$/.exec(s))) { h = +mm[1]; m = 0; }
  else if ((mm = /^(\d{1,2})(\d{2})$/.exec(s))) { h = +mm[1]; m = +mm[2]; }
  else if ((mm = /^(\d{1,2}):(\d{1,2})$/.exec(s))) { h = +mm[1]; m = mm[2].length === 1 ? +mm[2] * 10 : +mm[2]; }
  else return { ok: false, error: 'Orario non valido: scrivere ad esempio 8:30 (oppure 8.30, 830, 8).' };
  if (h > 23 || m > 59) return { ok: false, error: `Orario non valido (${s0}): ore da 0 a 23, minuti da 00 a 59.` };
  return { ok: true, value: `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`, dash: false };
}

/** Interpreta un numero di ore: "1,5", "1.5", "3", "2:30" (=2,5), "3h". */
export function parseHours(input, max = 24) {
  const s0 = String(input ?? '').trim();
  if (!s0 || DASH_RE.test(s0)) return { ok: true, value: null };
  let s = s0.toLowerCase().replace(/\s*(ore|ora|h)$/, '').replace(/\s+/g, '');
  let v = null;
  let mm;
  if ((mm = /^(\d{1,3}):(\d{2})$/.exec(s))) {
    if (+mm[2] < 60) v = +mm[1] + (+mm[2]) / 60;
  } else {
    s = s.replace(',', '.');
    if (/^\d+(\.\d+)?$|^\.\d+$/.test(s)) v = parseFloat(s);
  }
  if (v === null || !Number.isFinite(v)) return { ok: false, error: 'Valore non valido: scrivere le ore con la virgola, ad esempio 3 oppure 1,5.' };
  if (v < 0 || v > max) return { ok: false, error: `Valore fuori intervallo: indicare un numero di ore tra 0 e ${fmtOre(max)}.` };
  return { ok: true, value: Math.round(v * 10000) / 10000 };
}

export function timeToMinutes(s) {
  const m = /^(\d{1,2}):(\d{2})$/.exec(String(s ?? ''));
  if (!m) return null;
  const h = +m[1]; const mi = +m[2];
  if (h > 24 || mi > 59) return null;
  return h * 60 + mi;
}

/** Ore tra entrata e uscita (null se mancanti, non valide o non positive). */
export function hoursBetween(a, b) {
  const x = timeToMinutes(a); const y = timeToMinutes(b);
  if (x === null || y === null || y <= x) return null;
  return Math.round(((y - x) / 60) * 100) / 100;
}

/* ------------------------------------------------------------------ varie */
export function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
export function uid() { return Math.random().toString(36).slice(2, 10); }

export function normalizeText(s) {
  return String(s ?? '').toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
}

export function colLetter(i) {
  let n = i + 1; let s = '';
  while (n > 0) { const r = (n - 1) % 26; s = String.fromCharCode(65 + r) + s; n = Math.floor((n - 1) / 26); }
  return s;
}

export function storageGet(key, fallback = null) {
  try {
    const v = localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch { return fallback; }
}
export function storageSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* archiviazione non disponibile */ }
}

export function isTypingTarget(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

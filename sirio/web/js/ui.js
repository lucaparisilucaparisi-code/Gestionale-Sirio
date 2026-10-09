// Componenti d'interfaccia globali: notifiche, finestre di conferma, suggerimenti, menu.

import { icon } from './icons.js';
import { esc, html, raw, Raw } from './util.js';

/* ================================================================== toast */
const TOAST_ICONS = { success: 'check-circle', error: 'x-circle', warning: 'alert-triangle', info: 'info' };

/**
 * Mostra una notifica.
 * @param {{type?: 'success'|'error'|'warning'|'info', title: string, message?: string|Raw,
 *          actions?: {label: string, onClick: Function}[], duration?: number}} opts
 */
export function toast(opts) {
  const { type = 'info', title, message = '', actions = [], duration } = opts;
  const host = document.getElementById('toasts');
  if (!host) return () => {};
  const ms = duration ?? (type === 'error' ? 9000 : type === 'warning' ? 7000 : 4500);
  const el = document.createElement('div');
  el.className = `toast is-${type}`;
  el.setAttribute('role', type === 'error' ? 'alert' : 'status');
  const msg = message instanceof Raw ? message.s : esc(message);
  el.innerHTML = `
    <div class="toast-icon">${icon(TOAST_ICONS[type] || 'info')}</div>
    <div class="toast-title">${esc(title)}</div>
    <button class="toast-close" type="button" aria-label="Chiudi notifica">${icon('x', 'icon-sm')}</button>
    ${msg ? `<div class="toast-text">${msg}</div>` : ''}
    ${actions.length ? `<div class="toast-actions">${actions.map((a, i) => `<button type="button" class="link-btn" data-i="${i}">${esc(a.label)}</button>`).join('')}</div>` : ''}
    ${ms > 0 ? `<div class="toast-timer" style="animation-duration:${ms}ms"></div>` : ''}`;
  let closed = false;
  let timer = null;
  const close = () => {
    if (closed) return;
    closed = true;
    clearTimeout(timer);
    el.classList.add('is-leaving');
    el.addEventListener('animationend', () => el.remove(), { once: true });
    setTimeout(() => el.remove(), 400);
  };
  el.querySelector('.toast-close').addEventListener('click', close);
  el.querySelectorAll('.toast-actions button').forEach((b) => {
    b.addEventListener('click', () => { const a = actions[Number(b.dataset.i)]; close(); a?.onClick?.(); });
  });
  let remaining = ms;
  let started = Date.now();
  const arm = () => { if (ms > 0) timer = setTimeout(close, remaining); };
  el.addEventListener('mouseenter', () => { clearTimeout(timer); remaining -= Date.now() - started; });
  el.addEventListener('mouseleave', () => { started = Date.now(); arm(); });
  host.appendChild(el);
  while (host.children.length > 4) host.firstElementChild.remove();
  arm();
  return close;
}

export const toastError = (err, title = 'Operazione non riuscita') =>
  toast({ type: 'error', title, message: err?.message || String(err || 'Errore sconosciuto.') });

/* ================================================================== dialog */
const DIALOG_ICONS = { info: 'info', danger: 'alert-triangle', warning: 'alert-triangle', success: 'check-circle', question: 'help' };
let dialogBusy = Promise.resolve();

/**
 * Finestra modale personalizzata (sostituisce window.confirm).
 * Restituisce una Promise con il valore del pulsante scelto (o null se annullata).
 */
export function dialog({ title, message = '', tone = 'info', icon: ic, buttons, wide = false, onOpen }) {
  const run = () => new Promise((resolve) => {
    const dlg = document.getElementById('dialog');
    const btns = buttons || [{ label: 'OK', value: true, kind: 'primary' }];
    const msg = message instanceof Raw ? message.s : `<p>${esc(message)}</p>`;
    dlg.className = `dialog is-${tone}${wide ? ' is-wide' : ''}`;
    dlg.innerHTML = `
      <form method="dialog">
        <div class="dialog-body">
          <div class="dialog-icon">${icon(ic || DIALOG_ICONS[tone] || 'info')}</div>
          <h2 class="dialog-title" id="dialog-title">${esc(title)}</h2>
          <div class="dialog-text">${msg}</div>
        </div>
        <div class="dialog-foot">
          ${btns.map((b, i) => `<button type="button" class="btn ${b.kind === 'primary' ? 'btn-primary' : b.kind === 'danger' ? 'btn-danger' : b.kind === 'success' ? 'btn-success' : ''}" data-i="${i}">${esc(b.label)}</button>`).join('')}
        </div>
      </form>`;
    let result = null;
    const finish = () => {
      dlg.removeEventListener('close', finish);
      resolve(result);
    };
    dlg.addEventListener('close', finish);
    dlg.querySelectorAll('.dialog-foot button').forEach((b) => {
      b.addEventListener('click', () => { result = btns[Number(b.dataset.i)].value; dlg.close(); });
    });
    dlg.addEventListener('cancel', () => { result = null; }, { once: true });
    dlg.showModal();
    const focusBtn = dlg.querySelector(`.dialog-foot button[data-i="${btns.findIndex((b) => b.autofocus) >= 0 ? btns.findIndex((b) => b.autofocus) : btns.length - 1}"]`);
    focusBtn?.focus();
    onOpen?.(dlg);
  });
  const p = dialogBusy.then(run);
  dialogBusy = p.catch(() => {});
  return p;
}

/** Conferma sì/no; risolve true/false. */
export async function confirmDialog({ title, message, confirmLabel = 'Conferma', cancelLabel = 'Annulla', tone = 'question', danger = false }) {
  const res = await dialog({
    title,
    message,
    tone: danger ? 'danger' : tone,
    buttons: [
      { label: cancelLabel, value: false },
      { label: confirmLabel, value: true, kind: danger ? 'danger' : 'primary' },
    ],
  });
  return res === true;
}

/* ================================================================== tooltip */
let ttTarget = null;
let ttTimer = null;

function placeTooltip(tt, target) {
  const r = target.getBoundingClientRect();
  const tr = tt.getBoundingClientRect();
  const margin = 8;
  let top = r.top - tr.height - margin;
  let below = false;
  if (top < 8) { top = r.bottom + margin; below = true; }
  let left = r.left + r.width / 2 - tr.width / 2;
  left = Math.max(8, Math.min(window.innerWidth - tr.width - 8, left));
  if (below && top + tr.height > window.innerHeight - 8) top = Math.max(8, r.top - tr.height - margin);
  tt.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
}

function showTooltip(target) {
  const tt = document.getElementById('tooltip');
  if (!tt || !target.isConnected) return;
  const text = target.getAttribute('data-tip');
  if (!text) return;
  const title = target.getAttribute('data-tip-title');
  tt.className = `tooltip${target.dataset.tipTone ? ' is-' + target.dataset.tipTone : ''}`;
  tt.innerHTML = (title ? `<span class="tt-title">${esc(title)}</span>` : '') + esc(text);
  tt.style.transform = 'translate(-9999px, -9999px)';
  tt.classList.add('is-visible');
  placeTooltip(tt, target);
}

export function hideTooltip() {
  clearTimeout(ttTimer);
  ttTarget = null;
  const tt = document.getElementById('tooltip');
  if (tt) tt.classList.remove('is-visible');
}

export function initTooltips() {
  document.addEventListener('mouseover', (e) => {
    const t = e.target.closest?.('[data-tip]');
    if (t === ttTarget) return;
    hideTooltip();
    if (!t) return;
    ttTarget = t;
    ttTimer = setTimeout(() => showTooltip(t), t.closest('.sheet') ? 450 : 320);
  });
  document.addEventListener('mousedown', hideTooltip, true);
  document.addEventListener('wheel', hideTooltip, { passive: true, capture: true });
  document.addEventListener('keydown', hideTooltip, true);
  document.addEventListener('focusin', (e) => {
    const t = e.target.closest?.('[data-tip]');
    if (!t || !e.target.matches(':focus-visible') || t.closest('.sheet')) return;
    hideTooltip();
    ttTarget = t;
    ttTimer = setTimeout(() => showTooltip(t), 300);
  });
  document.addEventListener('focusout', hideTooltip);
}

/* ================================================================== menu */
let openMenu = null;

export function closeMenu() {
  if (openMenu) { openMenu.remove(); openMenu = null; }
}

/**
 * Menu a comparsa ancorato a un elemento.
 * items: [{label, icon, onClick, danger}] oppure 'sep'.
 */
export function showMenu(anchor, items) {
  closeMenu();
  const el = document.createElement('div');
  el.className = 'popover';
  el.setAttribute('role', 'menu');
  el.innerHTML = items.map((it, i) => it === 'sep'
    ? '<div class="menu-sep" role="separator"></div>'
    : `<button type="button" role="menuitem" class="menu-item${it.danger ? ' is-danger' : ''}" data-i="${i}">${it.icon ? icon(it.icon) : ''}<span>${esc(it.label)}</span></button>`).join('');
  document.body.appendChild(el);
  const r = anchor.getBoundingClientRect();
  const w = el.offsetWidth; const h = el.offsetHeight;
  let left = r.right - w;
  if (left < 8) left = r.left;
  let top = r.bottom + 6;
  if (top + h > window.innerHeight - 8) top = r.top - h - 6;
  el.style.left = `${Math.max(8, left)}px`;
  el.style.top = `${Math.max(8, top)}px`;
  el.addEventListener('click', (e) => {
    const b = e.target.closest('.menu-item');
    if (!b) return;
    const it = items[Number(b.dataset.i)];
    closeMenu();
    it?.onClick?.();
  });
  el.addEventListener('keydown', (e) => {
    const btns = Array.from(el.querySelectorAll('.menu-item'));
    const i = btns.indexOf(document.activeElement);
    if (e.key === 'ArrowDown') { e.preventDefault(); btns[(i + 1) % btns.length]?.focus(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); btns[(i - 1 + btns.length) % btns.length]?.focus(); }
    else if (e.key === 'Escape') { e.preventDefault(); closeMenu(); anchor.focus(); }
  });
  openMenu = el;
  el.querySelector('.menu-item')?.focus();
  return el;
}

export function initMenus() {
  document.addEventListener('mousedown', (e) => {
    if (openMenu && !openMenu.contains(e.target)) closeMenu();
  });
  window.addEventListener('resize', closeMenu);
  window.addEventListener('blur', closeMenu);
}

/* ================================================================== piccoli componenti */
const STATUS_INFO = {
  in_coda: { label: 'In coda', tone: 'neutral' },
  in_lavorazione: { label: 'In lavorazione', tone: 'info', busy: true },
  errore: { label: 'Errore', tone: 'err' },
  scartato: { label: 'Scartato', tone: 'neutral' },
  ok: { label: 'OK', tone: 'ok' },
  da_verificare: { label: 'Da verificare', tone: 'warn' },
  errori: { label: 'Errori', tone: 'err' },
};

/** Chiave di stato visualizzata per un documento (riepilogo o completo). */
export function statusKey(doc) {
  if (!doc) return 'in_coda';
  if (doc.status !== 'completato') return doc.status;
  if (doc.is_foglio_firma === false) return 'scartato';
  return doc.totals?.stato || 'ok';
}

export function statusBadge(doc, small = false) {
  const key = statusKey(doc);
  const info = STATUS_INFO[key] || STATUS_INFO.in_coda;
  let label = info.label;
  if (key === 'in_lavorazione' && doc.progress) label = `${info.label} · ${Math.round(doc.progress * 100)}%`;
  return html`<span class="badge${small ? ' badge-sm' : ''} tone-${info.tone}${info.busy ? ' is-busy' : ''}">${label}</span>`;
}
export const statusLabel = (key) => (STATUS_INFO[key] || STATUS_INFO.in_coda).label;

export function skeletonRows(n, cols) {
  return raw(Array.from({ length: n }, () => `<tr>${cols.map((w) => `<td><div class="skeleton" style="width:${w};height:12px"></div></td>`).join('')}</tr>`).join(''));
}

export function emptyState({ art = '', title, text = '', action = '' }) {
  return html`<div class="empty">${raw(art)}<div class="empty-title">${title}</div>${text ? html`<p class="empty-text">${text}</p>` : ''}${action}</div>`;
}

// Utilità comuni agli script Playwright dell'interfaccia di Sirio OCR.
//
// Avviano il server di sviluppo (motore OCR finto, documenti sintetici con nomi fittizi)
// oppure usano un server già avviato indicato da SIRIO_UI_URL (+ SIRIO_UI_TOKEN).
//
// Variabili d'ambiente:
//   SIRIO_UI_URL / SIRIO_UI_TOKEN  server esistente (facoltativo)
//   SIRIO_UI_DATA                  cartella in cui creare i dati temporanei (predefinita: temp di sistema)
//   SIRIO_UI_SHOTS                 cartella delle schermate (predefinita: <SIRIO_UI_DATA>/ui-shots)
//   SIRIO_PYTHON                   interprete Python (predefinito: .venv/bin/python del repository)
//   CHROMIUM_PATH                  eseguibile di Chromium (facoltativo)

import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

// Playwright installato globalmente: gli import ESM ignorano NODE_PATH, quindi si ricorre a require.
async function loadPlaywright() {
  try {
    return await import('playwright');
  } catch {
    const require = createRequire(import.meta.url);
    for (const candidate of ['playwright', ...(process.env.NODE_PATH || '').split(path.delimiter).filter(Boolean).map((p) => path.join(p, 'playwright')), '/opt/node22/lib/node_modules/playwright']) {
      try { return require(candidate); } catch { /* prossimo */ }
    }
    throw new Error('Playwright non trovato: installarlo (npm i -g playwright) e impostare NODE_PATH.');
  }
}
const { chromium } = await loadPlaywright();

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export function workDir(prefix = 'sirio-ui-') {
  const base = process.env.SIRIO_UI_DATA || tmpdir();
  mkdirSync(base, { recursive: true });
  return mkdtempSync(path.join(base, prefix));
}

export function shotsDir() {
  const dir = process.env.SIRIO_UI_SHOTS || path.join(process.env.SIRIO_UI_DATA || tmpdir(), 'ui-shots');
  mkdirSync(dir, { recursive: true });
  return dir;
}

/** Avvia tests/dev_server.py e attende che risponda. */
export async function startServer({ documents = 6, delay = 1.2, concurrency = 2, extraArgs = [] } = {}) {
  if (process.env.SIRIO_UI_URL) {
    return { url: process.env.SIRIO_UI_URL.replace(/\/?$/, '/'), token: process.env.SIRIO_UI_TOKEN || '', stop: async () => {} };
  }
  const port = 8800 + Math.floor(Math.random() * 900);
  const token = `ui-${Math.random().toString(36).slice(2)}${Math.random().toString(36).slice(2)}`;
  const dataDir = workDir('dati-');
  const python = process.env.SIRIO_PYTHON || path.join(ROOT, '.venv', 'bin', 'python');
  const proc = spawn(python, [
    'tests/dev_server.py', '--porta', String(port), '--documenti', String(documents), '--ritardo', String(delay),
    '--concorrenza', String(concurrency), '--token', token, '--cartella-dati', dataDir, '--log-level', 'WARNING', ...extraArgs,
  ], { cwd: ROOT, stdio: ['ignore', 'pipe', 'pipe'], env: { ...process.env, SIRIO_EXPORT_DIR: path.join(dataDir, 'export') } });
  let log = '';
  proc.stdout.on('data', (d) => { log += d; });
  proc.stderr.on('data', (d) => { log += d; });
  const url = `http://127.0.0.1:${port}/`;
  for (let i = 0; i < 150; i++) {
    if (proc.exitCode !== null) throw new Error(`Il server di sviluppo è terminato:\n${log}`);
    try {
      const r = await fetch(`${url}api/health`);
      if (r.ok) break;
    } catch { /* non ancora pronto */ }
    await sleep(200);
  }
  return {
    url, token, dataDir, proc,
    log: () => log,
    stop: async () => {
      if (proc.exitCode !== null) return;
      proc.kill('SIGINT');
      for (let i = 0; i < 40 && proc.exitCode === null; i++) await sleep(100);
      if (proc.exitCode === null) proc.kill('SIGKILL');
    },
  };
}

export async function api(server, pathname, init = {}) {
  const res = await fetch(new URL(pathname, server.url), {
    ...init,
    headers: { 'X-Sirio-Token': server.token, 'Content-Type': 'application/json', ...(init.headers || {}) },
  });
  if (!res.ok) throw new Error(`${pathname}: HTTP ${res.status} ${await res.text()}`);
  return res.json();
}

/** Attende la fine delle letture (nessun documento in coda o in lavorazione). */
export async function waitIdle(server, { min = 1, timeout = 60000 } = {}) {
  const t0 = Date.now();
  for (;;) {
    const s = await api(server, '/api/state');
    const q = s.queue || {};
    if ((s.documents || []).length >= min && !q.in_coda && !q.in_lavorazione) return s;
    if (Date.now() - t0 > timeout) throw new Error('Timeout in attesa della fine delle letture');
    await sleep(400);
  }
}

export async function launch() {
  const opts = { headless: true };
  if (process.env.CHROMIUM_PATH) opts.executablePath = process.env.CHROMIUM_PATH;
  try {
    return await chromium.launch(opts);
  } catch (err) {
    if (opts.executablePath) throw err;
    return chromium.launch({ ...opts, executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
  }
}

/** Raccoglie errori della console e della pagina (devono restare zero). */
export function watchConsole(page, label = '') {
  const errors = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(`${label}console: ${msg.text()}`);
  });
  page.on('pageerror', (err) => errors.push(`${label}pageerror: ${err.message}`));
  page.on('requestfailed', (req) => {
    const f = req.failure()?.errorText || '';
    if (!f.includes('ERR_ABORTED')) errors.push(`${label}requestfailed: ${req.url()} ${f}`);
  });
  return errors;
}

export async function setTheme(page, theme) {
  await page.evaluate((t) => {
    localStorage.setItem('sirio.tema', t);
    document.documentElement.setAttribute('data-theme', t === 'scuro' ? 'dark' : 'light');
  }, theme);
}

export function check(cond, message) {
  if (!cond) throw new Error(`Verifica fallita: ${message}`);
  console.log(`  ✓ ${message}`);
}

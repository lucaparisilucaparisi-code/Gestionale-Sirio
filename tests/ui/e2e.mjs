// Prova completa dell'interfaccia contro il server reale (motore OCR finto, dati sintetici):
//   (a) avvio, heartbeat, navigazione;
//   (b) griglia di revisione: modifica di un orario, cella illeggibile, casella, annulla,
//       valore non valido, intestazione, F8, conferma del documento;
//   (c) importazione di un PDF sintetico tramite trascinamento e tramite selezione file;
//   (d) anteprima Excel: modifica del dettaglio, generazione, scheda di esito, download;
//   (e) nessun errore nella console del browser.
//
//   NODE_PATH=/opt/node22/lib/node_modules node tests/ui/e2e.mjs

import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { ROOT, api, check, launch, shotsDir, sleep, startServer, waitIdle, watchConsole, workDir } from './_helpers.mjs';

const server = await startServer({ documents: 4, delay: 0.6 });
const browser = await launch();
const errors = [];
const out = shotsDir();

function makeSyntheticFiles() {
  const dir = workDir('file-');
  const python = process.env.SIRIO_PYTHON || path.join(ROOT, '.venv', 'bin', 'python');
  const script = `
import sys
sys.path.insert(0, ${JSON.stringify(ROOT)})
import cv2
from tests.dev_server import demo_data
from tests.synthetic import make_synthetic_sheet, sheet_to_pdf
img, _ = make_synthetic_sheet(demo_data(0), seed=77, skew_deg=0.3)
open(${JSON.stringify(path.join(dir, 'foglio_firma_01_rossi.pdf'))}, 'wb').write(sheet_to_pdf([img]))
img2, _ = make_synthetic_sheet(demo_data(5), seed=78)
cv2.imwrite(${JSON.stringify(path.join(dir, 'scansione_nuova.png'))}, img2)
`;
  execFileSync(python, ['-c', script], { cwd: ROOT, stdio: 'inherit' });
  return { pdf: path.join(dir, 'foglio_firma_01_rossi.pdf'), png: path.join(dir, 'scansione_nuova.png') };
}

const cell = (page, giorno, key) => page.locator(`#rv-grid tr[data-r="${giorno - 1}"] td[data-c="${COLS.indexOf(key)}"]`);
const COLS = ['giorno', 'gg', 'prog_entrata', 'prog_uscita', 'eff_entrata', 'eff_uscita', 'ore_dichiarate', 'ore_calcolate',
  'assenza_alunno', 'assenza_operatore', 'firma', 'note', 'esito'];

function waitPut(page, id) {
  return page.waitForResponse((r) => r.url().includes(`/api/documents/${id}`) && r.request().method() === 'PUT', { timeout: 10000 });
}

try {
  const state = await waitIdle(server, { min: 4 });
  const docs = state.documents.filter((d) => d.status === 'completato' && d.is_foglio_firma);
  const doc = docs.find((d) => d.header.operatore === 'ROSSI MARIO');
  check(!!doc, 'documento sintetico ROSSI MARIO letto');

  const ctx = await browser.newContext({ viewport: { width: 1480, height: 940 }, acceptDownloads: true });
  const page = await ctx.newPage();
  errors.push(...watchConsole(page));

  // ------------------------------------------------------------------ (a) avvio
  const beat = page.waitForRequest((r) => r.url().endsWith('/api/heartbeat') && r.method() === 'POST');
  await page.goto(server.url);
  await page.waitForSelector('#splash', { state: 'detached' });
  await beat;
  check(true, 'heartbeat inviato all\'avvio');
  check(await page.locator('.kpi').count() === 6, 'sei indicatori nella dashboard');
  const tokenOk = await page.evaluate(() => document.querySelector('meta[name="sirio-token"]').content !== '__SIRIO_TOKEN__');
  check(tokenOk, 'token di sessione inserito dal server');

  // ricerca globale
  await page.keyboard.press('Control+k');
  await page.keyboard.type('rossi');
  await page.waitForSelector('.search-item');
  check(await page.locator('.search-item').count() >= 1, 'la ricerca globale trova il documento');
  await page.keyboard.press('Escape');

  // ------------------------------------------------------------------ (b) revisione
  await page.goto(`${server.url}#/revisione/${doc.id}`);
  await page.waitForSelector('#rv-grid td.is-active');
  await page.waitForTimeout(400);
  const full0 = await api(server, `/api/documents/${doc.id}`);
  const day4 = full0.rows[3];
  check(day4.eff_uscita === '11:00' || day4.eff_uscita === '14:00', 'giorno 4 con uscita effettiva letta');

  // 1. modifica di un orario digitando "1130"
  await cell(page, 4, 'eff_uscita').click();
  let put = waitPut(page, doc.id);
  await page.keyboard.type('1130');
  await page.keyboard.press('Enter');
  let res = await put;
  check(res.ok(), 'PUT inviato dopo la modifica dell\'orario');
  await page.waitForTimeout(150);
  const c4 = cell(page, 4, 'eff_uscita');
  check((await c4.textContent()).trim() === '11:30', 'orario normalizzato in 11:30');
  check(await c4.evaluate((el) => el.classList.contains('st-corretto')), 'cella modificata evidenziata in blu');
  check((await c4.getAttribute('data-tip') || '').includes('Valore OCR originale'), 'suggerimento con il valore OCR originale');
  check((await page.locator('#rv-save').textContent()).includes('Salvato'), 'indicatore «Salvato»');
  await page.screenshot({ path: path.join(out, 'e2e-01-modifica-orario.png') });

  // 2. cella illeggibile compilata
  const illegCell = page.locator('#rv-grid td.st-illeggibile').first();
  check(await illegCell.count() === 1, 'una cella illeggibile segnalata');
  const illegRow = Number(await illegCell.evaluate((el) => el.parentElement.dataset.r)) + 1;
  const illegKey = COLS[Number(await illegCell.getAttribute('data-c'))];
  await illegCell.click();
  put = waitPut(page, doc.id);
  await page.keyboard.type('104');
  await page.keyboard.press('Enter');
  await put;
  await page.waitForTimeout(150);
  const filled = cell(page, illegRow, illegKey);
  check(!(await filled.evaluate((el) => el.classList.contains('st-illeggibile'))), 'la cella non è più illeggibile');
  check((await filled.textContent()).trim() === '104', 'valore inserito nella cella illeggibile');

  // 3. casella: assenza alunno del giorno 5, poi annulla con Ctrl+Z
  put = waitPut(page, doc.id);
  await cell(page, 5, 'assenza_alunno').locator('.cb').click();
  await put;
  check(await cell(page, 5, 'assenza_alunno').locator('.cb.is-on').count() === 1, 'casella spuntata con un clic');
  put = waitPut(page, doc.id);
  await page.keyboard.press('Control+z');
  await put;
  await page.waitForTimeout(150);
  check(await cell(page, 5, 'assenza_alunno').locator('.cb.is-on').count() === 0, 'Ctrl+Z annulla la spunta');

  // 4. firma del giorno 6 con la barra spaziatrice
  await cell(page, 6, 'firma').click({ position: { x: 4, y: 4 } });
  const firmaBefore = await cell(page, 6, 'firma').locator('.cb.is-on').count();
  put = waitPut(page, doc.id);
  await page.keyboard.press(' ');
  await put;
  check(await cell(page, 6, 'firma').locator('.cb.is-on').count() !== firmaBefore, 'Spazio cambia la casella della firma');

  // 5. valore non valido: l'editor resta aperto con il messaggio d'errore
  await cell(page, 6, 'prog_entrata').click();
  await page.keyboard.type('25:99');
  await page.keyboard.press('Enter');
  await page.waitForSelector('.cell-error');
  check(await page.locator('.cell-editor.is-invalid').count() === 1, 'orario non valido rifiutato con messaggio');
  await page.keyboard.press('Escape');
  check(await page.locator('.cell-editor').count() === 0, 'Esc annulla la modifica');

  // 6. ore con la virgola
  await cell(page, 6, 'ore_dichiarate').click();
  put = waitPut(page, doc.id);
  await page.keyboard.type('2,5');
  await page.keyboard.press('Tab');
  await put;
  check((await cell(page, 6, 'ore_dichiarate').textContent()).trim() === '2,5', 'ore «2,5» accettate e mostrate all\'italiana');

  // 6b. copia e incolla (anche un blocco di due celle separate da tabulazione)
  await cell(page, 9, 'prog_entrata').click();
  const copied = await page.evaluate(() => {
    const dt = new DataTransfer();
    document.dispatchEvent(new ClipboardEvent('copy', { clipboardData: dt, bubbles: true, cancelable: true }));
    return dt.getData('text/plain');
  });
  check(copied === '08:00' || copied === '11:00', `Ctrl+C copia il valore della cella («${copied}»)`);
  await cell(page, 7, 'eff_entrata').click();
  put = waitPut(page, doc.id);
  await page.evaluate(() => {
    const dt = new DataTransfer();
    dt.setData('text/plain', '9.15\t1230');
    document.dispatchEvent(new ClipboardEvent('paste', { clipboardData: dt, bubbles: true, cancelable: true }));
  });
  await put;
  check((await cell(page, 7, 'eff_entrata').textContent()).trim() === '09:15'
    && (await cell(page, 7, 'eff_uscita').textContent()).trim() === '12:30', 'Ctrl+V incolla un blocco di celle (orari normalizzati)');
  put = waitPut(page, doc.id);
  await page.keyboard.press('Control+z');
  await put;
  await page.waitForTimeout(150);
  check((await cell(page, 7, 'eff_entrata').textContent()).trim() === '', 'Ctrl+Z annulla l\'incolla in un solo passo');

  // 7. intestazione
  put = waitPut(page, doc.id);
  await page.fill('#hf-istituto', 'IC 1 VERDI - PLESSO NORD');
  await page.keyboard.press('Enter');
  await put;
  check(await page.locator('#hf-istituto.is-edited').count() === 1, 'campo d\'intestazione modificato evidenziato');

  // 8. F8: prossimo campo da verificare
  await page.locator('#rv-grid').focus();
  await page.keyboard.press('F8');
  await page.waitForTimeout(200);
  const focusedFlag = await page.evaluate(() => {
    const td = document.querySelector('#rv-grid td.is-active');
    const a = document.activeElement;
    return (td && (td.classList.contains('st-incerto') || td.classList.contains('st-illeggibile'))) || (a && a.classList.contains('is-uncert'));
  });
  check(focusedFlag, 'F8 porta al prossimo campo incerto o illeggibile');

  // 9. conferma del documento
  await page.click('#rv-confirm');
  const dlg = page.locator('dialog[open]');
  if (await dlg.count()) {
    await dlg.locator('.btn-primary, .btn-danger').first().click();
  }
  await page.waitForFunction(() => document.querySelector('#rv-confirm')?.textContent.includes('Confermato'), null, { timeout: 8000 });
  check(true, 'documento confermato («Confermato»)');
  await page.screenshot({ path: path.join(out, 'e2e-02-documento-confermato.png') });

  const full = await api(server, `/api/documents/${doc.id}`);
  check(full.user_verified === true, 'API: user_verified = true');
  check(full.rows[3].eff_uscita === '11:30', 'API: orario del giorno 4 salvato');
  check(full.rows[illegRow - 1][illegKey] === '104', 'API: valore della cella illeggibile salvato');
  check(full.rows[5].ore_dichiarate === 2.5, 'API: ore del giorno 6 salvate (2,5)');
  check(full.user_edited.includes('rows.4.eff_uscita'), 'API: modifica tracciata in user_edited');
  check(full.ocr_originali['rows.4.eff_uscita'] === day4.eff_uscita, 'API: valore OCR originale conservato');
  check(full.header.istituto === 'IC 1 VERDI - PLESSO NORD', 'API: intestazione salvata');
  check(!full.rows[4].assenza_alunno, 'API: annulla applicato (assenza alunno non spuntata)');

  // navigazione al documento successivo con Alt+→
  await page.locator('#rv-grid').focus();
  await page.keyboard.press('Alt+ArrowRight');
  await page.waitForFunction((id) => !location.hash.includes(id), doc.id, { timeout: 5000 });
  check(true, 'Alt+→ apre il documento successivo');

  // ------------------------------------------------------------------ (c) importazione
  const files = makeSyntheticFiles();
  await page.goto(`${server.url}#/dashboard`);
  await page.waitForSelector('#dropzone');
  const before = (await api(server, '/api/state')).documents.length;
  const b64 = readFileSync(files.pdf).toString('base64');
  const upload1 = page.waitForResponse((r) => r.url().endsWith('/api/upload') && r.request().method() === 'POST');
  await page.evaluate(async ({ data, name }) => {
    const bytes = Uint8Array.from(atob(data), (ch) => ch.charCodeAt(0));
    const dt = new DataTransfer();
    dt.items.add(new File([bytes], name, { type: 'application/pdf' }));
    const zone = document.getElementById('dropzone');
    for (const type of ['dragenter', 'dragover', 'drop']) {
      zone.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: dt }));
    }
  }, { data: b64, name: path.basename(files.pdf) });
  const up1 = await upload1;
  const body1 = await up1.json();
  check(up1.ok() && body1.documents.length === 1, 'PDF trascinato nell\'area di importazione e caricato');
  const newId = body1.documents[0].id;
  await page.waitForSelector(`.queue-item[data-open="${newId}"], .queue-item:has-text("ROSSI MARIO")`, { timeout: 20000 });
  for (let i = 0; i < 60; i++) {
    const d = (await api(server, '/api/state')).documents.find((x) => x.id === newId);
    if (d && d.status === 'completato') break;
    await sleep(300);
  }
  await page.waitForSelector(`.queue-item[data-open="${newId}"] .badge`, { timeout: 10000 });
  check(true, 'il documento importato compare nella coda come letto');
  await page.screenshot({ path: path.join(out, 'e2e-03-importazione.png') });

  const upload2 = page.waitForResponse((r) => r.url().endsWith('/api/upload') && r.request().method() === 'POST');
  await page.setInputFiles('#file-input', files.png);
  const up2 = await upload2;
  check(up2.ok(), 'immagine caricata tramite la selezione dei file');
  await waitIdle(server, { min: before + 2 });
  const after = (await api(server, '/api/state')).documents.length;
  check(after === before + 2, `documenti importati e letti (${before} → ${after})`);
  await page.waitForSelector('.toast.is-success');

  // ------------------------------------------------------------------ (d) anteprima ed Excel
  await page.goto(`${server.url}#/anteprima`);
  await page.waitForSelector('#xl-grid tbody tr');
  await page.click('[data-tab="dettaglio"]');
  await page.waitForSelector('#xl-grid tbody tr');
  // modifica della nota della prima riga di dettaglio
  const noteCol = await page.$$eval('#xl-grid thead tr.h-labels th', (ths) => ths.findIndex((th) => th.textContent.trim() === 'Note'));
  check(noteCol > 0, 'colonna Note presente nel dettaglio');
  const firstNote = page.locator(`#xl-grid tbody tr[data-r="0"] td[data-c="${noteCol}"]`);
  await firstNote.click();
  const putDet = page.waitForResponse((r) => r.url().includes('/api/documents/') && r.request().method() === 'PUT', { timeout: 10000 });
  await page.keyboard.type('NOTA DI PROVA');
  await page.keyboard.press('Enter');
  check((await putDet).ok(), 'modifica del dettaglio salvata nel documento (PUT)');
  await page.waitForFunction((c) => document.querySelector(`#xl-grid tbody tr[data-r="0"] td[data-c="${c}"]`)?.textContent.includes('NOTA DI PROVA'), noteCol, { timeout: 8000 });
  check(true, 'valore modificato visibile nell\'anteprima ricaricata');

  const exp = page.waitForResponse((r) => r.url().endsWith('/api/export') && r.request().method() === 'POST', { timeout: 60000 });
  await page.click('[data-act="generate"]');
  const expRes = await exp;
  const expBody = await expRes.json();
  check(expRes.ok() && expBody.filename.endsWith('.xlsx'), `Excel generato: ${expBody.filename}`);
  await page.waitForSelector('.export-done');
  check((await page.locator('.export-done-name').textContent()).trim() === expBody.filename, 'scheda di esito con il nome del file');
  await page.waitForSelector(`.export-item-name:has-text("${expBody.filename}")`);
  check(true, 'il file compare tra le esportazioni precedenti');
  const dl = page.waitForEvent('download');
  await page.click('.export-done [data-act="download"]');
  const download = await dl;
  check(download.suggestedFilename() === expBody.filename, 'download del file con il token (URL blob)');
  await page.screenshot({ path: path.join(out, 'e2e-04-excel-generato.png') });

  // ------------------------------------------------------------------ documenti: filtri e selezione
  await page.goto(`${server.url}#/documenti?stato=errori`);
  await page.waitForSelector('[data-filter="errori"][aria-pressed="true"]');
  const errRows = await page.locator('.table tbody tr').count();
  check(errRows >= 1, `filtro «Con errori» (${errRows} documenti)`);
  await page.click('[data-filter="tutti"]');
  await page.locator('.table tbody tr').first().locator('input[type="checkbox"]').check();
  await page.waitForSelector('.bulkbar');
  check(true, 'barra delle azioni di gruppo dopo la selezione');

  // impostazioni: anagrafica appresa dal documento confermato, aggiunta e rimozione di un nome
  await page.goto(`${server.url}#/impostazioni`);
  await page.waitForSelector('[data-theme-opt="scuro"]');
  const regVisible = await page.locator('#set-registry .reg-list').count();
  if (regVisible) {
    await page.waitForSelector('#set-registry .reg-name:has-text("ROSSI MARIO")', { timeout: 5000 });
    check(true, 'anagrafica: operatore appreso dal documento confermato');
    await page.click('[data-reg-tab="istituto"]');
    await page.fill('#reg-new', 'IC 99 PROVA');
    await page.click('[data-act="reg-add"]');
    await page.waitForSelector('#set-registry .reg-name:has-text("IC 99 PROVA")');
    check(true, 'anagrafica: nome aggiunto a mano');
    await page.click('[data-reg-del="IC 99 PROVA"]');
    await page.locator('dialog[open] .btn-danger').click();
    await page.waitForSelector('#set-registry .reg-name:has-text("IC 99 PROVA")', { state: 'detached' });
    check(true, 'anagrafica: nome rimosso');
  } else {
    console.log('  – anagrafica non disponibile sul server: verifica saltata');
  }

  // impostazioni: tema scuro e ritorno ad automatico
  await page.click('[data-theme-opt="scuro"]');
  await page.waitForFunction(() => document.documentElement.dataset.theme === 'dark');
  check(true, 'tema scuro applicato dalle impostazioni');
  await page.click('[data-theme-opt="auto"]');
  await page.waitForTimeout(400);

  await ctx.close();
  if (errors.length) {
    console.error(`\nErrori nella console del browser (${errors.length}):`);
    for (const e of errors) console.error(`  ${e}`);
    process.exitCode = 1;
  } else {
    console.log('  ✓ nessun errore nella console del browser');
    console.log('\nProva completa superata.');
  }
} catch (err) {
  console.error(err);
  console.error(server.log?.() || '');
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.stop();
}

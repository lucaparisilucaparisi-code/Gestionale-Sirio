// Prove di regressione dell'interfaccia (server di sviluppo, motore finto, dati sintetici):
//   1. cambio documento (Alt+←/→, pulsante «indietro» del mouse) con una cella o un campo
//      dell'intestazione ancora in modifica: la correzione va al documento giusto;
//   2. salvataggio automatico non riuscito: «Riprova» reinvia, un rifiuto del server annulla
//      la modifica, non si lascia il documento con correzioni non salvate, l'Excel non viene
//      generato senza le correzioni dell'anteprima;
//   3. chiusura della finestra subito dopo una correzione: la correzione viene salvata;
//   4. Documenti e Dashboard non tolgono il fuoco della tastiera durante gli aggiornamenti;
//   5. pagina scartata o in errore: la griglia non è modificabile dietro il riquadro;
//   6. Tab esce dalla griglia dall'ultima cella (e Maiusc+Tab dalla prima);
//   7. indicatore di fuoco sulla scansione;
//   8. tabella dei documenti senza scorrimento orizzontale a 1366 e 1280 pixel;
//   9. passaggio a Claude Vision solo dopo una conferma; nessun costo in Dashboard
//      con il motore locale.
//
//   NODE_PATH=/opt/node22/lib/node_modules node tests/ui/regressioni.mjs

import { api, launch, sleep, startServer, waitIdle } from './_helpers.mjs';

const server = await startServer({ documents: 6, delay: 1.2, concurrency: 1 });
const browser = await launch();
const pageErrors = [];

const COLS = ['giorno', 'gg', 'prog_entrata', 'prog_uscita', 'eff_entrata', 'eff_uscita', 'ore_dichiarate', 'ore_calcolate',
  'assenza_alunno', 'assenza_operatore', 'firma', 'note', 'esito'];
const cell = (page, giorno, key) => page.locator(`#rv-grid tr[data-r="${giorno - 1}"] td[data-c="${COLS.indexOf(key)}"]`);
const getDoc = (id) => api(server, `/api/documents/${id}`);
const PUT_DOC = /\/api\/documents\/[^/?]+$/;
const failures = [];

// le verifiche non interrompono la prova: ogni sezione prosegue e alla fine si elencano gli errori
function check(cond, message) {
  if (cond) console.log(`  ✓ ${message}`);
  else { console.log(`  ✗ ${message}`); failures.push(message); }
}

async function section(title, fn) {
  console.log(`\n${title}`);
  try { await fn(); } catch (err) { console.log(`  ✗ ${title}: ${err.message.split('\n')[0]}`); failures.push(`${title}: ${err.message.split('\n')[0]}`); }
}

async function until(fn, message, timeout = 8000) {
  const t0 = Date.now();
  for (;;) {
    let v;
    try { v = await fn(); } catch { v = false; }
    if (v) { check(true, message); return v; }
    if (Date.now() - t0 > timeout) { check(false, `${message} (tempo scaduto)`); return false; }
    await sleep(150);
  }
}

async function newPage(ctx) {
  const page = await ctx.newPage();
  // errori di script (le risorse non caricate sono volute nelle prove di salvataggio non riuscito)
  page.on('pageerror', (err) => pageErrors.push(`pageerror: ${err.message}`));
  page.on('console', (msg) => {
    if (msg.type() === 'error' && !/Failed to load resource/.test(msg.text())) pageErrors.push(`console: ${msg.text()}`);
  });
  return page;
}

async function openReview(page, id) {
  await page.goto(`${server.url}#/revisione/${id}`);
  await page.waitForFunction((x) => location.hash.includes(x), id);
  await page.waitForSelector('#rv-grid td.is-active');
  await page.waitForFunction(() => document.querySelector('#rv-loading')?.hidden);
  await page.waitForTimeout(250);
}

async function typeNote(page, giorno, text, { enter = true } = {}) {
  await cell(page, giorno, 'note').click();
  await page.keyboard.press('F2');
  await page.keyboard.press('Control+a');
  await page.keyboard.type(text);
  if (enter) await page.keyboard.press('Enter');
}

const saveText = (page) => page.locator('#rv-save').textContent();

try {
  const state = await waitIdle(server, { min: 8, timeout: 90000 });
  const all = state.documents;
  const isFF = (d) => d.status === 'completato' && d.is_foglio_firma;
  const ff = all.filter(isFF);
  check(ff.length >= 6, `fogli firma sintetici letti (${ff.length})`);
  const ctx = await browser.newContext({ viewport: { width: 1480, height: 940 } });

  await section('5. Pagina scartata o in errore', async () => {
  for (const status of ['scartato', 'errore']) {
    const d = all.find((x) => x.status === status);
    if (!d) { console.log(`  – nessun documento «${status}»: verifica saltata`); continue; }
    const page = await newPage(ctx);
    const puts = [];
    page.on('request', (r) => { if (r.method() === 'PUT' && PUT_DOC.test(r.url())) puts.push(r.url()); });
    await openReview(page, d.id);
    await page.waitForSelector('#rv-overlay .review-overlay');
    const before = await getDoc(d.id);
    await page.locator('#rv-grid').focus();
    await page.keyboard.press('9');
    await page.keyboard.press('Enter');
    await page.keyboard.press('Delete');
    await page.waitForTimeout(900);
    const after = await getDoc(d.id);
    check(puts.length === 0 && JSON.stringify(after.rows) === JSON.stringify(before.rows),
      `documento «${status}»: la griglia dietro il riquadro non accetta modifiche`);
    await page.close();
  }
  });

  await section('6-7. Tab fuori dalla griglia, fuoco sulla scansione', async () => {
    const page = await newPage(ctx);
    await openReview(page, ff[0].id);
    // nella revisione la griglia è l'ultimo elemento attivabile: il fuoco esce dalla pagina
    await page.locator('#rv-grid').focus();
    await page.keyboard.press('Control+End');
    await page.keyboard.press('Tab');
    check(await page.evaluate(() => document.activeElement?.id !== 'rv-grid'), 'Tab dall\'ultima cella porta il fuoco fuori dalla griglia');
    await page.locator('#rv-grid').focus();
    await page.keyboard.press('Control+Home');
    await page.keyboard.press('Shift+Tab');
    check(await page.evaluate(() => document.activeElement && document.activeElement.id !== 'rv-grid' && document.activeElement !== document.body),
      'Maiusc+Tab dalla prima cella porta il fuoco fuori dalla griglia');
    await page.locator('#rv-grid').focus();
    await page.keyboard.press('Control+Home');
    await page.keyboard.press('Tab');
    check(await page.evaluate(() => document.activeElement?.id === 'rv-grid'), 'Tab all\'interno della griglia passa alla cella successiva');

    // ---------------------------------------------------------------- 7. fuoco visibile sulla scansione
    await page.keyboard.press('Shift');
    await page.locator('#rv-canvas').focus();
    const ring = await page.evaluate(() => {
      const el = document.getElementById('rv-canvas');
      const after = getComputedStyle(el, '::after');
      return el.matches(':focus-visible') && after.content !== 'none' && /inset/.test(after.boxShadow);
    });
    check(ring, 'la scansione mostra l\'indicatore di fuoco');

    // anteprima: dall'ultima cella Tab raggiunge le schede del foglio
    await page.goto(`${server.url}#/anteprima`);
    await page.waitForSelector('#xl-grid tbody tr[data-r="0"]');
    await page.locator('#xl-grid').focus();
    await page.keyboard.press('Control+End');
    await page.keyboard.press('Tab');
    check(await page.evaluate(() => document.activeElement?.classList.contains('xl-tab')), 'anteprima: Tab dall\'ultima cella raggiunge le schede del foglio');
    await page.close();
  });

  await section('8. Larghezza della tabella dei documenti', async () => {
  for (const [w, h] of [[1366, 768], [1280, 720]]) {
    const c2 = await browser.newContext({ viewport: { width: w, height: h } });
    const page = await newPage(c2);
    await page.goto(`${server.url}#/documenti`);
    await page.waitForSelector('.table tbody tr[data-id]');
    const m = await page.evaluate(() => {
      const wrap = document.querySelector('.table-wrap');
      const menu = document.querySelector('.table tbody tr [data-act="menu"]').getBoundingClientRect();
      const name = document.querySelector('.table tbody tr .doc-name');
      return { sw: wrap.scrollWidth, cw: wrap.clientWidth, right: menu.right, vw: window.innerWidth, nameW: name.clientWidth };
    });
    check(m.sw <= m.cw + 1 && m.right <= m.vw, `${w}×${h}: tabella dei documenti senza scorrimento orizzontale (${m.sw}/${m.cw}), menu visibile`);
    check(m.nameW >= 130, `${w}×${h}: spazio sufficiente per il nome dell'operatore (${m.nameW} px)`);
    await c2.close();
  }
  });

  await section('1. Cambio documento con modifiche aperte', async () => {
    const i = all.findIndex((d, k) => isFF(d) && all[k + 1] && isFF(all[k + 1]));
    const A = all[i]; const B = all[i + 1];
    const a0 = await getDoc(A.id); const b0 = await getDoc(B.id);
    const page = await newPage(ctx);
    await openReview(page, A.id);

    // cella in modifica (F2) e Alt+→
    await typeNote(page, 5, 'NOTA PER A', { enter: false });
    await page.keyboard.press('Alt+ArrowRight');
    await page.waitForFunction((id) => location.hash.includes(id), B.id);
    await page.waitForFunction((x) => document.getElementById('rv-title')?.textContent === x, b0.header.operatore);
    await until(async () => (await getDoc(A.id)).rows[4].note === 'NOTA PER A', 'Alt+→ con la cella in modifica: nota salvata nel documento di partenza');
    await cell(page, 6, 'prog_entrata').click();
    await page.waitForTimeout(900);
    let b1 = await getDoc(B.id);
    check(b1.rows[4].note === b0.rows[4].note && !b1.user_edited.includes('rows.5.note'), 'il documento successivo non riceve la nota');
    check(((await cell(page, 5, 'note').textContent()) || '').trim() !== 'NOTA PER A', 'la griglia mostra i dati del documento successivo');

    // campo dell'intestazione in modifica e Alt+←
    await page.locator('#hf-istituto').click();
    await page.keyboard.press('Control+a');
    await page.keyboard.type('ISTITUTO DI B CORRETTO');
    await page.keyboard.press('Alt+ArrowLeft');
    await page.waitForFunction((id) => location.hash.includes(id), A.id);
    await page.waitForFunction((x) => document.getElementById('rv-title')?.textContent === x, a0.header.operatore);
    await until(async () => (await getDoc(B.id)).header.istituto === 'ISTITUTO DI B CORRETTO', 'Alt+← con l\'intestazione in modifica: istituto salvato nel documento di partenza');
    check(await page.locator('#hf-istituto').inputValue() === a0.header.istituto, 'il campo mostra l\'istituto del documento aperto');
    await cell(page, 6, 'prog_entrata').click();
    await page.waitForTimeout(900);
    const a1 = await getDoc(A.id);
    check(a1.header.istituto === a0.header.istituto, 'l\'istituto digitato non finisce nel documento aperto dopo');

    // pulsante «indietro» (cambio di indirizzo senza clic) con la cella in modifica
    await typeNote(page, 7, 'NOTA INDIETRO', { enter: false });
    await page.goBack();
    await page.waitForFunction((id) => location.hash.includes(id), B.id);
    await until(async () => (await getDoc(A.id)).rows[6].note === 'NOTA INDIETRO', '«indietro» con la cella in modifica: nota salvata nel documento giusto');
    await page.waitForTimeout(700);
    b1 = await getDoc(B.id);
    check(b1.rows[6].note === b0.rows[6].note, 'il documento raggiunto con «indietro» non riceve la nota');
    await page.close();
  });

  await section('2. Salvataggio non riuscito', async () => {
    const C = ff[2];
    const page = await newPage(ctx);
    await openReview(page, C.id);

    // errore di rete → «Riprova» reinvia la modifica
    await page.route(PUT_DOC, (route) => (route.request().method() === 'PUT' ? route.abort('failed') : route.continue()));
    await typeNote(page, 2, 'PROVA123');
    await until(async () => (await saveText(page)).includes('Non salvato'), 'rete assente: indicatore «Non salvato»');
    check(((await cell(page, 2, 'note').textContent()) || '').trim() === 'PROVA123', 'la correzione resta visibile');
    await page.unroute(PUT_DOC);
    await page.click('#rv-save [data-retry]');
    await until(async () => (await getDoc(C.id)).rows[1].note === 'PROVA123', '«Riprova» salva la correzione');
    await until(async () => (await saveText(page)).includes('Salvato'), 'indicatore «Salvato» dopo «Riprova»');

    // errore del server → la modifica resta in sospeso e parte con il salvataggio successivo
    await page.route(PUT_DOC, (route) => (route.request().method() === 'PUT'
      ? route.fulfill({ status: 500, contentType: 'application/json', body: JSON.stringify({ detail: 'Errore interno simulato.' }) })
      : route.continue()));
    await typeNote(page, 3, 'PROVA500');
    await until(async () => (await saveText(page)).includes('Non salvato'), 'errore del server: indicatore «Non salvato»');
    await page.unroute(PUT_DOC);
    await typeNote(page, 4, 'DOPO');
    await until(async () => {
      const d = await getDoc(C.id);
      return d.rows[2].note === 'PROVA500' && d.rows[3].note === 'DOPO';
    }, 'il salvataggio successivo invia anche la correzione non riuscita');

    // modifica rifiutata (409) → annullata, nessun «Riprova» inutile
    await page.route(PUT_DOC, (route) => (route.request().method() === 'PUT'
      ? route.fulfill({ status: 409, contentType: 'application/json', body: JSON.stringify({ detail: 'Il documento è in elaborazione: attendere il termine della lettura prima di modificarlo.' }) })
      : route.continue()));
    const reload = page.waitForResponse((r) => PUT_DOC.test(r.url()) && r.request().method() === 'GET');
    await typeNote(page, 8, 'RIFIUTATA');
    await reload;
    await page.waitForTimeout(300);
    check(!(await saveText(page)).includes('Non salvato'), 'modifica rifiutata: nessun «Non salvato · Riprova»');
    check(((await cell(page, 8, 'note').textContent()) || '').trim() !== 'RIFIUTATA', 'modifica rifiutata: la cella torna al valore salvato');
    await page.unroute(PUT_DOC);

    // correzione non salvata: non si cambia documento senza conferma
    await page.route(PUT_DOC, (route) => (route.request().method() === 'PUT' ? route.abort('failed') : route.continue()));
    await typeNote(page, 9, 'IN SOSPESO');
    await until(async () => (await saveText(page)).includes('Non salvato'), 'rete assente: correzione in sospeso');
    await page.locator('#rv-grid').focus();
    await page.keyboard.press('Alt+ArrowRight');
    await page.waitForSelector('dialog[open]');
    check((await page.locator('dialog[open] .dialog-title').textContent()).includes('non salvate'), 'uscita con correzioni non salvate: richiesta di conferma');
    await page.locator('dialog[open] .dialog-foot button[data-i="0"]').click();
    await page.waitForTimeout(400);
    check(await page.evaluate((id) => location.hash.includes(id), C.id), '«Resta sul documento»: il documento resta aperto');
    check(((await cell(page, 9, 'note').textContent()) || '').trim() === 'IN SOSPESO', 'la correzione non salvata è ancora presente');
    await page.unroute(PUT_DOC);
    await page.click('#rv-save [data-retry]');
    await until(async () => (await getDoc(C.id)).rows[8].note === 'IN SOSPESO', 'la correzione viene poi salvata');
    await page.close();

    // anteprima: correzione non salvata → l'Excel non viene generato senza di essa
    const p2 = await newPage(ctx);
    let exports = 0;
    p2.on('request', (r) => { if (r.url().endsWith('/api/export') && r.method() === 'POST') exports++; });
    await p2.goto(`${server.url}#/anteprima`);
    await p2.waitForSelector('#xl-grid tbody tr');
    await p2.click('[data-tab="dettaglio"]');
    await p2.waitForSelector('#xl-grid tbody tr[data-r="0"]');
    const noteCol = await p2.$$eval('#xl-grid thead tr.h-labels th', (ths) => ths.findIndex((th) => th.textContent.trim() === 'Note'));
    const docCol = await p2.$$eval('#xl-grid thead tr.h-labels th', (ths) => ths.findIndex((th) => th.textContent.trim() === 'Operatore'));
    const opName = (await p2.locator(`#xl-grid tbody tr[data-r="0"] td[data-c="${docCol}"]`).textContent()).trim();
    const firstDoc = all.find((d) => d.header?.operatore === opName);
    await p2.route(PUT_DOC, (route) => (route.request().method() === 'PUT' ? route.abort('failed') : route.continue()));
    await p2.locator(`#xl-grid tbody tr[data-r="0"] td[data-c="${noteCol}"]`).click();
    await p2.keyboard.type('XLS IN SOSPESO');
    await p2.keyboard.press('Enter');
    await p2.waitForTimeout(1200);
    await p2.click('[data-act="generate"]');
    await p2.waitForSelector('.toast.is-error:has-text("Excel non generato")');
    check(exports === 0, 'anteprima: con una correzione non salvata l\'Excel non viene generato');
    await p2.unroute(PUT_DOC);
    const exp = p2.waitForResponse((r) => r.url().endsWith('/api/export') && r.request().method() === 'POST', { timeout: 30000 });
    await p2.click('[data-act="generate"]');
    check((await exp).ok(), 'anteprima: al nuovo tentativo la correzione viene salvata e l\'Excel generato');
    const fd = await getDoc(firstDoc.id);
    check(fd.rows.some((r) => r.note === 'XLS IN SOSPESO'), 'anteprima: correzione presente nel documento');
    await p2.close();
  });

  await section('3. Chiusura della finestra', async () => {
    const closeAndCheck = async (page, fn, message) => {
      let prompted = false;
      page.on('dialog', (d) => { if (d.type() === 'beforeunload') prompted = true; d.accept().catch(() => {}); });
      await page.close({ runBeforeUnload: true });
      await until(fn, message);
      return prompted;
    };
    let page = await newPage(ctx);
    await openReview(page, ff[3].id);
    await typeNote(page, 3, 'CHIUSURA1', { enter: false });
    const prompted = await closeAndCheck(page, async () => (await getDoc(ff[3].id)).rows[2].note === 'CHIUSURA1', 'chiusura con la cella in modifica: nota salvata');
    check(prompted, 'chiusura con modifiche in sospeso: il browser chiede conferma');

    page = await newPage(ctx);
    await openReview(page, ff[4].id);
    await typeNote(page, 3, 'CHIUSURA2');
    await page.waitForTimeout(80);
    await closeAndCheck(page, async () => (await getDoc(ff[4].id)).rows[2].note === 'CHIUSURA2', 'chiusura subito dopo Invio: nota salvata');

    page = await newPage(ctx);
    await openReview(page, ff[5].id);
    await page.fill('#hf-istituto', 'ISTITUTO ALLA CHIUSURA');
    await closeAndCheck(page, async () => (await getDoc(ff[5].id)).header.istituto === 'ISTITUTO ALLA CHIUSURA', 'chiusura con l\'intestazione in modifica: istituto salvato');

    page = await newPage(ctx);
    await page.goto(`${server.url}#/anteprima`);
    await page.waitForSelector('#xl-grid tbody tr');
    await page.click('[data-tab="dettaglio"]');
    await page.waitForSelector('#xl-grid tbody tr[data-r="1"]');
    const noteCol = await page.$$eval('#xl-grid thead tr.h-labels th', (ths) => ths.findIndex((th) => th.textContent.trim() === 'Note'));
    await page.locator(`#xl-grid tbody tr[data-r="1"] td[data-c="${noteCol}"]`).click();
    await page.keyboard.type('CHIUSURA XLS');
    await page.keyboard.press('Enter');
    await closeAndCheck(page, async () => {
      const st = await api(server, '/api/state');
      for (const d of st.documents.filter(isFF)) if ((await getDoc(d.id)).rows.some((r) => r.note === 'CHIUSURA XLS')) return true;
      return false;
    }, 'anteprima: chiusura subito dopo la modifica, nota salvata');
  });

  await section('4. Fuoco durante gli aggiornamenti', async () => {
    const page = await newPage(ctx);
    const reprocess = async (ids) => { for (const id of ids) await api(server, `/api/documents/${id}/reprocess`, { method: 'POST' }); };
    const stable = ff[0].id;
    const busyIds = ff.slice(1, 6).map((d) => d.id);
    await page.goto(`${server.url}#/documenti`);
    await page.waitForSelector(`tr[data-id="${stable}"]`);
    await reprocess(busyIds);
    await page.locator(`tr[data-id="${stable}"]`).focus();
    await page.waitForTimeout(2600);
    check(await page.evaluate((id) => document.activeElement?.tagName === 'TR' && document.activeElement.dataset.id === id, stable),
      'Documenti: la riga con il fuoco lo mantiene durante la lettura di altri documenti');
    const busyRow = page.locator(`tr[data-id="${busyIds[busyIds.length - 1]}"]`);
    await busyRow.locator('input[type="checkbox"]').focus();
    await page.waitForTimeout(2600);
    check(await page.evaluate((id) => document.activeElement?.dataset?.check === id, busyIds[busyIds.length - 1]),
      'Documenti: la casella di una riga in lettura mantiene il fuoco');

    await waitIdle(server, { min: 1, timeout: 60000 });
    await page.goto(`${server.url}#/dashboard`);
    await page.waitForSelector(`.queue-item[data-open="${stable}"]`);
    await reprocess(busyIds);
    await page.locator(`.queue-item[data-open="${stable}"]`).focus();
    await page.waitForTimeout(2600);
    check(await page.evaluate((id) => document.activeElement?.dataset?.open === id, stable),
      'Dashboard: la voce della coda con il fuoco lo mantiene durante la lettura');
    await page.locator('.kpi[data-nav="#/documenti"]').focus();
    await page.waitForTimeout(2600);
    check(await page.evaluate(() => document.activeElement?.dataset?.nav === '#/documenti'), 'Dashboard: l\'indicatore con il fuoco lo mantiene');
    await page.close();
    await waitIdle(server, { min: 1, timeout: 60000 });
  });

  await section('9. Claude Vision solo dopo conferma; costi in Dashboard', async () => {
    const setEngine = (engine) => api(server, '/api/settings', { method: 'PUT', body: JSON.stringify({ engine }) });
    const engineNow = async () => (await api(server, '/api/settings')).settings.engine;
    await setEngine('locale');
    const page = await newPage(ctx);
    await page.goto(`${server.url}#/impostazioni`);
    await page.waitForSelector('[data-engine="claude"]');
    await page.click('[data-engine="claude"]');
    await page.waitForSelector('dialog[open]');
    const text = await page.locator('dialog[open]').textContent();
    check(/Anthropic/.test(text) && /minori/.test(text), 'scelta di Claude Vision: avviso su invio dei dati e costi');
    await page.locator('dialog[open] .dialog-foot button[data-i="0"]').click();
    await page.waitForTimeout(600);
    check(await engineNow() === 'locale', 'annullando, il motore resta quello locale');
    await page.click('[data-engine="claude"]');
    await page.waitForSelector('dialog[open]');
    await page.locator('dialog[open] .dialog-foot button[data-i="1"]').click();
    await until(async () => (await engineNow()) === 'claude', 'confermando, Claude Vision viene attivato');

    await page.goto(`${server.url}#/dashboard`);
    await page.waitForSelector('.kpi');
    check((await page.locator('.kpi').allTextContents()).some((t) => /Costo stimato/.test(t)), 'Claude Vision: la Dashboard mostra il costo stimato');
    // tornando al motore locale dalle impostazioni, senza letture a pagamento: nessun costo
    await api(server, '/api/documents', { method: 'DELETE' });
    await page.goto(`${server.url}#/impostazioni`);
    await page.click('[data-engine="locale"]');
    await until(async () => (await engineNow()) === 'locale', 'motore locale selezionato senza richiesta di conferma');
    await page.goto(`${server.url}#/dashboard`);
    await until(async () => {
      const kpis = await page.locator('.kpi').allTextContents();
      return kpis.length === 6 && !kpis.some((t) => /Costo|Claude|\$/.test(t));
    }, 'motore locale: la Dashboard non mostra costi né Claude');
    await page.close();
  });

  await ctx.close();
  check(!pageErrors.length, 'nessun errore di script nel browser');
  for (const e of pageErrors) console.error(`    ${e}`);
  if (failures.length) {
    console.error(`\nVerifiche non superate (${failures.length}):`);
    for (const f of failures) console.error(`  - ${f}`);
    process.exitCode = 1;
  } else {
    console.log('\nProve di regressione superate.');
  }
} catch (err) {
  console.error(err);
  console.error(server.log?.() || '');
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.stop();
}

// Schermate di tutte le viste, tema chiaro e scuro, a 1480×940 (più alcune a 1100 px).
//
//   NODE_PATH=/opt/node22/lib/node_modules node tests/ui/screenshots.mjs
//
// Le immagini vanno in SIRIO_UI_SHOTS (vedi _helpers.mjs). Esce con codice 1 se la console
// del browser registra errori.

import path from 'node:path';
import { api, check, launch, shotsDir, sleep, startServer, waitIdle, watchConsole } from './_helpers.mjs';

const server = await startServer({ documents: 6, delay: 0.8 });
const out = shotsDir();
const browser = await launch();
const errors = [];
const shots = [];

async function ready(page) {
  await page.waitForSelector('#splash', { state: 'detached', timeout: 15000 });
  await page.waitForTimeout(350);
}

async function shot(page, name) {
  const file = path.join(out, `${name}.png`);
  await page.screenshot({ path: file, fullPage: false });
  shots.push(file);
}

try {
  const state = await waitIdle(server, { min: 6 });
  const docs = state.documents.filter((d) => d.status === 'completato' && d.is_foglio_firma);
  check(docs.length >= 4, 'documenti sintetici letti');
  const target = docs.find((d) => d.totals.campi_illeggibili > 0) || docs[0];

  for (const scheme of ['light', 'dark']) {
    const ctx = await browser.newContext({ viewport: { width: 1480, height: 940 }, colorScheme: scheme, deviceScaleFactor: 1 });
    const page = await ctx.newPage();
    errors.push(...watchConsole(page, `[${scheme}] `));
    const tag = scheme === 'light' ? 'chiaro' : 'scuro';

    await page.goto(server.url);
    await ready(page);
    await page.waitForSelector('.kpi');
    await shot(page, `01-dashboard-${tag}`);

    await page.click('a[data-route="documenti"]');
    await page.waitForSelector('.table tbody tr');
    await page.waitForTimeout(300);
    await shot(page, `02-documenti-${tag}`);
    if (scheme === 'light') {
      await page.click('[data-view="cards"]');
      await page.waitForSelector('.doc-card');
      await page.waitForTimeout(400);
      await shot(page, `02b-documenti-schede-${tag}`);
      await page.click('[data-view="table"]');
    }

    await page.goto(`${server.url}#/revisione/${target.id}`);
    await page.waitForSelector('.sheet td.is-active');
    await page.waitForTimeout(900);
    await shot(page, `03-revisione-${tag}`);

    // tooltip di una cella illeggibile
    const illeg = page.locator('.sheet td.st-illeggibile').first();
    if (await illeg.count()) {
      await illeg.hover();
      await page.waitForTimeout(700);
      await shot(page, `03b-revisione-illeggibile-${tag}`);
    }

    await page.click('a[data-route="anteprima"]');
    await page.waitForSelector('#xl-grid .sheet tbody tr');
    for (const tab of ['riepilogo', 'dettaglio', 'anomalie']) {
      await page.click(`[data-tab="${tab}"]`);
      await page.waitForTimeout(350);
      await shot(page, `04-anteprima-${tab}-${tag}`);
    }

    await page.click('a[data-route="impostazioni"]');
    await page.waitForSelector('.engine-card');
    await page.waitForTimeout(300);
    await shot(page, `05-impostazioni-${tag}`);
    await page.evaluate(() => document.querySelector('.view').scrollTo(0, 99999));
    await page.waitForTimeout(250);
    await shot(page, `05b-impostazioni-fondo-${tag}`);
    await ctx.close();
  }

  // finestra stretta (1100 px)
  const ctx = await browser.newContext({ viewport: { width: 1100, height: 820 }, colorScheme: 'light' });
  const page = await ctx.newPage();
  errors.push(...watchConsole(page, '[1100] '));
  await page.goto(server.url);
  await ready(page);
  await shot(page, '06-dashboard-1100');
  await page.goto(`${server.url}#/revisione/${target.id}`);
  await page.waitForSelector('.sheet td.is-active');
  await page.waitForTimeout(800);
  await shot(page, '06-revisione-1100');
  await page.goto(`${server.url}#/anteprima`);
  await page.waitForSelector('#xl-grid .sheet tbody tr');
  await page.waitForTimeout(300);
  await shot(page, '06-anteprima-1100');
  await ctx.close();

  // banner della chiave API: impostazioni con motore Claude e nessuna chiave sono quelle del server di sviluppo
  await api(server, '/api/settings', { method: 'PUT', body: JSON.stringify({ tema: 'auto' }) });

  console.log(`\nSchermate (${shots.length}):`);
  for (const f of shots) console.log(`  ${f}`);
  if (errors.length) {
    console.error(`\nErrori nella console del browser (${errors.length}):`);
    for (const e of errors) console.error(`  ${e}`);
    process.exitCode = 1;
  } else {
    console.log('\n  ✓ nessun errore nella console del browser');
  }
} catch (err) {
  console.error(err);
  console.error(server.log?.() || '');
  process.exitCode = 1;
} finally {
  await browser.close();
  await server.stop();
  await sleep(100);
}

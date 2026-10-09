"""Server di sviluppo per l'interfaccia: applicazione completa con motore OCR finto.

Avvia il server reale (archivio, coda, API, file statici di ``sirio/web``) con un
motore di prova al posto di Claude/TrOCR e precarica alcuni fogli firma
sintetici (nomi fittizi), cosi' l'interfaccia si puo' sviluppare e provare senza
chiave API e senza scaricare modelli::

    python tests/dev_server.py --porta 8765 --documenti 6

Stampa l'indirizzo e il token. Non chiude il server automaticamente (come
``--senza-finestra``): Ctrl+C per terminare. I dati vanno in una cartella
temporanea, eliminata all'uscita (salvo ``--cartella-dati`` o ``--mantieni``).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import secrets
import shutil
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("sirio.dev")

# Persone, scuole e mesi di fantasia.
PERSONE = (
    ("ROSSI MARIO", "BIANCHI LUCA", "IC 1 VERDI"),
    ("VERDI ANNA", "ESPOSITO GIULIA", "IC 2 MAZZINI"),
    ("RUSSO PAOLO", "ROMANO SARA", "IC 3 GARIBALDI"),
    ("COLOMBO LAURA", "RICCI MATTEO", "IC 4 CAVOUR"),
    ("MARINO ELENA", "GRECO DAVIDE", "IC 5 MANZONI"),
    ("BRUNO CHIARA", "GALLO MARCO", "IC 6 LEOPARDI"),
    ("CONTI FRANCESCO", "DE LUCA SOFIA", "IC 7 CARDUCCI"),
    ("COSTA GIORGIA", "MANCINI ANDREA", "IC 8 PASCOLI"),
)
PERIODI = ((2, 2026), (3, 2026), (1, 2026), (11, 2025), (12, 2025), (4, 2026))


def demo_data(i: int) -> dict:
    """Dati di un foglio firma d'esempio (mese realistico, giorni feriali compilati)."""
    from sirio import calendario  # noqa: PLC0415

    operatore, alunno, istituto = PERSONE[i % len(PERSONE)]
    mese, anno = PERIODI[i % len(PERIODI)]
    rng = random.Random(1000 + i)
    pomeriggio = rng.randrange(0, 5)          # giorno della settimana con orario pomeridiano
    errore_ore = (i % 3 == 1)                 # un'incoerenza tra ore e orario
    rows = []
    for g in range(1, calendario.giorni_nel_mese(anno, mese) + 1):
        row: dict = {"giorno": g}
        if calendario.is_lavorativo(anno, mese, g):
            from datetime import date  # noqa: PLC0415

            a, b = ("11:00", "14:00") if date(anno, mese, g).weekday() == pomeriggio else ("08:00", "11:00")
            row.update(prog_entrata=a, prog_uscita=b)
            caso = rng.random()
            if caso < 0.06:
                row.update(assenza_alunno=True, ore_dichiarate=1.5, firma=True, trattino_effettivo=True)
            elif caso < 0.10:
                row.update(assenza_operatore=True, note="104", firma=True, trattino_effettivo=True)
            elif caso < 0.13:
                row.update(note="USCITA DIDATTICA", trattino_effettivo=True)
            else:
                row.update(eff_entrata=a, eff_uscita=b, ore_dichiarate=3.0, firma=True)
                if errore_ore:
                    row["ore_dichiarate"] = 2.5
                    errore_ore = False
        rows.append(row)
    totale = sum(r.get("ore_dichiarate") or 0.0 for r in rows)
    return {
        "header": {
            "anno_scolastico": "2025/2026" if (anno, mese) >= (2025, 9) else "2024/2025",
            "lotto": "1",
            "municipalita": str(1 + i % 10),
            "ente": "COOPERATIVA SOCIALE ESEMPIO",
            "istituto": istituto,
            "operatore": operatore,
            "alunno": alunno,
            "mese": mese,
            "anno": anno,
            "ore_pei": 15.0,
            "sostituzione": "NO",
            "firma_coordinatore": True,
            "timbro_referente": i % 4 != 3,
            "totale_mensile_dichiarato": totale,
        },
        "rows": rows,
    }


def build_samples(n: int, special: bool) -> tuple[list[tuple[str, bytes]], dict[str, dict], dict[str, Exception]]:
    """File d'esempio (nome, contenuto), verita' per il motore finto e file che falliscono."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    from sirio.engines.base import EngineError  # noqa: PLC0415
    from tests.synthetic import make_synthetic_sheet, sheet_to_pdf  # noqa: PLC0415

    files: list[tuple[str, bytes]] = []
    truths: dict[str, dict] = {}
    failures: dict[str, Exception] = {}
    sheets = []
    for i in range(n):
        img, truth = make_synthetic_sheet(demo_data(i), seed=i, skew_deg=(i % 5 - 2) * 0.4)
        sheets.append((img, truth))
    multipage = sheets[-2:] if n >= 4 else []
    singles = sheets[: n - len(multipage)]
    for i, (img, truth) in enumerate(singles):
        surname = truth["header"]["operatore"].split()[0].lower()
        kind = i % 3
        if kind == 0:
            name = f"foglio_firma_{i + 1:02d}_{surname}.pdf"
            data = sheet_to_pdf([img])
        elif kind == 1:
            name = f"scansione_{i + 1:02d}_{surname}.jpg"
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
            data = buf.tobytes()
        else:
            name = f"foto_{i + 1:02d}_{surname}.png"
            ok, buf = cv2.imencode(".png", img)
            data = buf.tobytes()
        files.append((name, data))
        truths[f"{name}#1"] = truth
    if multipage:
        name = "fogli_firma_multipagina.pdf"
        files.append((name, sheet_to_pdf([img for img, _ in multipage])))
        for page, (_, truth) in enumerate(multipage, start=1):
            truths[f"{name}#{page}"] = truth
    if special:
        blank = np.full((2338, 1654, 3), 246, np.uint8)
        cv2.putText(blank, "Verbale di consegna", (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (40, 40, 40), 4)
        ok, buf = cv2.imencode(".png", blank)
        files.append(("pagina_non_pertinente.png", buf.tobytes()))
        img, _ = make_synthetic_sheet(demo_data(n), seed=99)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        files.append(("errore_simulato.jpg", buf.tobytes()))
        failures["errore_simulato.jpg"] = EngineError(
            "Lettura non riuscita (errore simulato dal server di sviluppo): connessione al servizio interrotta. "
            "Riprovare con «Rielabora»."
        )
    return files, truths, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Server di sviluppo di Sirio OCR (motore OCR finto).")
    parser.add_argument("--porta", type=int, default=8765)
    parser.add_argument("--documenti", type=int, default=6, help="numero di fogli firma sintetici da precaricare")
    parser.add_argument("--ritardo", type=float, default=2.5, help="durata simulata di una lettura (secondi)")
    parser.add_argument("--concorrenza", type=int, default=2, help="letture simultanee del motore finto")
    parser.add_argument("--token", default=None, help="token fisso (predefinito: casuale)")
    parser.add_argument("--cartella-dati", default=None, help="cartella dei dati (predefinita: temporanea)")
    parser.add_argument("--mantieni", action="store_true", help="non eliminare la cartella temporanea all'uscita")
    parser.add_argument("--senza-casi-speciali", action="store_true",
                        help="non aggiungere la pagina non pertinente e il documento in errore")
    parser.add_argument("--motore-non-pronto", action="store_true",
                        help="simula un motore non disponibile (documenti in attesa)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    temporary = args.cartella_dati is None
    data_dir = Path(args.cartella_dati or tempfile.mkdtemp(prefix="sirio-dev-")).resolve()
    os.environ["SIRIO_DATA_DIR"] = str(data_dir)
    os.environ.setdefault("SIRIO_EXPORT_DIR", str(data_dir / "export"))

    import uvicorn  # noqa: PLC0415

    from sirio import config  # noqa: PLC0415
    from sirio.app import setup_logging  # noqa: PLC0415
    from sirio.processing import Processor  # noqa: PLC0415
    from sirio.server import create_app  # noqa: PLC0415
    from sirio.store import DocumentStore  # noqa: PLC0415
    from tests.fake_engine import FakeEngine  # noqa: PLC0415

    setup_logging(args.log_level)
    # Impostazioni di sviluppo: motore "claude" (finto) con concorrenza configurabile.
    settings = config.load_settings().model_copy(update={"engine": "claude", "concorrenza": max(1, args.concorrenza)})
    config.save_settings(settings)

    files, truths, failures = build_samples(max(0, args.documenti), special=not args.senza_casi_speciali)
    engine = FakeEngine(
        name="claude",
        delay=max(0.0, args.ritardo),
        truths=truths,
        unknown="scartato",
        fail_for=failures,
        available=not args.motore_non_pronto,
        message="" if not args.motore_non_pronto else "Motore di prova non disponibile (--motore-non-pronto).",
    )
    store = DocumentStore(config.documents_dir())
    processor = Processor(store, config.load_settings, lambda _s: engine)
    token = args.token or secrets.token_urlsafe(24)
    app = create_app(store, processor, token, on_shutdown=lambda: setattr(server, "should_exit", True))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.porta, log_config=None,
                                           access_log=False, lifespan="off", ws="none"))

    def preload() -> None:
        for name, data in files:
            try:
                store.add_file(name, data, on_document=lambda d: processor.enqueue(d.id))
            except ValueError as exc:
                log.warning("Precaricamento di %s non riuscito: %s", name, exc)
        log.info("Precaricati %d file d'esempio", len(files))

    processor.start()
    threading.Thread(target=preload, name="sirio-dev-precarica", daemon=True).start()
    url = f"http://127.0.0.1:{args.porta}/"
    print(f"Sirio OCR (sviluppo) su {url}", flush=True)
    print(f"Token: {token}", flush=True)
    print(f"Dati: {data_dir}  (export: {config.export_dir()})", flush=True)
    print(f"Esempio: curl -H 'X-Sirio-Token: {token}' {url}api/state", flush=True)
    try:
        server.run()
    finally:
        processor.stop(timeout=5)
        if temporary and not args.mantieni:
            shutil.rmtree(data_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

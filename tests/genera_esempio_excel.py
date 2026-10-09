"""Genera docs/esempio/Rendicontazione_esempio.xlsx da fogli firma di FANTASIA.

Uso (dalla cartella del progetto)::

    python -m tests.genera_esempio_excel [percorso.xlsx]

I nomi di operatori, alunni e istituti sono inventati. I documenti coprono i
casi che l'Excel deve mostrare: campo illeggibile, letture incerte, correzione
manuale, assenza dell'alunno con ore parziali, assenza dell'operatore con nota
"104", ponte non svolto, ore non coerenti, totale mensile diverso, sabato
lavorato oltre il PEI, documenti non inclusi.

``documenti_esempio()`` è usata anche dai test di ``tests/test_excel.py``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from sirio.excel_export import ExportOptions, export_workbook
from sirio.models import DayRow, Document, Header, Usage, empty_rows

ENTE = "COOPERATIVA SOCIALE SIRIO"
DESTINAZIONE = Path(__file__).resolve().parents[1] / "docs" / "esempio" / "Rendicontazione_esempio.xlsx"


def _giorni(anno: int, mese: int, settimana: set[int]) -> list[int]:
    """Giorni del mese il cui giorno della settimana (lun=0) è in ``settimana``."""
    out = []
    for g in range(1, 32):
        try:
            if date(anno, mese, g).weekday() in settimana:
                out.append(g)
        except ValueError:
            break
    return out


def _lavorato(g: int, entrata: str, uscita: str, ore: float, **kw) -> DayRow:
    return DayRow(
        giorno=g,
        prog_entrata=entrata,
        prog_uscita=uscita,
        eff_entrata=entrata,
        eff_uscita=uscita,
        ore_dichiarate=ore,
        firma=True,
        **kw,
    )


def _documento(doc_id: str, file: str, header: Header, rows: list[DayRow], **kw) -> Document:
    base = dict(
        id=doc_id,
        source_file=file,
        source_page=1,
        page_count=1,
        status="completato",
        is_foglio_firma=True,
        engine="claude",
        model="claude-opus-5-5",
        confidence=0.95,
        created_at="2026-04-02T09:15:00+02:00",
        updated_at="2026-04-02T09:42:00+02:00",
        usage=Usage(input_tokens=9800, output_tokens=2100, cost_usd=0.08, seconds=41.0),
        header=header,
        rows=rows,
    )
    base.update(kw)
    return Document(**base)


def _header(**kw) -> Header:
    base = dict(
        anno_scolastico="2025/2026",
        lotto="1",
        municipalita="2",
        ente=ENTE,
        firma_coordinatore=True,
        timbro_referente=True,
    )
    base.update(kw)
    return Header(**base)


def documenti_esempio() -> list[Document]:
    docs: list[Document] = []

    # 1) ROSSI MARIO - febbraio 2026: campo illeggibile, letture incerte, assenza alunno, ponte non svolto.
    rows = empty_rows()
    for g in _giorni(2026, 2, {0, 1, 2, 3, 4}):
        rows[g - 1] = _lavorato(g, "08:00", "11:00", 3)
    rows[3] = DayRow(giorno=4, prog_entrata="08:00", prog_uscita="11:00", ore_dichiarate=1.5,
                     assenza_alunno=True, firma=True, trattino_effettivo=True)
    for g in (16, 17):
        rows[g - 1] = DayRow(giorno=g, prog_entrata="08:00", prog_uscita="11:00", trattino_effettivo=True,
                             note="PONTE DI CARNEVALE")
    rows[11] = rows[11].model_copy(update={"eff_uscita": None, "illeggibili": ["eff_uscita"]})
    rows[17] = rows[17].model_copy(update={"incerti": ["ore_dichiarate"], "confidenza": {"ore_dichiarate": 0.55}})
    docs.append(_documento(
        "a1b2c3d4e5f6", "fogli_firma_febbraio.pdf",
        _header(istituto="IC 12 VITTORIO EMANUELE", operatore="ROSSI MARIO", alunno="BIANCHI LUCA",
                mese=2, anno=2026, ore_pei=15, sostituzione="NO", data_compilazione="27/02/2026",
                totale_mensile_dichiarato=52.5, incerti=["alunno"]),
        rows, source_page=1, page_count=2, confidence=0.91,
    ))

    # 2) ESPOSITO ANNA - febbraio 2026: assenza operatore "104", correzione manuale, documento confermato.
    rows = empty_rows()
    for g in _giorni(2026, 2, {0, 1, 3, 4}):
        rows[g - 1] = _lavorato(g, "11:00", "14:00", 3)
    rows[9] = DayRow(giorno=10, prog_entrata="11:00", prog_uscita="14:00", assenza_operatore=True, note="104")
    docs.append(_documento(
        "b2c3d4e5f6a1", "fogli_firma_febbraio.pdf",
        _header(istituto="IC 47 SARRIA-MONTI", operatore="ESPOSITO ANNA", alunno="VERDI SOFIA",
                mese=2, anno=2026, ore_pei=12, sostituzione="NO", data_compilazione="27/02/2026",
                totale_mensile_dichiarato=45),
        rows, source_page=2, page_count=2, confidence=0.97, user_verified=True,
        user_edited=["rows.5.eff_entrata"], ocr_originali={"rows.5.eff_entrata": "11:30"},
    ))

    # 3) ROSSI MARIO - marzo 2026: ore non coerenti, firma mancante, totale mensile diverso dalla somma.
    rows = empty_rows()
    for g in _giorni(2026, 3, {0, 1, 2, 3, 4}):
        rows[g - 1] = _lavorato(g, "08:00", "11:00", 3)
    rows[10] = rows[10].model_copy(update={"eff_uscita": "10:30"})
    rows[18] = DayRow(giorno=19, prog_entrata="08:00", prog_uscita="11:00", ore_dichiarate=1.5,
                      assenza_alunno=True, firma=True, trattino_effettivo=True)
    rows[24] = rows[24].model_copy(update={"firma": False})
    rows[22] = rows[22].model_copy(update={"incerti": ["prog_uscita"], "confidenza": {"prog_uscita": 0.6}})
    docs.append(_documento(
        "c3d4e5f6a1b2", "rossi_marzo.pdf",
        _header(istituto="IC 12 VITTORIO EMANUELE", operatore="ROSSI MARIO", alunno="BIANCHI LUCA",
                mese=3, anno=2026, ore_pei=15, sostituzione="NO", data_compilazione="31/03/2026",
                totale_mensile_dichiarato=64),
        rows, confidence=0.93,
    ))

    # 4) DE LUCA GIOVANNA - marzo 2026: sabato di recupero oltre il PEI, orario diverso, timbro mancante.
    rows = empty_rows()
    for g in _giorni(2026, 3, {0, 1, 2, 3, 4}):
        rows[g - 1] = _lavorato(g, "09:00", "11:00", 2)
    rows[13] = DayRow(giorno=14, eff_entrata="09:00", eff_uscita="11:00", ore_dichiarate=2, firma=True,
                      note="RECUPERO")
    rows[19] = rows[19].model_copy(update={"eff_uscita": "11:15", "ore_dichiarate": 2.25})
    docs.append(_documento(
        "d4e5f6a1b2c3", "de_luca_marzo.pdf",
        _header(istituto="IC 12 VITTORIO EMANUELE", operatore="DE LUCA GIOVANNA", alunno="NERI MATTEO",
                mese=3, anno=2026, ore_pei=10, sostituzione="SI", data_compilazione="31/03/2026",
                totale_mensile_dichiarato=46.25, timbro_referente=False, incerti=["ore_pei"]),
        rows, confidence=0.89, engine="claude", model="claude-sonnet-5-5",
    ))

    # Documenti che non entrano nella rendicontazione
    docs.append(Document(id="e5f6a1b2c3d4", source_file="lettera_trasmissione.pdf", status="scartato",
                         is_foglio_firma=False, engine="claude", model="claude-opus-5-5"))
    docs.append(Document(id="f6a1b2c3d4e5", source_file="scansione_aprile.pdf", status="errore",
                         error="Connessione al servizio OCR non riuscita."))
    return docs


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    destinazione = Path(args[0]) if args else DESTINAZIONE
    percorso = export_workbook(documenti_esempio(), destinazione, ExportOptions())
    print(f"Excel di esempio generato: {percorso}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

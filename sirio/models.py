"""Modello dati condiviso da tutti i moduli di Sirio OCR.

Questo file e' il *contratto* tra motori OCR, validazione, archivio, server
web ed esportazione Excel: ogni modulo legge e scrive questi oggetti.

Convenzioni
-----------
* Gli orari sono stringhe normalizzate ``"HH:MM"`` (24h, con zero iniziale),
  oppure ``None`` se la cella e' vuota o contiene solo un trattino.
* Le ore sono numeri decimali (``1.5`` = un'ora e mezza).
* I giorni vanno da 1 a 31; un documento ha sempre esattamente 31 righe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Nomi canonici delle colonne della tabella giornaliera (usati anche come
# chiavi in ``DayRow.incerti`` e nelle anomalie ``Anomaly.campo``).
DAY_FIELDS: tuple[str, ...] = (
    "prog_entrata",
    "prog_uscita",
    "eff_entrata",
    "eff_uscita",
    "ore_dichiarate",
    "assenza_alunno",
    "assenza_operatore",
    "firma",
    "note",
)

# Nomi dei campi d'intestazione (chiavi di ``Header`` e di ``Header.incerti``).
HEADER_FIELDS: tuple[str, ...] = (
    "anno_scolastico",
    "lotto",
    "municipalita",
    "ente",
    "istituto",
    "operatore",
    "alunno",
    "mese",
    "anno",
    "ore_pei",
    "sostituzione",
    "data_compilazione",
    "firma_coordinatore",
    "timbro_referente",
    "totale_mensile_dichiarato",
)

Severity = Literal["errore", "attenzione", "info"]
DocStatus = Literal["in_coda", "in_lavorazione", "completato", "errore", "scartato"]
DocState = Literal["ok", "da_verificare", "errori"]


class Header(BaseModel):
    """Intestazione e pie' di pagina del foglio firma."""

    anno_scolastico: str | None = None          # es. "2025/2026"
    lotto: str | None = None                    # es. "1"
    municipalita: str | None = None             # es. "2"
    ente: str | None = None                     # es. "Cooperativa Sociale Sirio"
    istituto: str | None = None                 # es. "IC 12 Vittorio Emanuele"
    operatore: str | None = None                # "COGNOME NOME" come scritto
    alunno: str | None = None
    mese: int | None = None                     # 1..12 (MESE/ANNO DI RIF.)
    anno: int | None = None                     # es. 2026
    ore_pei: float | None = None                # ore settimanali da PEI
    sostituzione: Literal["SI", "NO"] | None = None
    data_compilazione: str | None = None        # "Napoli, gg/mm/aaaa" -> "gg/mm/aaaa"
    firma_coordinatore: bool = False            # firma del coordinatore dell'ente
    timbro_referente: bool = False              # timbro/firma del referente scolastico
    totale_mensile_dichiarato: float | None = None  # "Totale ore effettive mensili"
    incerti: list[str] = Field(default_factory=list)  # campi di HEADER_FIELDS letti con incertezza
    illeggibili: list[str] = Field(default_factory=list)  # campi scritti ma NON leggibili (valore = None)


class DayRow(BaseModel):
    """Una riga giornaliera del foglio firma (giorno 1..31)."""

    giorno: int
    prog_entrata: str | None = None             # orario programmato - entrata
    prog_uscita: str | None = None              # orario programmato - uscita
    eff_entrata: str | None = None              # orario effettivo - entrata
    eff_uscita: str | None = None               # orario effettivo - uscita
    ore_dichiarate: float | None = None         # "Tot. ore effettive" scritto sul foglio
    assenza_alunno: bool = False
    assenza_operatore: bool = False
    firma: bool = False                         # firma dell'operatore presente
    note: str | None = None
    trattino_effettivo: bool = False            # "-" scritto negli orari effettivi
    incerti: list[str] = Field(default_factory=list)  # campi di DAY_FIELDS letti con incertezza
    illeggibili: list[str] = Field(default_factory=list)  # campi scritti ma NON leggibili (valore = None)
    confidenza: dict[str, float] = Field(default_factory=dict)  # 0..1 per campo (facoltativo)

    def has_content(self) -> bool:
        """True se nella riga e' scritto qualcosa."""
        return any(
            [
                self.prog_entrata,
                self.prog_uscita,
                self.eff_entrata,
                self.eff_uscita,
                self.ore_dichiarate is not None,
                self.assenza_alunno,
                self.assenza_operatore,
                self.firma,
                bool(self.note and self.note.strip()),
                self.trattino_effettivo,
                bool(self.illeggibili),
            ]
        )


def empty_rows() -> list[DayRow]:
    return [DayRow(giorno=g) for g in range(1, 32)]


class Anomaly(BaseModel):
    """Esito di un controllo di validazione."""

    codice: str                                 # es. "E01_ORE_NON_COERENTI"
    gravita: Severity
    messaggio: str                              # frase leggibile in italiano
    giorno: int | None = None                   # None = riguarda l'intero documento
    campo: str | None = None                    # campo coinvolto (DAY_FIELDS / HEADER_FIELDS)
    valore_letto: str | None = None
    valore_atteso: str | None = None


class WeekTotal(BaseModel):
    """Totale di una settimana (lunedi'-domenica) all'interno del mese."""

    settimana: int                              # numero progressivo nel mese (1..6)
    dal: int                                    # primo giorno del mese incluso
    al: int                                     # ultimo giorno del mese incluso
    ore: float                                  # ore effettive (riconosciute) nella settimana
    ore_pei: float | None = None
    differenza: float | None = None             # ore - ore_pei


class Totals(BaseModel):
    """Totali e statistiche calcolati dalla validazione."""

    giorni_programmati: int = 0                 # righe con orario programmato
    giorni_lavorati: int = 0                    # righe con ore riconosciute > 0 e senza assenza operatore
    ore_programmate: float = 0.0                # somma (prog_uscita - prog_entrata)
    ore_calcolate: float = 0.0                  # somma (eff_uscita - eff_entrata)
    ore_dichiarate: float = 0.0                 # somma della colonna "Tot. ore effettive"
    ore_riconosciute: float = 0.0               # per riga: dichiarate se presenti, altrimenti calcolate
    totale_mensile_dichiarato: float | None = None
    differenza_totale: float | None = None      # ore_dichiarate - totale_mensile_dichiarato
    assenze_alunno: int = 0
    assenze_operatore: int = 0
    firme_mancanti: int = 0                     # righe con ore/orari ma senza firma operatore
    campi_incerti: int = 0
    campi_illeggibili: int = 0
    n_errori: int = 0
    n_attenzioni: int = 0
    n_info: int = 0
    stato: DocState = "ok"
    settimane: list[WeekTotal] = Field(default_factory=list)


class Usage(BaseModel):
    """Consumo e costo stimato dell'elaborazione OCR."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0


class ExtractionResult(BaseModel):
    """Quello che un motore OCR restituisce per una pagina."""

    is_foglio_firma: bool = True
    header: Header = Field(default_factory=Header)
    rows: list[DayRow] = Field(default_factory=empty_rows)
    ocr_notes: str | None = None                # osservazioni del motore (correzioni, cancellature...)
    confidence: float | None = None             # 0..1 complessiva
    engine: str = ""                            # "claude" | "locale" | "demo"
    model: str | None = None
    usage: Usage = Field(default_factory=Usage)


class Document(BaseModel):
    """Un foglio firma (= una pagina di un PDF o un'immagine) in lavorazione."""

    id: str
    source_file: str                            # nome del file caricato
    source_page: int = 1                        # pagina (1-based) nel file d'origine
    page_count: int = 1                         # pagine totali del file d'origine
    source_sha256: str = ""                     # hash del file d'origine
    created_at: str = ""                        # ISO 8601
    updated_at: str = ""
    status: DocStatus = "in_coda"
    progress: float = 0.0                       # 0..1
    status_message: str = ""
    error: str | None = None

    is_foglio_firma: bool = True
    engine: str | None = None
    model: str | None = None
    header: Header = Field(default_factory=Header)
    rows: list[DayRow] = Field(default_factory=empty_rows)
    anomalies: list[Anomaly] = Field(default_factory=list)
    totals: Totals = Field(default_factory=Totals)
    ocr_notes: str | None = None
    confidence: float | None = None
    usage: Usage = Field(default_factory=Usage)

    user_verified: bool = False                 # l'utente ha confermato il documento
    user_edited: list[str] = Field(default_factory=list)  # es. ["header.operatore", "rows.3.eff_uscita"]
    # Valori originali letti dall'OCR per i campi poi modificati dall'utente
    # (chiave come in user_edited) - servono per la tracciabilita' nell'Excel.
    ocr_originali: dict[str, str | None] = Field(default_factory=dict)

    def apply_extraction(self, result: ExtractionResult) -> None:
        self.is_foglio_firma = result.is_foglio_firma
        self.header = result.header
        rows = {r.giorno: r for r in result.rows if 1 <= r.giorno <= 31}
        self.rows = [rows.get(g, DayRow(giorno=g)) for g in range(1, 32)]
        self.ocr_notes = result.ocr_notes
        self.confidence = result.confidence
        self.engine = result.engine
        self.model = result.model
        self.usage = result.usage

    def display_name(self) -> str:
        op = (self.header.operatore or "").strip()
        if self.header.mese and self.header.anno:
            periodo = f"{self.header.mese:02d}/{self.header.anno}"
        else:
            periodo = ""
        if op:
            return f"{op} {periodo}".strip()
        suffix = f" (pag. {self.source_page})" if self.page_count > 1 else ""
        return f"{self.source_file}{suffix}"
